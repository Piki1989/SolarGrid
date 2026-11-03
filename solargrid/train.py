"""Trening segmentacji PV (2× GPU, DDP).

Źródła danych to katalogi z images/ (JPG/PNG) i masks/ (PNG: 0 tło, 1 PV, 255 ignoruj)
oraz znaną rozdzielczością [m/px]. Każdy kafelek jest losowo przeskalowany do docelowej
rozdzielczości z TARGET_GSD — model ma działać od 3 cm (lotnicze) do ~30 cm (satelitarne).

    torchrun --nproc_per_node 2 -m solargrid.train --src osm_sam:0.05 --src bdappv_google:0.1 ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

IGNORE = 255
TARGET_GSD = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3]
CROP = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def list_source(spec: str) -> list[tuple[str, str, float]]:
    """'ścieżka:gsd' -> [(obraz, maska, gsd)]; ścieżki względne od --root."""
    path, gsd = spec.rsplit(":", 1)
    root = Path(path)
    out = []
    for m in sorted((root / "masks").glob("*.png")):
        for ext in (".jpg", ".png"):
            img = root / "images" / (m.stem + ext)
            if img.exists():
                out.append((str(img), str(m), float(gsd)))
                break
    return out


def spatial_split(items, val_frac: float, seed: int = 0):
    """Podział po kafelkach 2 km (z records.jsonl gdy jest), inaczej po pliku — bez przecieków
    między sąsiednimi kafelkami tej samej instalacji."""
    groups: dict[str, list] = {}
    bounds_cache: dict[str, dict] = {}
    for it in items:
        root = Path(it[1]).parent.parent
        if root not in bounds_cache:
            rec = root / "records.jsonl"
            bounds_cache[root] = ({r["tile"]: r["bounds"] for r in map(json.loads, rec.open())}
                                  if rec.exists() else {})
        b = bounds_cache[root].get(Path(it[1]).stem)
        key = f"{root}:{int(b[0] // 2000)}:{int(b[1] // 2000)}" if b else f"{root}:{Path(it[1]).stem}"
        groups.setdefault(key, []).append(it)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_val = max(1, int(len(keys) * val_frac)) if val_frac > 0 else 0
    val = [it for k in keys[:n_val] for it in groups[k]]
    train = [it for k in keys[n_val:] for it in groups[k]]
    return train, val


class PVDataset(Dataset):
    def __init__(self, items, train: bool, gsd: float | None = None):
        self.items, self.train, self.fixed_gsd = items, train, gsd
        self.aug = A.Compose([
            A.D4(),
            A.RandomBrightnessContrast(0.25, 0.25, p=0.7),
            A.HueSaturationValue(8, 20, 15, p=0.5),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.ImageCompression(quality_range=(40, 90))], p=0.4),
            A.GaussNoise(std_range=(0.01, 0.04), p=0.2),
        ]) if train else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_p, mask_p, gsd = self.items[i]
        img = cv2.imread(img_p, cv2.IMREAD_COLOR_RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if mask.max() == 255 and not (mask == 1).any():  # maski 0/255 (np. BDAPPV)
            mask = (mask > 127).astype(np.uint8)
        if self.fixed_gsd:
            target = max(self.fixed_gsd, gsd)
        else:
            target = random.choice([g for g in TARGET_GSD if g >= gsd] or [gsd])
        scale = gsd / target
        if abs(scale - 1) > 1e-3:
            size = (max(1, round(img.shape[1] * scale)), max(1, round(img.shape[0] * scale)))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        img, mask = self._crop(img, mask)
        if self.aug:
            out = self.aug(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        x = torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).astype(np.float32))
        return x, torch.from_numpy(mask.astype(np.int64))

    def _crop(self, img, mask):
        h, w = mask.shape
        ph, pw = max(0, CROP - h), max(0, CROP - w)
        if ph or pw:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=IGNORE)
            h, w = mask.shape
        if self.train:
            # Połowa próbek wycentrowana na PV — w kafelkach PV to zwykle <5% pikseli.
            ys, xs = np.nonzero(mask == 1)
            if len(ys) and random.random() < 0.5:
                j = random.randrange(len(ys))
                y0 = int(np.clip(ys[j] - CROP // 2 + random.randint(-CROP // 4, CROP // 4), 0, h - CROP))
                x0 = int(np.clip(xs[j] - CROP // 2 + random.randint(-CROP // 4, CROP // 4), 0, w - CROP))
            else:
                y0, x0 = random.randint(0, h - CROP), random.randint(0, w - CROP)
        else:
            y0, x0 = (h - CROP) // 2, (w - CROP) // 2
        return img[y0:y0 + CROP, x0:x0 + CROP], mask[y0:y0 + CROP, x0:x0 + CROP]


def build_model(arch: str, encoder: str):
    return getattr(smp, arch)(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=1)


def loss_fn(logits, target):
    logits = logits[:, 0]
    valid = target != IGNORE
    t = (target == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits[valid], t[valid])
    p = torch.sigmoid(logits) * valid
    inter = (p * t).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + (t * valid).sum() + 1)
    return bce + dice


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = torch.zeros((), device=device)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x)[:, 0] > 0
        valid = y != IGNORE
        gt = y == 1
        tp = tp + (pred & gt & valid).sum()
        fp = fp + (pred & ~gt & valid).sum()
        fn = fn + (~pred & gt & valid).sum()
    stats = torch.stack([tp, fp, fn]).float()
    if dist.is_initialized():
        dist.all_reduce(stats)
    tp, fp, fn = stats.tolist()
    model.train()
    return {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True, help="katalog:gsd, można wielokrotnie")
    ap.add_argument("--val-src", action="append", default=[], help="osobny zbiór walidacyjny katalog:gsd")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", default="UnetPlusPlus")
    ap.add_argument("--encoder", default="tu-convnext_base.fb_in22k_ft_in1k")
    ap.add_argument("--init", type=Path, help="checkpoint do dostrojenia")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=12, help="na GPU")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        # thebrain: wtyczka RoCE (bnxt_re) i interfejs link-local BMC powodują SIGSEGV w NCCL.
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        dist.init_process_group("nccl")
        rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    else:
        rank, local = 0, 0
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    main_proc = rank == 0

    items = [it for s in args.src for it in list_source(s)]
    train_items, val_items = spatial_split(items, args.val_frac)
    for s in args.val_src:
        val_items += list_source(s)
    if main_proc:
        args.out.mkdir(parents=True, exist_ok=True)
        print(f"trening {len(train_items)}  walidacja {len(val_items)}")
        (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))

    train_ds = PVDataset(train_items, train=True)
    # Walidacja na stałej rozdzielczości 0.1 m — porównywalna między epokami.
    val_ds = PVDataset(val_items, train=False, gsd=0.1)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None
    train_dl = DataLoader(train_ds, args.bs, sampler=train_sampler, shuffle=train_sampler is None,
                          num_workers=args.workers, pin_memory=True, drop_last=True,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, args.bs, sampler=val_sampler, num_workers=args.workers // 2, pin_memory=True)

    model = build_model(args.arch, args.encoder).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
    model = model.to(memory_format=torch.channels_last)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(train_dl)
    warm = max(1, min(1000, steps // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / max(steps, 1)))))

    best = -1.0
    log = (args.out / "log.jsonl").open("a") if main_proc else None
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        total, n = 0.0, 0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
        metrics = evaluate(model, val_dl, device)
        if main_proc:
            row = {"epoch": epoch, "loss": total / max(n, 1), **metrics, "lr": sched.get_last_lr()[0]}
            print(json.dumps(row))
            log.write(json.dumps(row) + "\n")
            log.flush()
            state = {"model": (model.module if ddp else model).state_dict(),
                     "arch": args.arch, "encoder": args.encoder, "epoch": epoch, "metrics": metrics}
            torch.save(state, args.out / "last.pt")
            if metrics["iou"] > best:
                best = metrics["iou"]
                torch.save(state, args.out / "best.pt")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
