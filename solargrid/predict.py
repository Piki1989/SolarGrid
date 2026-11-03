"""Inferencja modelu PV na wycinku ortofoto (okno przesuwne + TTA)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio.features
import segmentation_models_pytorch as smp
import torch
from shapely.geometry import shape

from .train import CROP, MEAN, STD

MODEL_GSD = 0.1  # rozdzielczość, na której model działa najlepiej (walidacja)


class Segmenter:
    def __init__(self, ckpt: Path, device: str = "cuda:0", threshold: float = 0.5):
        state = torch.load(ckpt, map_location="cpu")
        self.model = getattr(smp, state["arch"])(encoder_name=state["encoder"], encoder_weights=None,
                                                 in_channels=3, classes=1)
        self.model.load_state_dict(state["model"])
        self.model.to(device).eval()
        self.device, self.threshold = device, threshold

    @torch.inference_mode()
    def prob(self, rgb: np.ndarray, gsd: float, tta: bool = True) -> np.ndarray:
        """Mapa prawdopodobieństwa PV w rozdzielczości wejścia."""
        h0, w0 = rgb.shape[:2]
        scale = gsd / max(gsd, MODEL_GSD)  # zdjęcia dokładniejsze niż 10 cm zmniejszamy
        img = rgb if scale == 1 else cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        ph, pw = (-h) % 32 + max(0, CROP - h), (-w) % 32 + max(0, CROP - w)
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        x = torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)).to(self.device)
        acc = torch.zeros(x.shape[-2:], device=self.device)
        cnt = torch.zeros_like(acc)
        step = CROP * 3 // 4
        H, W = x.shape[-2:]
        ys = sorted({min(y, H - CROP) for y in range(0, H, step)})
        xs = sorted({min(v, W - CROP) for v in range(0, W, step)})
        flips = [(), (-1,), (-2,), (-2, -1)] if tta else [()]
        with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16):
            for y in ys:
                for xx in xs:
                    patch = x[..., y:y + CROP, xx:xx + CROP]
                    p = 0
                    for f in flips:
                        out = self.model(patch.flip(f) if f else patch)
                        p = p + torch.sigmoid((out.flip(f) if f else out).float())[0, 0]
                    acc[y:y + CROP, xx:xx + CROP] += p / len(flips)
                    cnt[y:y + CROP, xx:xx + CROP] += 1
        prob = (acc / cnt).cpu().numpy()[:h, :w]
        return prob if scale == 1 else cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)

    def mask(self, rgb, gsd, tta=True) -> np.ndarray:
        return self.prob(rgb, gsd, tta) > self.threshold


def polygons(mask: np.ndarray, transform, min_area_m2: float = 1.0, simplify_m: float = 0.1):
    """Maska -> lista poligonów shapely w EPSG:2180."""
    out = []
    for geom, val in rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        p = shape(geom).simplify(simplify_m)
        if p.area >= min_area_m2:
            out.append(p)
    return out
