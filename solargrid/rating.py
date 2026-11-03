"""Ślepe porównanie dwóch modeli na domach z Małopolski (strona /ocena w app.py).

    python -m solargrid.rating build --models bdappv_v1 mp_v5 --n 100 --out labels/ocena_v1
    python -m solargrid.rating summary labels/ocena_v1

Każdy dom: samo zdjęcie + maski obu modeli (przycięte do budynku jak w estimate) w losowej
kolejności A/B. Kolejność jest w items.json, strona dostaje ją bez nazw modeli.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import cv2
import numpy as np
import rasterio.features
from shapely.geometry import box

from . import buildings, estimate, labels, ortho
from .predict import Segmenter

CONTEXT_M = 12.0
MIN_HOUSE_M2 = 60.0
OSM_FRAC = 0.7  # reszta to losowe budynki (fałszywe wykrycia)


def sample_buildings(n: int, seed: int = 0) -> list[tuple[str, object]]:
    rng = random.Random(seed)
    idx = ortho.index()
    covered = idx[idx["path"].notna() & (idx["piksel"] <= 0.07)]
    gdf = buildings._gdf()
    out, seen = [], set()

    def add(kind, geom):
        hit = gdf.iloc[gdf.sindex.query(geom, predicate="intersects")]
        hit = hit[hit.area >= MIN_HOUSE_M2]
        if hit.empty:
            return
        i = hit.area.idxmax()
        if i not in seen and covered.intersects(hit.geometry[i]).any():
            seen.add(i)
            out.append((kind, hit.geometry[i]))

    pts = [o["geom"] for o in labels.load_osm() if o["geom"].geom_type == "Point"]
    rng.shuffle(pts)
    for p in pts:
        if len(out) >= round(n * OSM_FRAC):
            break
        add("osm_pv", p.buffer(1.0))
    sheets = list(covered.geometry)
    while len(out) < n:
        minx, miny, maxx, maxy = rng.choice(sheets).bounds
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        add("losowy", box(x - 30, y - 30, x + 30, y + 30))
    rng.shuffle(out)
    return out


def build(models: list[str], n: int, out: Path, device: str):
    (out / "img").mkdir(parents=True, exist_ok=True)
    segs = {m: Segmenter(Path("runs") / m / "best.pt", device) for m in models}
    rng = random.Random(1)
    items = []
    for k, (kind, b) in enumerate(sample_buildings(n)):
        crop = ortho.crop(b, buffer_m=CONTEXT_M)
        if crop is None:
            continue
        rgb, tr, meta = crop
        on_roof = rasterio.features.geometry_mask(
            [b.buffer(estimate.BUILDING_PAD_M)], out_shape=rgb.shape[:2], transform=tr, invert=True)
        order = models[:]
        rng.shuffle(order)
        item = {"id": f"d{k:03d}", "kind": kind, "models": order, "data_nalotu": meta["data_nalotu"],
                "centroid": [b.centroid.x, b.centroid.y]}
        empty = np.zeros(rgb.shape[:2], bool)
        cv2.imwrite(str(out / "img" / f"{item['id']}_0.jpg"),
                    _decode(estimate.preview(rgb, tr, b, empty)["zdjecie"]))
        for j, m in enumerate(order):
            mask = segs[m].mask(rgb, meta["piksel_m"]) & on_roof
            item[f"area_{j}"] = round(float(mask.sum() * abs(tr.a * tr.e)), 1)
            cv2.imwrite(str(out / "img" / f"{item['id']}_{'ab'[j]}.jpg"),
                        _decode(estimate.preview(rgb, tr, b, mask)["zdjecie_panele"]))
        items.append(item)
        print(item["id"], kind, item["area_0"], item["area_1"], flush=True)
    (out / "items.json").write_text(json.dumps(items, indent=1))
    print(f"{len(items)} domów -> {out}")


def _decode(data_url: str) -> np.ndarray:
    import base64
    return cv2.imdecode(np.frombuffer(base64.b64decode(data_url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)


def summary(out: Path):
    items = {it["id"]: it for it in json.loads((out / "items.json").read_text())}
    votes = {}
    path = out / "ratings.jsonl"
    for line in path.open() if path.exists() else []:
        r = json.loads(line)
        votes[r["id"]] = r["vote"]  # ostatni głos wygrywa
    wins = collections.Counter()
    by_kind = collections.defaultdict(collections.Counter)
    for i, v in votes.items():
        it = items[i]
        key = {"a": f"lepszy: {it['models'][0]}", "b": f"lepszy: {it['models'][1]}"}.get(v, v)
        wins[key] += 1
        by_kind[it["kind"]][key] += 1
    print(f"ocenionych: {len(votes)}/{len(items)}")
    for k, c in wins.most_common():
        print(f"  {k:28s} {c}")
    for kind, c in by_kind.items():
        print(f"  [{kind}] " + ", ".join(f"{k}: {v}" for k, v in c.most_common()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--models", nargs=2, required=True)
    b.add_argument("--n", type=int, default=100)
    b.add_argument("--out", type=Path, default=ortho.DATA / "labels" / "ocena_v1")
    b.add_argument("--device", default="cuda:0")
    s = sub.add_parser("summary")
    s.add_argument("out", type=Path, nargs="?", default=ortho.DATA / "labels" / "ocena_v1")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.models, args.n, args.out, args.device)
    else:
        summary(args.out)


if __name__ == "__main__":
    main()
