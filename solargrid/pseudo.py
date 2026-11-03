"""Pseudoetykiety z modelu-nauczyciela na kafelkach z Małopolski.

Etykiety OSM + SAM zaznaczają często całe połacie zamiast paneli; model z pretreningu
(BDAPPV, ręczne maski) rysuje panele precyzyjnie, ale nie zna polskich zdjęć. Bierzemy
jego pewne wykrycia na dachach jako PV, a niepewne miejsca i miejsca, gdzie OSM mówi
o instalacji, a nauczyciel jej nie widzi, oznaczamy jako ignoruj.

    python -m solargrid.pseudo --teacher runs/bdappv_v1/best.pt \\
        --src labels/osm_sam_r3 --out labels/teacher_r4 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from rasterio.transform import from_bounds
from tqdm import tqdm

from . import buildings
from .labels import IGNORE
from .predict import Segmenter

P_HI = 0.5  # pewne PV
P_LO = 0.2  # poniżej: pewne tło
ROOF_PAD_M = 3.0  # obrys EGiB vs dach na zdjęciu (przesunięcie radialne) + okapy
MIN_PV_PX = 400  # 1 m² przy 5 cm — mniejsze plamy to szum


def pseudo_label(prob: np.ndarray, old: np.ndarray, roofs: np.ndarray, osm_ignore: bool = True) -> np.ndarray:
    pv = prob > P_HI
    n, cc, stats, _ = cv2.connectedComponentsWithStats(pv.astype(np.uint8), connectivity=8)
    small = np.isin(cc, np.nonzero(stats[:, cv2.CC_STAT_AREA] < MIN_PV_PX)[0]) & pv
    label = np.zeros(prob.shape, np.uint8)
    label[pv & roofs & ~small] = 1
    label[(prob > P_LO) & (label == 0)] = IGNORE  # niepewne, poza dachem, drobne plamy
    # OSM/SAM mówi PV, nauczyciel nie: może to polski typ instalacji, którego nie zna.
    # Bez tego (osm_ignore=False): w strefach ignoruj model nie dostaje kary i mp_v4
    # rozlewał tam okrągłe plamy na całe dachy i trawniki.
    if osm_ignore:
        label[(old != 0) & (label == 0)] = IGNORE
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True, help="katalog etykiet OSM + SAM (images, masks, records)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-osm-ignore", action="store_true", help="tło tam, gdzie OSM mówi PV, a nauczyciel nie")
    args = ap.parse_args()
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    seg = Segmenter(args.teacher, f"cuda:{args.gpu}")
    recs = [json.loads(line) for line in (args.src / "records.jsonl").open()]
    stats = {"pv_px": 0, "old_pv_px": 0}
    with (args.out / "records.jsonl").open("w") as out:
        for r in tqdm(recs):
            t = r["tile"]
            img_p = args.src / "images" / f"{t}.jpg"
            rgb = cv2.imread(str(img_p), cv2.IMREAD_COLOR_RGB)
            old = cv2.imread(str(args.src / "masks" / f"{t}.png"), cv2.IMREAD_GRAYSCALE)
            h, w = old.shape
            tr = from_bounds(*r["bounds"], w, h)
            roofs = buildings.roof_mask(r["bounds"], tr, (h, w), pad_m=ROOF_PAD_M)
            label = pseudo_label(seg.prob(rgb, r["res"], tta=True), old, roofs, not args.no_osm_ignore)
            shutil.copyfile(img_p, args.out / "images" / f"{t}.jpg")
            cv2.imwrite(str(args.out / "masks" / f"{t}.png"), label)
            stats["pv_px"] += int((label == 1).sum())
            stats["old_pv_px"] += int((old == 1).sum())
            out.write(json.dumps({**r, "pv_frac": round(float((label == 1).mean()), 4),
                                  "teacher": str(args.teacher)}) + "\n")
    px = recs[0]["res"] ** 2 if recs else 0
    print(f"PV: nauczyciel {stats['pv_px'] * px:.0f} m², OSM+SAM {stats['old_pv_px'] * px:.0f} m²")


if __name__ == "__main__":
    main()
