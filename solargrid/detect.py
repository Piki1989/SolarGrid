"""Wykrywanie PV na całym obszarze (siatka 500 m) -> detections/pv_shard<s>_<paczka>.gpkg.

    python -m solargrid.detect --ckpt runs/v1/best.pt --shard 0 --nshards 2 --device cuda:0 &
    python -m solargrid.detect --ckpt runs/v1/best.pt --shard 1 --nshards 2 --device cuda:1
    python -m solargrid.detect --merge
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm

from . import buildings, ortho
from .predict import MODEL_GSD, Segmenter, polygons
from .estimate import MIN_PART_M2
from .roof import KWP_PER_M2

CELL_M = 500.0
OUT = ortho.DATA / "detections"
# Bez korekty nachylenia (brak NMPT w trybie masowym): typowy dach 35° -> rzut × 1.22.
TYPICAL_TILT_CORR = 1.22
CHUNK = 200  # komórek na plik: przerwany przebieg wznawia się od pierwszej brakującej paczki


def region():
    woj = gpd.read_file(ortho.DATA / "meta" / "woj.xml")
    return woj[woj["JPT_NAZWA_"] == "małopolskie"].to_crs(2180).geometry.iloc[0]


def cells(area) -> list[tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = area.bounds
    out = []
    for x in np.arange(minx, maxx, CELL_M):
        for y in np.arange(miny, maxy, CELL_M):
            b = (x, y, x + CELL_M, y + CELL_M)
            if area.intersects(box(*b)):
                out.append(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.merge:
        parts = [gpd.read_file(p) for p in sorted(OUT.glob("pv_shard*_*.gpkg"))]
        gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=2180)
        gdf.to_file(OUT / "pv.gpkg", driver="GPKG")
        print(f"{len(gdf)} wykryć, {gdf['moc_kwp'].sum() / 1000:.1f} MWp -> {OUT / 'pv.gpkg'}")
        return

    seg = Segmenter(args.ckpt, args.device)
    covered = unary_union(list(ortho.index().query("path.notna()").geometry))
    todo = [c for i, c in enumerate(cells(region().intersection(covered))) if i % args.nshards == args.shard]
    for k in range(0, len(todo), CHUNK):
        path = OUT / f"pv_shard{args.shard}_{k // CHUNK:04d}.gpkg"
        if path.exists():
            continue
        rows = detect_cells(seg, todo[k:k + CHUNK], f"shard {args.shard} {k}/{len(todo)}")
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=2180) if rows else \
            gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=2180)
        gdf.to_file(path.with_suffix(".tmp"), driver="GPKG")
        path.with_suffix(".tmp").rename(path)
    print(f"shard {args.shard}: gotowe")


def detect_cells(seg: Segmenter, todo, desc: str) -> list[dict]:
    rows = []
    for b in tqdm(todo, desc=desc, mininterval=60):
        if buildings.in_bounds(b).empty:  # ~połowa regionu to lasy i pola
            continue
        sheets = ortho.sheets_for(box(*b))
        if sheets.empty:
            continue
        # Od razu w rozdzielczości modelu: odczyt w pełnej (3–5 cm, do 16k × 16k px) tylko po to,
        # żeby poznać piksel, był głównym kosztem w gęstej zabudowie (30.10: ~13 s na komórkę).
        gsd = max(float(sheets.iloc[0]["piksel"]), MODEL_GSD)
        crop = ortho.crop_bounds(b, res=gsd)
        if crop is None:
            continue
        rgb, tr, meta = crop
        roofs = buildings.roof_mask(b, tr, rgb.shape[:2])
        if not roofs.any():
            continue
        for p in polygons(seg.mask(rgb, gsd, tta=False) & roofs, tr, min_area_m2=MIN_PART_M2):
            area = p.area * TYPICAL_TILT_CORR
            rows.append({"geometry": p, "powierzchnia_rzut_m2": round(p.area, 1),
                         "moc_kwp": round(area * KWP_PER_M2["mid"], 1),
                         "data_nalotu": meta["data_nalotu"], "piksel_m": meta["piksel_zrodla_m"]})
    return rows

if __name__ == "__main__":
    main()
