"""Adres -> budynek -> wykryte panele -> szacowana moc instalacji.

    python -m solargrid.estimate "Niepołomice, Zabierzowska 12" --ckpt runs/v1/best.pt
"""
from __future__ import annotations

import argparse
import base64
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import rasterio.features
from pyproj import Transformer
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform

from . import align, gugik, ortho, roof
from .predict import Segmenter, polygons

BUILDING_PAD_M = 3.0  # okapy + przesunięcie dachu na zdjęciu względem obrysu EGiB (do kilku m)
ALIGNED_PAD_M = 0.75  # obrys dopasowany do dachu (align.py): margines tylko na niedokładność
CONTEXT_M = 15.0  # kontekst wokół budynku dla modelu
# Fragmenty < 4 m² (mniej niż ~2 panele) to głównie świetliki i ciemne łaty dachu.
# Ocena 100 domów (28.10.2026): fałszywe wykrycia na 9 zamiast 13 z 30 budynków bez PV, 0 utraconych instalacji.
MIN_PART_M2 = 4.0
PREVIEW_PX = 720  # dłuższy bok podglądu w karcie wyniku

_to_wgs = Transformer.from_crs(2180, 4326, always_xy=True).transform


def to_geojson(geom):
    return mapping(shp_transform(_to_wgs, geom))


def _jpeg(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def preview(rgb: np.ndarray, tr, building, mask: np.ndarray) -> dict:
    """Zdjęcie, na którym liczył model: surowe i z nałożonymi panelami i obrysem budynku."""
    scale = min(1.0, PREVIEW_PX / max(rgb.shape[:2]))
    size = (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale))
    img = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    ov = img.copy()
    ov[m > 0] = (0.45 * ov[m > 0] + 0.55 * np.array([255, 59, 48])).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cs, -1, (255, 255, 255), 2)
    inv = ~tr
    parts = getattr(building, "geoms", [building])  # EGiB/OSM bywa MultiPolygon
    rings = [(np.array([inv * xy for xy in p.exterior.coords]) * scale).round().astype(np.int32) for p in parts]
    cv2.polylines(ov, rings, True, (10, 132, 255), 2)
    return {"zdjecie": _jpeg(img), "zdjecie_panele": _jpeg(ov)}


def drop_small(mask: np.ndarray, px_area: float, min_m2: float = MIN_PART_M2) -> np.ndarray:
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.nonzero(stats[:, cv2.CC_STAT_AREA] * px_area >= min_m2)[0]
    return np.isin(cc, keep[keep > 0])


_sam_lock = threading.Lock()  # SAM2ImagePredictor trzyma obraz w stanie — jedno zapytanie naraz
sam_tile = None  # kafelek /popraw, którego obraz siedzi w predyktorze SAM (app.popraw_sam)


def estimate_building(seg: Segmenter, building, res: float | None = None, sam=None) -> dict:
    crop = ortho.crop(building, buffer_m=CONTEXT_M, res=res)
    if crop is None:
        return {"status": "brak_ortofoto"}
    rgb, tr, meta = crop
    mask = seg.mask(rgb, meta["piksel_m"])
    pad, extra = BUILDING_PAD_M, {}
    if sam is not None:
        global sam_tile
        with _sam_lock:
            sam_tile = None
            roof_geom, info = align.align(rgb, tr, building, sam)
        extra["dopasowanie_obrysu"] = info
        if info["dopasowano"]:
            building, pad = roof_geom, ALIGNED_PAD_M
            extra["budynek"] = to_geojson(building)
    on_roof = rasterio.features.geometry_mask(
        [building.buffer(pad)], out_shape=mask.shape, transform=tr, invert=True)
    mask &= on_roof
    mask = drop_small(mask, abs(tr.a * tr.e))
    panels = polygons(mask, tr)
    metrics = roof.panel_metrics(mask, tr, building)
    return {"status": "ok" if panels else "brak_pv", "ortofoto": meta, **metrics, **extra,
            "panele": [to_geojson(p) for p in panels], **preview(rgb, tr, building, mask)}


def estimate_address(seg: Segmenter, address: str, sam=None) -> dict:
    hits = gugik.geocode(address)
    if not hits:
        return {"status": "nie_znaleziono_adresu", "adres": address}
    a = hits[0]
    found = gugik.building_near(a.x, a.y)
    if not found:
        return {"status": "brak_budynku", "adres": a.label}
    bid, building = found
    lon, lat = _to_wgs(a.x, a.y)
    return {"adres": a.label, "punkt": [lon, lat], "budynek_id": bid,
            "budynek": to_geojson(building), "powierzchnia_budynku_m2": round(building.area, 1),
            **estimate_building(seg, building, sam=sam)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("address")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    res = estimate_address(Segmenter(args.ckpt, args.device), args.address)
    for k in ("panele", "budynek", "zdjecie", "zdjecie_panele"):
        res.pop(k, None)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
