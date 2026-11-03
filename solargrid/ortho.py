"""Indeks pobranych arkuszy ortofotomapy i wycinanie fragmentów w EPSG:2180."""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import requests
from lxml import etree
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, box

DATA = Path.home() / "datasets" / "malopolska_pv"
INDEX_PATH = DATA / "meta" / "ortho_index.gpkg"
GML = "{http://www.opengis.net/gml/3.2}"


def _parse_wfs(path: Path) -> list[dict]:
    rows = []
    for member in etree.parse(str(path)).getroot():
        if not isinstance(member.tag, str) or not member.tag.endswith("member"):
            continue
        feat = next(c for c in member if isinstance(c.tag, str))
        pos = [float(v) for v in next(feat.iter(f"{GML}posList")).text.split()]
        row = {"geometry": Polygon(zip(pos[1::2], pos[0::2]))}  # WFS podaje (N, E)
        for c in feat:
            if not isinstance(c.tag, str):
                continue
            key = c.tag.split("}")[1]
            if key in ("akt_data", "dt_pzgik"):
                tp = c.find(f".//{GML}timePosition")
                row[key] = tp.text if tp is not None else None
            elif key not in ("msGeometry", "boundedBy"):
                row[key] = (c.text or "").strip()
        rows.append(row)
    return rows


LENGTHS_PATH = DATA / "meta" / "content_length.json"


def _active_downloads() -> set[str]:
    """Nazwy plików, które właśnie pobiera jakiś wget (Linux /proc)."""
    names = set()
    for cmd in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = cmd.read_bytes().split(b"\0")
        except OSError:
            continue
        if args and args[0].endswith(b"wget"):
            names.update(a.decode().rsplit("/", 1)[-1] for a in args[1:] if a.startswith(b"http"))
    return names


def _content_length(url: str) -> int | None:
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        return int(r.headers["Content-Length"])
    except (requests.RequestException, KeyError, ValueError):
        return None


def _load_lengths() -> dict:
    try:
        return json.loads(LENGTHS_PATH.read_text())
    except (OSError, ValueError):
        return {}


def _save_lengths(lengths: dict) -> None:
    lengths = {k: v for k, v in lengths.items() if v is not None}  # błędy sieci: spróbuj następnym razem
    tmp = LENGTHS_PATH.with_name(f".{LENGTHS_PATH.name}.{os.getpid()}")
    tmp.write_text(json.dumps(lengths))
    os.replace(tmp, LENGTHS_PATH)


def build_index() -> gpd.GeoDataFrame:
    """Łączy skorowidze WFS z plikami na dysku. Zapisuje meta/ortho_index.gpkg."""
    rows = []
    for wfs in sorted((DATA / "meta").glob("wfs_*.xml")):
        rows += _parse_wfs(wfs)
    gdf = gpd.GeoDataFrame(rows, crs=2180)
    gdf = gdf[gdf["kolor"] == "RGB"].copy()
    gdf["piksel"] = gdf["piksel"].astype(float)
    gdf["rok"] = gdf["akt_rok"].astype(int)
    gdf["plik"] = gdf["url_do_pobrania"].str.rsplit("/", n=1).str[1]
    # Plik w trakcie pobierania (wget -c pisze w miejscu) jest krótszy niż w skorowidzu (MiB);
    # >105% = dopisany dwa razy (dwa procesy wget na jednym pliku) — uszkodzony.
    # Skorowidz bywa niedokładny o ~5% — wtedy rozstrzyga Content-Length z serwera.
    sizes = {p.name: (str(p), p.stat().st_size) for p in (DATA / "ortho").glob("*/*.tif")}
    expected = gdf["wlk_pliku_mb"].astype(float) * 1024 * 1024
    active = _active_downloads()
    lengths = _load_lengths()

    def complete(name, url, exp):
        hit = sizes.get(name)
        if not hit:
            return None
        if 0.99 * exp <= hit[1] <= 1.05 * exp:
            return hit[0]
        if name in active or not 0.85 * exp <= hit[1] <= 1.15 * exp:
            return None
        if name not in lengths:
            lengths[name] = _content_length(url)
        return hit[0] if hit[1] == lengths[name] else None

    gdf["path"] = [complete(n, u, e) for n, u, e in zip(gdf["plik"], gdf["url_do_pobrania"], expected)]
    _save_lengths(lengths)
    # Kilka procesów (ready, labels, detect) przebudowuje indeks — zapis atomowy.
    tmp = INDEX_PATH.with_name(f".{INDEX_PATH.stem}.{os.getpid()}.gpkg")
    gdf.to_file(tmp, driver="GPKG", layer="ortho_index")
    os.replace(tmp, INDEX_PATH)
    global _index
    _index = gdf
    return gdf


_index: gpd.GeoDataFrame | None = None


def index() -> gpd.GeoDataFrame:
    global _index
    if _index is None:
        _index = gpd.read_file(INDEX_PATH) if INDEX_PATH.exists() else build_index()
    return _index


def sheets_for(geom, max_pixel: float = 0.25) -> gpd.GeoDataFrame:
    """Arkusze pokrywające geometrię, posortowane: najnowsze, potem najdokładniejsze."""
    idx = index()
    hits = idx[idx.intersects(geom) & idx["path"].notna() & (idx["piksel"] <= max_pixel)]
    return hits.sort_values(["akt_data", "piksel"], ascending=[False, True])


def read_window(path: str, bounds: tuple[float, float, float, float], res: float) -> np.ndarray | None:
    """Wycina RGB (H, W, 3) uint8 w EPSG:2180 o rozdzielczości res [m]. None gdy poza arkuszem/uszkodzony."""
    minx, miny, maxx, maxy = bounds
    w, h = round((maxx - minx) / res), round((maxy - miny) / res)
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, w, h)
    try:
        with rasterio.open(path) as src, WarpedVRT(
            src, crs="EPSG:2180", transform=transform, width=w, height=h,
            resampling=Resampling.bilinear, nodata=0,
        ) as vrt:
            arr = vrt.read([1, 2, 3], window=from_bounds(minx, miny, maxx, maxy, transform))
    except rasterio.errors.RasterioIOError:
        return None
    return arr.transpose(1, 2, 0)


def crop(geom, buffer_m: float = 5.0, res: float | None = None):
    """Najnowszy dostępny wycinek wokół geometrii.

    Zwraca (rgb, transform, meta) lub None. Gdy res=None, używa natywnej rozdzielczości
    najnowszego arkusza z danymi. Arkusze na skraju nalotu bywają w większości puste
    (czy_ark_wypelniony=NIE) — brakujące piksele uzupełniamy ze starszych nalotów;
    meta opisuje nalot, z którego pochodzi większość pikseli.
    """
    return crop_bounds(geom.buffer(buffer_m).bounds, res)


WMS_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution"
WMS_MAX_PX = 2048
WMS_MIN_RES = 0.05  # model i tak zmniejsza do 10 cm; 3 cm z WMS to tylko wolniejsze pobieranie


def _wms_window(bounds, w: int, h: int) -> np.ndarray:
    minx, miny, maxx, maxy = bounds
    r = requests.get(WMS_URL, timeout=60, params={
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.1.1", "LAYERS": "Raster", "STYLES": "",
        "SRS": "EPSG:2180", "BBOX": f"{minx},{miny},{maxx},{maxy}", "WIDTH": w, "HEIGHT": h,
        "FORMAT": "image/jpeg"})
    r.raise_for_status()
    if not r.headers.get("Content-Type", "").startswith("image/"):
        raise requests.RequestException(f"WMS: {r.text[:200]}")
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]


def crop_wms(bounds: tuple[float, float, float, float], res: float | None = None, max_pixel: float = 0.25):
    """Jak crop_bounds(), ale z WMS GUGiK (bez arkuszy na dysku, np. na laptopie).

    WMS HighResolution pokazuje najnowszy nalot w natywnej rozdzielczości; meta (data nalotu,
    piksel) bierzemy ze skorowidza: najnowszy arkusz pokrywający środek prostokąta.
    """
    minx, miny, maxx, maxy = bounds
    idx = index()
    hits = idx[idx.intersects(box(*bounds).centroid) & (idx["piksel"] <= max_pixel)]
    hits = hits.sort_values(["akt_data", "piksel"], ascending=[False, True])
    if hits.empty:
        return None
    date, pix = hits.iloc[0]["akt_data"], float(hits.iloc[0]["piksel"])
    res = res or max(pix, WMS_MIN_RES)
    w, h = round((maxx - minx) / res), round((maxy - miny) / res)
    out = np.zeros((h, w, 3), np.uint8)
    # Duże budynki: kilka zapytań po ≤ WMS_MAX_PX (granica serwera).
    for y0 in range(0, h, WMS_MAX_PX):
        for x0 in range(0, w, WMS_MAX_PX):
            tw, th = min(WMS_MAX_PX, w - x0), min(WMS_MAX_PX, h - y0)
            out[y0:y0 + th, x0:x0 + tw] = _wms_window(
                (minx + x0 * res, maxy - (y0 + th) * res, minx + (x0 + tw) * res, maxy - y0 * res), tw, th)
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, w, h)
    meta = {"data_nalotu": date, "piksel_zrodla_m": pix, "piksel_m": res, "pokrycie": 1.0,
            "naloty": [date], "zrodlo": "wms"}
    return out, transform, meta


def crop_bounds(bounds: tuple[float, float, float, float], res: float | None = None,
                max_pixel: float = 0.25):
    """Jak crop(), ale dla prostokąta (minx, miny, maxx, maxy) w EPSG:2180.

    SOLARGRID_ORTHO=wms: z WMS zamiast z dysku (laptop bez arkuszy). Bez automatycznego
    przełączania — etykiety i detekcja na serwerze mają czytać tylko pobrane arkusze.
    """
    if os.environ.get("SOLARGRID_ORTHO") == "wms":
        return crop_wms(bounds, res, max_pixel)
    minx, miny, maxx, maxy = bounds
    sheets = sheets_for(box(*bounds), max_pixel=max_pixel)
    out, filled, used = None, None, []
    for (date, pix), group in sheets.groupby(["akt_data", "piksel"], sort=False):
        r = res or float(pix)
        for path in group["path"]:
            arr = read_window(path, (minx, miny, maxx, maxy), r)
            if arr is None:
                continue
            # Puste obszary arkusza = 0, ale JPEG zostawia tam szum 1–3.
            has = arr.astype(np.uint16).sum(-1) > 15
            if out is None:
                out, filled = np.zeros_like(arr), np.zeros(has.shape, bool)
                res = r
            new = has & ~filled
            if new.any():
                out[new] = arr[new]
                filled |= new
                used.append((date, float(pix), int(new.sum())))
        if filled is not None and filled.mean() > 0.99:
            break
    if out is None or not filled.any():
        return None
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, out.shape[1], out.shape[0])
    date, pix, _ = max(used, key=lambda u: u[2])
    meta = {"data_nalotu": date, "piksel_zrodla_m": pix, "piksel_m": res,
            "pokrycie": round(float(filled.mean()), 3), "naloty": sorted({u[0] for u in used}, reverse=True)}
    return out, transform, meta
