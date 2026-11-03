"""Klienci usług GUGiK: geokoder adresów (UUG), budynki (ULDK), NMPT (WCS).

Wszystkie współrzędne w EPSG:2180 (PL-1992), kolejność (x=easting, y=northing).
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import rasterio
import requests
from shapely import wkt
from shapely.geometry import Polygon

UUG_URL = "https://services.gugik.gov.pl/uug/"
ULDK_URL = "https://uldk.gugik.gov.pl/"
DSM_WCS_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMPT/GRID1/WCS/DigitalSurfaceModel"
TIMEOUT = 60

_session = requests.Session()
_session.headers["User-Agent"] = "SolarGrid/0.1 (AGH KIS)"
# Usługi GUGiK czasem zrywają połączenie bez odpowiedzi — ponawiamy.
_retry = requests.adapters.Retry(total=3, connect=3, read=3, backoff_factor=1,
                                 status_forcelist=(502, 503, 504), allowed_methods=("GET",))
_session.mount("https://", requests.adapters.HTTPAdapter(max_retries=_retry))


@dataclass
class Address:
    label: str
    x: float
    y: float
    teryt: str
    accuracy: float


def geocode(address: str) -> list[Address]:
    """Zwraca dopasowane adresy, najlepsze pierwsze. Pusta lista gdy brak."""
    r = _session.get(UUG_URL, params={"request": "GetAddress", "address": address}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    out = []
    for res in (data.get("results") or {}).values():
        parts = [res.get("city"), res.get("street"), res.get("number")]
        out.append(Address(
            label=" ".join(p for p in parts if p),
            x=float(res["x"]),
            y=float(res["y"]),
            teryt=res.get("teryt") or "",
            accuracy=float(res.get("accuracy") or 0),
        ))
    return sorted(out, key=lambda a: -a.accuracy)


def building_at(x: float, y: float) -> tuple[str, Polygon] | None:
    """Obrys budynku z EGiB zawierającego punkt (ULDK). None gdy brak budynku."""
    r = _session.get(ULDK_URL, params={
        "request": "GetBuildingByXY", "xy": f"{x},{y},2180", "result": "id,geom_wkt",
    }, timeout=TIMEOUT)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    if not lines or lines[0].strip() != "0" or len(lines) < 2:
        return None
    bid, geom = lines[1].split("|", 1)
    return bid, wkt.loads(geom.split(";", 1)[1])


def building_near(x: float, y: float, search_m: float = 15.0, step_m: float = 3.0) -> tuple[str, Polygon] | None:
    """Punkt adresowy bywa obok budynku — szukaj spiralnie w promieniu search_m."""
    hit = building_at(x, y)
    if hit:
        return hit
    r = step_m
    while r <= search_m:
        for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            hit = building_at(x + r * np.cos(ang), y + r * np.sin(ang))
            if hit:
                return hit
        r += step_m
    return None


def dsm(bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, rasterio.Affine]:
    """NMPT (wysokości pokrycia terenu, 0.5 m) dla prostokąta (minx, miny, maxx, maxy)."""
    # image/tiff z tej usługi jest kwantowany do pełnych metrów — x-aaigrid ma pełne float.
    minx, miny, maxx, maxy = bounds
    r = _session.get(DSM_WCS_URL, params=[
        ("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
        ("COVERAGEID", "DSM_PL-EVRF2007-NH"), ("FORMAT", "image/x-aaigrid"),
        ("SUBSET", f"x({minx},{maxx})"), ("SUBSET", f"y({miny},{maxy})"),
    ], timeout=TIMEOUT)
    r.raise_for_status()
    text = r.text
    start = text.find("ncols")
    if start < 0:
        raise RuntimeError(f"NMPT WCS: nieoczekiwana odpowiedź: {text[:300]}")
    body = text[start:].split("\n--wcs", 1)[0]  # odpowiedź multipart/mixed
    with rasterio.MemoryFile(io.BytesIO(body.encode()), ext=".asc") as mf, mf.open() as ds:
        return ds.read(1).astype(np.float32), ds.transform
