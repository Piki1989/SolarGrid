"""Geometria dachu z NMPT i przeliczenie powierzchni paneli na moc."""
from __future__ import annotations

import numpy as np
import rasterio.features
from shapely.geometry import Polygon

from . import gugik

# kWp na m² rzeczywistej powierzchni modułów. Moduły 400–460 W mają ~1.9–2.0 m²,
# starsze (2015–2020) 250–330 W przy ~1.65 m². BDAPPV (deklarowana moc / powierzchnia,
# 28 tys. instalacji): mediana 0.15 (2011–16), 0.18–0.19 (2019–21), p90 ~0.22–0.23.
# Boom PV w Polsce to 2019+, więc środek 0.20.
KWP_PER_M2 = {"low": 0.16, "mid": 0.20, "high": 0.23}


def slope_aspect(dem: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Nachylenie [°] i ekspozycja [° od północy, zgodnie z zegarem] z siatki wysokości."""
    dzdy, dzdx = np.gradient(dem, res)  # wiersze rosną na południe
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # kierunek spadku (E, N) = (-dz/dx, +dz/dwiersz); azymut = atan2(E, N)
    aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360.0) % 360.0
    return slope, aspect


def roof_slope_map(building: Polygon, pad_m: float = 2.0):
    """Mapa nachylenia/ekspozycji dachu (NMPT 0.5 m) przycięta do obrysu budynku."""
    bounds = building.buffer(pad_m).bounds
    dem, transform = gugik.dsm(bounds)
    slope, aspect = slope_aspect(dem, transform.a)
    inside = rasterio.features.geometry_mask(
        [building.buffer(-0.5)], out_shape=dem.shape, transform=transform, invert=True)
    return slope, aspect, inside, transform


def panel_metrics(panel_mask: np.ndarray, mask_transform, building: Polygon) -> dict:
    """Powierzchnia rzutu, powierzchnia rzeczywista (korekta nachylenia) i szacowana moc.

    panel_mask: maska bool modelu w EPSG:2180 (dowolna rozdzielczość).
    """
    px_area = abs(mask_transform.a * mask_transform.e)
    proj_area = float(panel_mask.sum() * px_area)
    if proj_area == 0:
        return {"powierzchnia_rzut_m2": 0.0, "powierzchnia_m2": 0.0, "nachylenie_deg": None,
                "moc_kwp": 0.0, "moc_kwp_zakres": [0.0, 0.0]}

    slope, aspect, inside, dsm_tr = roof_slope_map(building)
    # Nachylenie pod panelami: próbkujemy NMPT w środkach pikseli maski.
    rows, cols = np.nonzero(panel_mask)
    xs, ys = rasterio.transform.xy(mask_transform, rows, cols)
    r, c = rasterio.transform.rowcol(dsm_tr, xs, ys)
    r = np.clip(np.asarray(r), 0, slope.shape[0] - 1)
    c = np.clip(np.asarray(c), 0, slope.shape[1] - 1)
    s = slope[r, c][inside[r, c]]
    # Krawędzie/kominy dają skrajne gradienty — mediana z rozsądnego zakresu.
    s = s[(s >= 0) & (s <= 60)]
    tilt = float(np.median(s)) if s.size else 0.0
    # Na dachach płaskich panele stoją na stelażach ~10–15°, rzut ≈ powierzchnia rzędu.
    corr = 1.0 / np.cos(np.radians(tilt)) if tilt >= 8 else 1.0
    area = proj_area * corr
    asp = aspect[r, c][inside[r, c]]
    return {
        "powierzchnia_rzut_m2": round(proj_area, 1),
        "powierzchnia_m2": round(area, 1),
        "nachylenie_deg": round(tilt, 1),
        "ekspozycja_deg": round(float(np.median(asp)), 0) if asp.size and tilt >= 8 else None,
        "moc_kwp": round(area * KWP_PER_M2["mid"], 1),
        "moc_kwp_zakres": [round(area * KWP_PER_M2["low"], 1), round(area * KWP_PER_M2["high"], 1)],
    }
