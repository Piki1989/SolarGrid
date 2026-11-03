"""Obrysy budynków (OSM, w Polsce w dużej mierze z EGiB/BDOT) — maski 'na dachu'."""
from __future__ import annotations

from functools import lru_cache

import geopandas as gpd
import numpy as np
import rasterio.features
from shapely.geometry import box

from . import ortho

PATH = ortho.DATA / "meta" / "buildings.gpkg"


@lru_cache(maxsize=1)
def _gdf() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(PATH)
    gdf.sindex  # budujemy indeks raz
    return gdf


def in_bounds(bounds) -> gpd.GeoSeries:
    gdf = _gdf()
    return gdf.geometry.iloc[gdf.sindex.query(box(*bounds), predicate="intersects")]


def roof_mask(bounds, transform, shape: tuple[int, int], pad_m: float = 1.0) -> np.ndarray:
    """True na dachach (obrys + pad_m na okapy i przesunięcie ortofoto)."""
    geoms = in_bounds(bounds)
    if geoms.empty:
        return np.zeros(shape, bool)
    return rasterio.features.geometry_mask(
        list(geoms.buffer(pad_m)), out_shape=shape, transform=transform, invert=True)
