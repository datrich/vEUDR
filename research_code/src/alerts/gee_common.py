"""Earth Engine helpers: init, Vietnam geometry, tiling, asset checks, paged feature download.

    python -m src.alerts.gee_common check     # verify every asset id in configs/gee.yaml
"""
from __future__ import annotations

import sys
from functools import lru_cache

from ..common.config import env, load


@lru_cache(maxsize=1)
def ee_init():
    import ee
    project = env(load("gee")["project_env"], required=True)
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


def vietnam(ee=None):
    """Vietnam outline from the project gazetteer (uploaded once) or FAO GAUL fallback."""
    ee = ee or ee_init()
    gaul = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq("ADM0_NAME", "Viet Nam"))
    return gaul.geometry()


def tiles(bounds=(102.1, 8.4, 109.5, 23.4), step: float | None = None) -> list[tuple[float, float, float, float]]:
    step = step or load("gee")["patches"]["tile_deg"]
    out = []
    x = bounds[0]
    while x < bounds[2]:
        y = bounds[1]
        while y < bounds[3]:
            out.append((round(x, 4), round(y, 4), round(min(x + step, bounds[2]), 4), round(min(y + step, bounds[3]), 4)))
            y += step
        x += step
    return out


def to_gdf(fc):
    """Download an ee.FeatureCollection into a GeoDataFrame in ONE computation.

    computeFeatures pages internally; the previous size() + toList().slice() approach made
    Earth Engine recompute the whole collection once per page (3-4x slower)."""
    import geopandas as gpd
    ee = ee_init()
    g = ee.data.computeFeatures({"expression": fc, "fileFormat": "GEOPANDAS_GEODATAFRAME"})
    if g is None or len(g) == 0:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame(g).set_crs("EPSG:4326", allow_override=True)


def check_assets():
    ee = ee_init()
    cfg = load("gee")
    ids = []
    for sec in ("alerts", "forest_masks", "imagery", "context"):
        for k, v in cfg[sec].items():
            vals = v if isinstance(v, list) else [v.get("asset") if isinstance(v, dict) else v]
            ids += [(f"{sec}.{k}", x) for x in vals if isinstance(x, str) and "/" in x]
    for name, a in ids:
        try:
            info = ee.data.getAsset(a)
            print(f"OK   {name:28s} {a}  ({info.get('type')})")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name:28s} {a}  -> {str(e)[:80]}")


if __name__ == "__main__":
    if (sys.argv[1:] or ["check"])[0] == "check":
        check_assets()
