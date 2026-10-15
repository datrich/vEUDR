"""Alert patches (de-tai §3 unit of analysis): 8-connected alert pixels within the same time bin.

    python -m src.alerts.patches --years 2020 2021 2022 2023 2024 2025 2026 [--tiles N]

For every 0.5° tile and every `time_bin_days` bin, alert pixels (GFW integrated alerts) are
vectorised with reduceToVectors; each polygon gets first/last alert date, pixel count and max
confidence. Patches smaller than `min_pixels` are dropped. Output (resumable, one file per tile-year):
    data/interim/patches/<year>/<tile>.parquet  ->  data/processed/alerts/patches_nrt.parquet
Records the alert asset version (ee asset update time) in the run log.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from ..common.config import load
from ..common.paths import GAZETTEER, INTERIM, PROCESSED
from ..common.runlog import Run
from .gee_common import ee_init, tiles, to_gdf

TMP = INTERIM / "patches"


def alert_image(ee):
    cfg = load("gee")["alerts"]["integrated"]
    raw = ee.Image(cfg["asset"]).select(0)
    days = raw.mod(10000).rename("days")
    conf = raw.divide(10000).floor().rename("conf")
    return days.updateMask(raw.gt(0)).addBands(conf.updateMask(raw.gt(0)))


def bins_for_year(year: int, step: int, epoch: date) -> list[tuple[int, int]]:
    start, end = date(year, 1, 1), date(year, 12, 31)
    out, d = [], start
    while d <= end:
        e = min(d + timedelta(days=step - 1), end)
        out.append(((d - epoch).days, (e - epoch).days))
        d = e + timedelta(days=1)
    return out


def tile_year(ee, img, tile, year, cfg) -> gpd.GeoDataFrame:
    p = cfg["patches"]
    epoch = date.fromisoformat(cfg["alerts"]["integrated"]["epoch"])
    region = ee.Geometry.Rectangle(list(tile))
    parts = []
    for d0, d1 in bins_for_year(year, p["time_bin_days"], epoch):
        sel = img.updateMask(img.select("days").gte(d0).And(img.select("days").lte(d1)))
        label = sel.select("days").gt(0).selfMask().rename("label").int()
        vec_img = label.addBands(sel.select("days")).addBands(sel.select("conf"))
        reducer = ee.Reducer.minMax().combine(ee.Reducer.count(), sharedInputs=True)
        fc = vec_img.select(["label", "days"]).reduceToVectors(
            geometry=region, scale=cfg["alerts"]["integrated"]["scale_m"], geometryType="polygon",
            eightConnected=True, reducer=reducer, maxPixels=1e10, tileScale=4, bestEffort=False)
        fc = fc.filter(ee.Filter.gte("count", p["min_pixels"]))
        # max confidence per patch
        fc = sel.select("conf").reduceRegions(fc, ee.Reducer.max().setOutputs(["conf_max"]),
                                              scale=cfg["alerts"]["integrated"]["scale_m"], tileScale=4)
        g = to_gdf(fc)
        if len(g):
            g["bin_start"] = d0
            parts.append(g)
    if not parts:
        return gpd.GeoDataFrame()
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    g = g.rename(columns={"min": "first_day", "max": "last_day", "count": "n_px"})
    g["first_date"] = [epoch + timedelta(days=int(x)) for x in g.first_day]
    g["last_date"] = [epoch + timedelta(days=int(x)) for x in g.last_day]
    g["year"] = year
    g["tile"] = "_".join(map(str, tile))
    return g.drop(columns=[c for c in ("label",) if c in g])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--tiles", type=int, default=0, help="limit number of tiles (smoke test)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent Earth Engine requests (tiles are independent)")
    a = ap.parse_args()
    ee = ee_init()
    cfg = load("gee")
    img = alert_image(ee)
    vn = gpd.read_file(GAZETTEER / "old_units.gpkg", layer="province").union_all()
    tl = [t for t in tiles() if vn.intersects(box(*t))]
    if a.tiles:
        tl = tl[: a.tiles]
    asset_info = ee.data.getAsset(cfg["alerts"]["integrated"]["asset"])
    with Run("patches", years=a.years, n_tiles=len(tl), alert_asset_updated=asset_info.get("updateTime")) as run:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        jobs = []
        for y in a.years:
            (TMP / str(y)).mkdir(parents=True, exist_ok=True)
            jobs += [(y, t) for t in tl if not (TMP / str(y) / f"{'_'.join(map(str, t))}.parquet").exists()]

        def one(job):
            y, t = job
            for attempt in range(3):
                try:
                    g = tile_year(ee, img, t, y, cfg)
                    break
                except Exception as e:  # noqa: BLE001 - EE transient errors / quota: back off and retry
                    if attempt == 2:
                        print("FAILED", y, t, repr(e)[:200], flush=True)
                        return 0
                    import time
                    time.sleep(30 * (attempt + 1))
            f = TMP / str(y) / f"{'_'.join(map(str, t))}.parquet"
            (g if len(g) else gpd.GeoDataFrame(geometry=[], crs=4326)).to_parquet(f)
            return len(g)

        n, done = 0, 0
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(one, j): j for j in jobs}
            for fu in as_completed(futs):
                n += fu.result()
                done += 1
                y, t = futs[fu]
                print(f"{done}/{len(jobs)}", y, t, fu.result(), flush=True)
        allp = [gpd.read_parquet(f) for f in sorted(TMP.rglob("*.parquet"))]
        allp = gpd.GeoDataFrame(pd.concat([g for g in allp if len(g)], ignore_index=True), crs=4326)
        allp["patch_id"] = [f"P{i:08d}" for i in range(len(allp))]
        allp["area_ha"] = allp.to_crs(32648).area / 1e4
        # 0.5-degree tiles overlap Laos/Cambodia/China: keep patches whose representative point is in VN
        prov = gpd.read_file(GAZETTEER / "old_units.gpkg", layer="province")[["province_key", "geometry"]].to_crs(4326)
        pts = gpd.GeoDataFrame(allp[["patch_id"]], geometry=allp.geometry.representative_point(), crs=4326)
        j = gpd.sjoin(pts, prov, how="inner", predicate="within").drop_duplicates("patch_id")
        n_all = len(allp)
        allp = allp.merge(j[["patch_id", "province_key"]], on="patch_id", how="inner")
        run.log(n_outside_vn=n_all - len(allp))
        (PROCESSED / "alerts").mkdir(parents=True, exist_ok=True)
        allp.to_parquet(PROCESSED / "alerts" / "patches_nrt.parquet")
        run.log(n_patches=len(allp), new=n)


if __name__ == "__main__":
    main()
