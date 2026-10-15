"""Evidence for each sampled polygon (alert patch or plot) -> one table row per item.

    python -m src.dossier.evidence

E1 place: old commune/district/province (GADM) and the commune/province after the 07/2025 merger.
E2 timing: Hansen v1.13 loss share per year (pre-2021 vs 2021-2025), GFW integrated alerts after the EUDR
   cut-off (share of polygon, earliest date; decoded at the native ~10 m grid).
E3 forest in 2020: JRC GFC2020 subtype shares (naturally regenerating / primary / planted), JRC-TMF natural
   vs plantation shares, Hansen tree cover 2000, WDPA overlap; FDP coffee probability 2020 and 2024.
E5 signals: alerts -> the per-patch feature set of src/alerts/features.py (NDVI/NBR pre/post/recovery, S1,
   Dynamic World before/after, FIRMS, MODIS burned...); plots -> dry-season (Jan-Apr) Sentinel-2 NDVI 2019-2026
   and Dynamic World tree/crop probability before the cut-off (2019, 07/2020-03/2021) and in 2025.
Risk prior: 1 km cell probability of the frozen 2025 risk model (trained on data up to 2024).
Output: data/processed/dossier/evidence.parquet
"""
from __future__ import annotations

from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd

from ..alerts.features import _safe, bin_features, s2_composite
from ..alerts.gee_common import ee_init, to_gdf
from ..alerts.patches import alert_image
from ..common.config import load
from ..common.paths import GAZETTEER, PROCESSED
from .sample import COFFEE, OUT

CUTOFF_DAYS = (date(2020, 12, 31) - date(2014, 12, 31)).days
TMF_PLANT = [41, 91, 92, 93]


def admin(g: gpd.GeoDataFrame) -> pd.DataFrame:
    pts = gpd.GeoDataFrame(g[["item_id"]], geometry=g.to_crs(4326).geometry.representative_point(), crs=4326)
    oc = gpd.read_file(GAZETTEER / "old_units.gpkg", layer="commune")[["NAME_1", "NAME_2", "NAME_3", "TYPE_3", "GID_3", "GID_2",
                                                                       "province_key", "geometry"]].to_crs(4326)
    nc = gpd.read_file(GAZETTEER / "new_units.gpkg", layer="commune")[["name", "province_name", "ma", "geometry"]].to_crs(4326)
    a = gpd.sjoin(pts, oc, how="left", predicate="within").drop(columns="index_right").drop_duplicates("item_id")
    b = gpd.sjoin(pts, nc, how="left", predicate="within").drop(columns="index_right").drop_duplicates("item_id")
    out = a.drop(columns="geometry").merge(b[["item_id", "name", "province_name", "ma"]], on="item_id", how="left")
    out = out.rename(columns={"NAME_1": "province_old", "NAME_2": "district_old", "NAME_3": "commune_old",
                              "TYPE_3": "commune_old_type", "name": "commune_new", "province_name": "province_new",
                              "ma": "commune_new_ma"})
    xy = pts.to_crs(32648).geometry
    out["cell"] = (np.floor(xy.x / 1000).astype(int)).astype(str).values + "_" + (np.floor(xy.y / 1000).astype(int)).astype(str).values
    out["lon"], out["lat"] = pts.geometry.x.values, pts.geometry.y.values
    risk = pd.read_parquet(PROCESSED / "pilot" / "risk_pred_final_2025.parquet")
    risk["risk_pct"] = risk.p.rank(pct=True)
    out = out.merge(risk[["cell", "p", "risk_pct"]].rename(columns={"p": "risk_p"}), on="cell", how="left")
    return out


def layer_image(ee):
    c = load("gee")
    h = ee.Image(c["forest_masks"]["hansen"])
    ly = h.select("lossyear").unmask(0)
    bands = [ly.gte(1).And(ly.lte(20)).rename("hl_pre2021"), ly.gte(16).And(ly.lte(20)).rename("hl_2016_2020"),
             ly.gte(21).And(ly.lte(25)).rename("hl_post2020")]
    bands += [ly.eq(y - 2000).rename(f"hl_{y}") for y in range(2021, 2026)]
    bands.append(h.select("treecover2000").rename("tc2000"))
    gfc = ee.Image(c["context"]["jrc_gfc2020_subtypes"]).select("Map")
    for v, n in ((1, "gfc_natregen"), (10, "gfc_primary"), (20, "gfc_planted")):
        bands.append(gfc.eq(v).unmask(0).rename(n))
    bands.append(gfc.mask().gt(0).unmask(0).rename("gfc_forest"))
    tmf = ee.ImageCollection(c["forest_masks"]["tmf_transition"]).mosaic()
    bands.append(tmf.gte(10).And(tmf.lt(40)).unmask(0).rename("tmf_natural"))
    bands.append(tmf.remap(TMF_PLANT, [1] * len(TMF_PLANT), 0).unmask(0).rename("tmf_plantation"))
    wdpa = ee.FeatureCollection(c["context"]["wdpa"]).filter(ee.Filter.eq("ISO3", "VNM"))
    bands.append(ee.Image(0).paint(wdpa, 1).rename("wdpa"))
    cof = ee.ImageCollection(COFFEE)
    for y in (2020, 2024):
        bands.append(cof.filterDate(f"{y}-01-01", f"{y + 1}-01-01").mosaic().select("probability").unmask(0).rename(f"coffee_{y}"))
    return ee.Image.cat(bands), tmf.rename("tmf_subtype")


def gee_common(ee, g: gpd.GeoDataFrame) -> pd.DataFrame:
    fc = ee.FeatureCollection(g[["item_id", "geometry"]].to_crs(4326).__geo_interface__)
    img, tmf = layer_image(ee)
    mean = to_gdf(img.reduceRegions(fc, ee.Reducer.mean(), scale=30, tileScale=8)).drop(columns="geometry")
    mode = to_gdf(tmf.reduceRegions(fc, ee.Reducer.mode().setOutputs(["tmf_mode"]), scale=30, tileScale=8)).drop(columns="geometry")
    al = alert_image(ee)
    post = al.select("days").gt(CUTOFF_DAYS)
    a_img = ee.Image.cat([post.unmask(0).rename("gfw_post2020"),
                          al.select("days").updateMask(post).rename("gfw_first_day")])
    # two bands x (mean, min) -> <band>_mean / <band>_min; we keep the share alerted and the earliest day
    alerts = to_gdf(a_img.reduceRegions(fc, ee.Reducer.mean().combine(ee.Reducer.min(), "", True),
                                        scale=10, tileScale=8)).drop(columns="geometry")
    keep = ["item_id", "gfw_post2020_mean", "gfw_first_day_min"]
    alerts = alerts[[c for c in keep if c in alerts]].rename(columns={"gfw_post2020_mean": "gfw_post2020",
                                                                      "gfw_first_day_min": "gfw_first_day"})
    out = pd.DataFrame(mean).merge(pd.DataFrame(mode), on="item_id").merge(pd.DataFrame(alerts), on="item_id", how="left")
    out["gfw_first_date"] = pd.to_datetime("2014-12-31") + pd.to_timedelta(out.gfw_first_day, unit="D")
    return out


def alert_signals(ee, g: gpd.GeoDataFrame) -> pd.DataFrame:
    parts = []
    for b, s in g.groupby("bin_start"):
        d0, d1 = pd.Timestamp(s.first_date.min()).date(), pd.Timestamp(s.last_date.max()).date()
        f = bin_features(ee, s[["patch_id", "geometry"]].to_crs(4326), d0, d1, int(s.year.iloc[0]), "s2")
        parts.append(f)
        print("alert features bin", b, len(s), flush=True)
    f = pd.concat(parts, ignore_index=True)
    return g[["item_id", "patch_id"]].merge(f, on="patch_id", how="left").drop(columns="patch_id")


def plot_signals(ee, g: gpd.GeoDataFrame) -> pd.DataFrame:
    fc = ee.FeatureCollection(g[["item_id", "geometry"]].to_crs(4326).__geo_interface__)
    region = fc.geometry().bounds()
    bands = [s2_composite(ee, ee.Date(f"{y}-01-01"), ee.Date(f"{y}-04-30"), region).select("ndvi").rename(f"ndvi_dry_{y}")
             for y in range(2019, 2027)]
    dw = ee.ImageCollection(load("gee")["context"]["dynamic_world"]).filterBounds(region)
    for tag, a, b in (("2019", "2019-01-01", "2019-12-31"), ("cutoff", "2020-07-01", "2021-03-31"), ("2025", "2025-01-01", "2025-12-31")):
        m = _safe(ee, dw.filterDate(a, b), ["trees", "crops", "bare"]).mean()
        bands += [m.select("trees").rename(f"dw_trees_{tag}"), m.select("crops").rename(f"dw_crops_{tag}"),
                  m.select("bare").rename(f"dw_bare_{tag}")]
    out = to_gdf(ee.Image.cat(bands).reduceRegions(fc, ee.Reducer.mean(), scale=10, tileScale=8)).drop(columns="geometry")
    return pd.DataFrame(out)


def main():
    ee = ee_init()
    A = gpd.read_parquet(OUT / "sample_alerts.parquet")
    B = gpd.read_parquet(OUT / "sample_plots.parquet")
    rows = []
    for g, sig in ((A, alert_signals), (B, plot_signals)):
        e = admin(g).merge(gee_common(ee, g), on="item_id", how="left").merge(sig(ee, g), on="item_id", how="left")
        print(g.kind.iloc[0], "evidence rows:", len(e), flush=True)
        rows.append(e)
    ev = pd.concat(rows, ignore_index=True)
    ev.to_parquet(OUT / "evidence.parquet")
    print(ev.describe().T[["count", "mean"]].to_string())


if __name__ == "__main__":
    main()
