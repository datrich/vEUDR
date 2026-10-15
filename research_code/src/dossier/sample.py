"""Feasibility-trial samples (Central Highlands, 5 old provinces).

    python -m src.dossier.sample

Module A (alerts): GFW integrated-alert patches >= 0.3 ha first alerted 01/2025-07/2025 (so the +10..+14 month
images exist). Strata: news-linked patches (pilot extraction, taken with certainty up to N_NEWS) and three area
bins for the rest. Each row keeps its sampling weight (stratum size / sample size) for population estimates.
Module B (plots): 1 ha squares around points where the Forest Data Partnership coffee model (2024) says coffee
(p >= 0.7), stratified on Earth Engine by the two facts an EUDR check turns on:
  B1 forest in JRC GFC2020 AND Hansen loss 2021-2025   (possible post-cut-off conversion)
  B2 forest in JRC GFC2020 AND no loss 2021-2025        ("yellow zone": coffee under a forest label)
  B3 not forest in GFC2020 AND no loss 2021-2025        (expected compliant)
These strata come from the same global layers the dossier uses, so they are a sampling device only; the truth for
evaluation is the blind human interpretation (label sheet), never the strata.
Outputs: data/processed/dossier/sample_alerts.parquet, sample_plots.parquet
"""
from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import pandas as pd

from ..alerts.gee_common import ee_init, to_gdf
from ..common.config import load
from ..common.paths import GAZETTEER, PROCESSED

# DOSSIER_SET selects the item set: unset = the 140-item feasibility batch (dot0), "s1" = the silver set (10/2026)
SET = os.environ.get("DOSSIER_SET", "")
OUT = PROCESSED / "dossier" / SET if SET else PROCESSED / "dossier"
TN = ["gia lai", "kon tum", "dak lak", "dak nong", "lam dong"]
SEED = 42
N_NEWS = 15
AREA_BINS = [(0.3, 0.5, 30), (0.5, 1.0, 30), (1.0, np.inf, 25)]
PLOT_STRATA = {1: 14, 2: 14, 3: 12}
COFFEE = "projects/forestdatapartnership/assets/coffee/model_2026a"


def sample_alerts() -> gpd.GeoDataFrame:
    p = gpd.read_parquet(PROCESSED / "alerts" / "patches_nrt.parquet")
    fd = pd.to_datetime(p.first_date)
    p = p[p.province_key.isin(TN) & (fd >= "2025-01-01") & (fd <= "2025-07-31") & (p.area_ha >= 0.3)].copy()
    linked = set(pd.read_parquet(PROCESSED / "pilot" / "pairs_2025_r2.parquet").patch_id)
    rng = np.random.default_rng(SEED)
    parts = []
    news = p[p.patch_id.isin(linked)]
    take = news.sample(min(N_NEWS, len(news)), random_state=SEED)
    parts.append(take.assign(stratum="news", weight=len(news) / max(len(take), 1)))
    rest = p[~p.patch_id.isin(linked)]
    for lo, hi, n in AREA_BINS:
        s = rest[(rest.area_ha >= lo) & (rest.area_ha < hi)]
        t = s.sample(min(n, len(s)), random_state=int(rng.integers(1e9)))
        parts.append(t.assign(stratum=f"area_{lo}-{hi if np.isfinite(hi) else 'inf'}", weight=len(s) / max(len(t), 1)))
    out = pd.concat(parts, ignore_index=True)
    out["item_id"] = [f"A{i:03d}" for i in range(1, len(out) + 1)]
    out["kind"] = "alert"
    print("alerts population:", len(p), "| sample:", out.stratum.value_counts().to_dict(), flush=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=p.crs)


def sample_plots() -> gpd.GeoDataFrame:
    ee = ee_init()
    c = load("gee")
    prov = gpd.read_file(GAZETTEER / "old_units.gpkg", layer="province")
    region_g = prov[prov.province_key.isin(TN)].to_crs(4326).union_all().simplify(0.01)
    region = ee.Geometry(region_g.__geo_interface__)
    coffee = ee.ImageCollection(COFFEE).filterBounds(region)
    cof24 = coffee.filterDate("2024-01-01", "2025-01-01").mosaic().select("probability").gte(0.7)
    forest20 = ee.Image(c["context"]["jrc_gfc2020_subtypes"]).select("Map").mask().gt(0).unmask(0)
    ly = ee.Image(c["forest_masks"]["hansen"]).select("lossyear").unmask(0)
    loss_post = ly.gte(21).And(ly.lte(25))
    cls = (ee.Image(0).where(forest20.And(loss_post), 1).where(forest20.And(loss_post.Not()), 2)
           .where(forest20.Not().And(loss_post.Not()), 3).updateMask(cof24).rename("stratum").toInt())
    fc = cls.stratifiedSample(numPoints=0, classBand="stratum", region=region, scale=30, seed=SEED,
                              classValues=list(PLOT_STRATA), classPoints=list(PLOT_STRATA.values()),
                              geometries=True, tileScale=16)
    pts = to_gdf(fc)
    utm = pts.to_crs(32648)
    sq = utm.geometry.buffer(50, cap_style="square")  # 100 m x 100 m = 1 ha
    g = gpd.GeoDataFrame({"stratum": "B" + pts.stratum.astype(int).astype(str)}, geometry=sq, crs=32648).to_crs(4326)
    g = g.sample(frac=1, random_state=SEED).reset_index(drop=True)
    g["item_id"] = [f"P{i:03d}" for i in range(1, len(g) + 1)]
    g["kind"] = "plot"
    g["area_ha"] = 1.0
    print("plots:", g.stratum.value_counts().to_dict(), flush=True)
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sample_alerts().to_parquet(OUT / "sample_alerts.parquet")
    sample_plots().to_parquet(OUT / "sample_plots.parquet")


if __name__ == "__main__":
    main()
