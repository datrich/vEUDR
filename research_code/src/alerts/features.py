"""Per-patch features on Earth Engine (de-tai §6.5 + human-pressure layers, Stream H).

    python -m src.alerts.features [--tier nrt|annual] [--limit-bins N]

Patches sharing a time bin share one set of composites (pre/post/recovery windows), so we
loop over bins and run reduceRegions on that bin's patches. Distance-based features from
vector data (OSM roads/settlements, WDPA, dams) are computed locally in features_local.py.

Output: data/processed/alerts/features_gee_<tier>.parquet (one row per patch_id)
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import geopandas as gpd
import pandas as pd

from ..common.config import load
from ..common.paths import INTERIM, PROCESSED
from ..common.runlog import Run
from .gee_common import ee_init

TMP = INTERIM / "features_gee"


# ---------------------------------------------------------------- optical / radar composites
def _safe(ee, col, bands):
    """Append a fully masked dummy image so empty collections (e.g. recovery windows in the future)
    yield masked bands instead of 'No band named ...' errors."""
    dummy = ee.Image.constant([0] * len(bands)).rename(bands).toFloat().updateMask(ee.Image(0))
    return col.select(bands).map(lambda i: i.toFloat()).merge(ee.ImageCollection([dummy]))


def s2_composite(ee, start, end, region):
    cfg = load("gee")["imagery"]

    def mask(img):
        scl = img.select("SCL")
        ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return img.updateMask(ok).divide(10000)

    col = ee.ImageCollection(cfg["s2_sr"]).filterBounds(region).filterDate(start, end).map(mask)
    med = _safe(ee, col, ["B4", "B8", "B11", "B12"]).median()
    ndvi = med.normalizedDifference(["B8", "B4"]).rename("ndvi")
    nbr = med.normalizedDifference(["B8", "B12"]).rename("nbr")
    ndmi = med.normalizedDifference(["B8", "B11"]).rename("ndmi")
    return ee.Image.cat([ndvi, nbr, ndmi])


def landsat_composite(ee, start, end, region):
    """Harmonised Landsat 5/7/8/9 C2 L2 NDVI/NBR (used for the long-term tier and pre-2017)."""
    cols = load("gee")["imagery"]["landsat"]

    def prep(img):
        qa = img.select("QA_PIXEL")
        ok = qa.bitwiseAnd(0b11111).eq(0)
        sr = img.select("SR_B.").multiply(0.0000275).add(-0.2)
        is89 = ee.Number(img.get("SPACECRAFT_ID")).format().match("LANDSAT_[89]").size().gt(0)
        nir = ee.Image(ee.Algorithms.If(is89, sr.select("SR_B5"), sr.select("SR_B4")))
        red = ee.Image(ee.Algorithms.If(is89, sr.select("SR_B4"), sr.select("SR_B3")))
        swir2 = sr.select("SR_B7")
        ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
        nbr = nir.subtract(swir2).divide(nir.add(swir2)).rename("nbr")
        return ndvi.addBands(nbr).updateMask(ok)

    merged = ee.ImageCollection([])
    for c in cols:
        merged = merged.merge(ee.ImageCollection(c).filterBounds(region).filterDate(start, end).map(prep))
    return _safe(ee, merged, ["ndvi", "nbr"]).median()


def s1_composite(ee, start, end, region):
    col = (ee.ImageCollection(load("gee")["imagery"]["s1"]).filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    return _safe(ee, col, ["VV", "VH"]).median()


# ---------------------------------------------------------------- static / context layers
def static_layers(ee, year: int):
    c = load("gee")
    ctx, fm = c["context"], c["forest_masks"]
    dem = ee.ImageCollection(ctx["dem"]).select("DEM").mosaic()
    slope = ee.Terrain.slope(dem.setDefaultProjection("EPSG:4326", None, 30))
    h = ee.Image(fm["hansen"])
    tmf = ee.ImageCollection(fm["tmf_transition"]).mosaic().rename("tmf_subtype")
    primary = ee.ImageCollection(fm["primary_humid_tropics"]).mosaic().unmask(0).gt(0).rename("primary")
    ly = h.select("lossyear").unmask(0)
    yy = year - 2000
    prior_loss = ly.gte(yy - 3).And(ly.lt(yy)).And(ly.gt(0)).rename("loss_prior3y")
    wp_year = min(max(year, 2000), 2020)
    worldpop = ee.ImageCollection(ctx["worldpop"]).filter(ee.Filter.eq("year", wp_year)).mosaic().rename("pop")
    ghs_year = min(max(5 * round(year / 5), 1975), 2030)
    built = ee.Image(f"{ctx['ghs_built']}/{ghs_year}").select("built_surface").rename("built_m2")
    # night lights: latest available image at or before `year` (VIIRS 2012+, DMSP before); never null
    if year >= 2013:
        ntl = ee.ImageCollection(ctx["viirs_v21"]).merge(ee.ImageCollection(ctx["viirs"])).filterDate("2012-01-01", f"{year + 1}-01-01")             .sort("system:time_start", False).first().select("average").rename("ntl")
    else:
        ntl = ee.ImageCollection(ctx["dmsp"]).filterDate("1992-01-01", f"{year + 1}-01-01")             .sort("system:time_start", False).first().select("stable_lights").rename("ntl")
    tt = ee.Image(ctx["travel_time"]).select("accessibility").rename("travel_min")
    return dict(dem=dem.rename("elev"), slope=slope.rename("slope"), tc2000=h.select("treecover2000"),
                tmf=tmf, primary=primary, prior_loss=prior_loss, pop=worldpop, built=built, ntl=ntl, tt=tt)


def bin_features(ee, gdf: gpd.GeoDataFrame, d_first: date, d_last: date, year: int, optical: str = "s2"):
    p = load("gee")["patches"]
    w = p["windows"]
    fc = ee.FeatureCollection(gdf[["patch_id", "geometry"]].__geo_interface__)
    region = fc.geometry().bounds()

    def win(a, b):
        return ee.Date(str(a)), ee.Date(str(b))

    pre = win(d_first - timedelta(days=w["pre_days"][0]), d_first - timedelta(days=w["pre_days"][1]))
    post = win(d_last + timedelta(days=w["post_days"][0]), d_last + timedelta(days=w["post_days"][1]))
    comp = s2_composite if optical == "s2" else landsat_composite
    img = comp(ee, *pre, region).rename(["ndvi_pre", "nbr_pre"] + (["ndmi_pre"] if optical == "s2" else []))
    img = img.addBands(comp(ee, *post, region).rename(["ndvi_post", "nbr_post"] + (["ndmi_post"] if optical == "s2" else [])))
    for m in w["recovery_months"]:
        a = d_last + timedelta(days=30 * m - 30)
        b = d_last + timedelta(days=30 * m + 30)
        img = img.addBands(comp(ee, ee.Date(str(a)), ee.Date(str(b)), region).select("ndvi").rename(f"ndvi_rec{m}m"))
    if optical == "s2" and year >= 2015:
        s1a = s1_composite(ee, *pre, region).rename(["vv_pre", "vh_pre"])
        s1b = s1_composite(ee, *post, region).rename(["vv_post", "vh_post"])
        img = img.addBands(s1a).addBands(s1b)
    st = static_layers(ee, year)
    img = img.addBands([st["dem"], st["slope"], st["tc2000"], st["primary"], st["prior_loss"], st["tt"]])
    ctx = load("gee")["context"]
    if year >= 2016:
        # what the land became: Dynamic World class probabilities 1-12 months after the alert,
        # and tree probability before it (separates conversion C1 from plantation rotation C2)
        dw = ee.ImageCollection(ctx["dynamic_world"]).filterBounds(region)
        dw_cls = ["trees", "crops", "shrub_and_scrub", "bare", "built", "grass", "water"]
        dw_post = _safe(ee, dw.filterDate(str(d_last + timedelta(days=30)), str(d_last + timedelta(days=365))), dw_cls)             .mean().rename([f"dw_post_{c}" for c in dw_cls])
        dw_pre = _safe(ee, dw.filterDate(str(d_first - timedelta(days=365)), str(d_first - timedelta(days=30))),
                       ["trees", "crops"]).mean().rename(["dw_pre_trees", "dw_pre_crops"])
        img = img.addBands(dw_post).addBands(dw_pre)
    burned = _safe(ee, ee.ImageCollection(ctx["modis_burned"]).filterDate(str(d_first - timedelta(days=30)),
                                                                           str(d_last + timedelta(days=30))),
                   ["BurnDate"]).max().gt(0).unmask(0).rename("modis_burned")
    img = img.addBands(burned)
    mean = img.reduceRegions(fc, ee.Reducer.mean(), scale=30, tileScale=8)
    cats = ee.Image.cat([st["tmf"], ee.Image(ctx["jrc_gfc2020_subtypes"]).select("Map").rename("gfc2020_type")])
    mode = cats.reduceRegions(fc, ee.Reducer.mode().forEach(["tmf_subtype", "gfc2020_type"]), scale=30, tileScale=8)

    # buffered human-pressure and fire/climate context
    buf1 = fc.map(lambda f: f.buffer(p["pop_buffer_m"][0]))
    buf5 = fc.map(lambda f: f.buffer(p["pop_buffer_m"][1]))
    human1 = ee.Image.cat([st["pop"].rename("pop_1km"), st["built"].rename("built_1km")]) \
        .reduceRegions(buf1, ee.Reducer.sum(), scale=100, tileScale=8)
    human5 = ee.Image.cat([st["pop"].rename("pop_5km"), st["ntl"].rename("ntl_5km")]) \
        .reduceRegions(buf5, ee.Reducer.mean(), scale=500, tileScale=8)
    # single-band reduceRegions names its output after the reducer ("mean"/"sum"), not the band: set it explicitly
    loss1 = st["prior_loss"].reduceRegions(buf1, ee.Reducer.mean().setOutputs(["loss_prior3y_1km"]), scale=30, tileScale=8)
    fd = p["firms_days"]
    firms = _safe(ee, ee.ImageCollection(ctx["firms"]).filterDate(str(d_first - timedelta(days=fd)),
                                                                  str(d_last + timedelta(days=fd))), ["T21"]) \
        .max().gt(0).unmask(0).rename("firms_px")
    fbuf = fc.map(lambda f: f.buffer(p["firms_buffer_m"]))
    fire = firms.reduceRegions(fbuf, ee.Reducer.sum().setOutputs(["firms_px"]), scale=1000, tileScale=4)
    chirps = ee.ImageCollection(ctx["chirps"]).filterDate(str(d_first - timedelta(days=90)), str(d_first))
    rain = ee.Image.cat([chirps.sum().rename("rain_90d"),
                         chirps.map(lambda i: i.lt(1)).sum().rename("dry_days_90d")]) \
        .reduceRegions(fc, ee.Reducer.mean(), scale=5000, tileScale=4)

    from .gee_common import to_gdf
    out = None
    for part in (mean, mode, human1, human5, loss1, fire, rain):
        df = pd.DataFrame(to_gdf(part).drop(columns="geometry", errors="ignore"))
        out = df if out is None else out.merge(df, on="patch_id", how="outer")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["nrt", "annual"], default="nrt")
    ap.add_argument("--limit-bins", type=int, default=0)
    a = ap.parse_args()
    ee = ee_init()
    src = PROCESSED / "alerts" / ("patches_nrt.parquet" if a.tier == "nrt" else "patches_annual.parquet")
    patches = gpd.read_parquet(src)
    key = "bin_start" if a.tier == "nrt" else "year"
    groups = list(patches.groupby(key))
    if a.limit_bins:
        groups = groups[: a.limit_bins]
    out_dir = TMP / a.tier
    out_dir.mkdir(parents=True, exist_ok=True)
    with Run("features_gee", tier=a.tier, n_groups=len(groups), n_patches=len(patches)) as run:
        for k, g in groups:
            f = out_dir / f"{k}.parquet"
            if f.exists():
                continue
            if a.tier == "nrt":
                d_first, d_last, year, optical = g.first_date.min(), g.last_date.max(), int(g.year.iloc[0]), "s2"
            else:
                year = int(k)
                d_first, d_last, optical = date(year, 1, 1), date(year, 12, 31), "landsat"
            d_first, d_last = pd.Timestamp(d_first).date(), pd.Timestamp(d_last).date()
            chunks = [g.iloc[i:i + 2000] for i in range(0, len(g), 2000)]
            parts = [bin_features(ee, c.to_crs(4326), d_first, d_last, year, optical) for c in chunks]
            pd.concat(parts, ignore_index=True).to_parquet(f)
            print(a.tier, k, len(g), flush=True)
        allf = pd.concat([pd.read_parquet(f) for f in out_dir.glob("*.parquet")], ignore_index=True)
        allf.to_parquet(PROCESSED / "alerts" / f"features_gee_{a.tier}.parquet")
        run.log(n_rows=len(allf))


if __name__ == "__main__":
    main()
