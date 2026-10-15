"""Build the public EUDR-CH dataset (version 0.9, no labels) for Kaggle from the project's internal files.

    .venv/Scripts/python -m src.release.build_kaggle [--out release/kaggle/eudr-ch] [--skip-images]

Everything is written in English. Left out on purpose (see release/kaggle/eudr-ch/README.md):
human labels, silver labels, model suggestions and predictions, labelling-mode flags, the official forest-status
map of Vietnam and its lot attributes, Forest Data Partnership and Nature Trace layers, WDPA, the GLAD and JRC
forest-type layers (licences not yet checked), and the Esri Wayback capture metadata (terms not yet checked).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("DOSSIER_SET", "s1")
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
S1 = ROOT / "data" / "processed" / "dossier" / "s1"
IS1 = ROOT / "data" / "interim" / "dossier" / "s1"
PKG = ROOT / "data" / "goi_gan_nhan" / "cho_nguoi_gan"
GADM = ROOT / "data" / "raw" / "gazetteer" / "gadm41_VNM.gpkg"
PROV = ["Kon Tum", "Gia Lai", "Dak Lak", "Dak Nong", "Lam Dong"]
TILE, TITLE = 192, 18                      # every sheet: tiles of 192 px under an 18 px title band, 2 rows, NDVI chart
INTERP_HALF = 0.0045                       # interpreter sheets and rendered scenes: centroid +- 0.0045 deg (~1 km)
STD_HALF = {"alert": 0.0036, "plot": 0.0027}
# window definitions copied from src/dossier/silver_set.py (interpreter) and src/dossier/ref_sheets.py (standard)
INTERP_WINS = {
    "alert": [("dry 2019", "2019-01-01", "2019-04-30"), ("dry 2020", "2020-01-01", "2020-04-30"),
              ("CUT-OFF 11/20-04/21", "2020-11-01", "2021-04-30"), ("-4..-1 mo", -120, -20), ("ALERT 0..+2 mo", 0, 60),
              ("+3..+6 mo", 90, 180), ("+10..+14 mo", 300, 420), ("+2 yr", 700, 820), ("+3 yr", 1065, 1185),
              ("dry 2026", "2026-01-01", "2026-04-30")],
    "plot": [(f"dry {y}", f"{y}-01-01", f"{y}-04-30") for y in (2017, 2018, 2019, 2020)]
            + [("CUT-OFF 11/20-04/21", "2020-11-01", "2021-04-30")]
            + [(f"dry {y}", f"{y}-01-01", f"{y}-04-30") for y in range(2022, 2027)]}
STD_WINS = {
    "alert": [("-12..-7 mo", -365, -200), ("-4..-1 mo", -120, -20), ("ALERT 0..+2 mo", 0, 60), ("+3..+6 mo", 90, 180),
              ("+10..+14 mo", 300, 420), ("+16..+20 mo", 480, 600)],
    "plot": [("dry 2017", "2016-12-01", "2017-04-30"), ("dry 2018", "2018-01-01", "2018-04-30"),
             ("dry 2019", "2019-01-01", "2019-04-30"), ("dry 2020", "2020-01-01", "2020-04-30"),
             ("CUT-OFF 11/20-04/21", "2020-11-01", "2021-04-30"), ("dry 2022", "2022-01-01", "2022-04-30"),
             ("dry 2024", "2024-01-01", "2024-04-30"), ("dry 2026", "2026-01-01", "2026-04-30")]}
LAST_IMAGE = pd.Timestamp("2026-10-01")
ROWS = ["true colour (B4, B3, B2)", "false colour NIR (B8, B4, B3)"]

# map features kept, with source and licence (open licences only; see the module docstring for what is left out)
SRC = {
    "hansen": ("Hansen et al., Global Forest Change v1.13 (UMD/hansen/global_forest_change_2025_v1_13)", "CC BY 4.0"),
    "tmf": ("JRC Tropical Moist Forests v1 2024 (projects/JRC/TMF/v1_2024)", "CC BY 4.0"),
    "gfc": ("JRC GFC2020 global forest cover 2020, versions 1-4 (EU Observatory)", "CC BY 4.0"),
    "gfw": ("GFW integrated deforestation alerts (projects/forma-250/assets/gfw_integrated_alerts)", "CC BY 4.0"),
    "dw": ("Dynamic World V1 (GOOGLE/DYNAMICWORLD/V1)", "CC BY 4.0"),
    "wc": ("ESA WorldCover 10 m 2020 v100", "CC BY 4.0"),
    "eth": ("ETH Global Canopy Height 2020 (Lang et al.)", "CC BY 4.0"),
    "meta": ("Meta / WRI high-resolution canopy height", "CC BY 4.0"),
    "s2": ("Derived from Copernicus Sentinel-2 L1C / L2A", "Copernicus data licence (free, attribution)"),
    "s1": ("Derived from Copernicus Sentinel-1 GRD", "Copernicus data licence (free, attribution)"),
    "modis": ("NASA MODIS MCD64A1 burned area", "NASA open data"),
    "firms": ("NASA FIRMS active fires (VIIRS, MODIS)", "NASA open data"),
    "dem": ("Copernicus DEM GLO-30", "Copernicus DEM licence (free, attribution)"),
    "drivers": ("Sims et al. 2025, drivers of forest loss 1 km v1.2", "CC BY 4.0"),
}
FEATURES = [
    ("area_ha", None, "Area of the item (ha)"),
    ("first_date", "gfw", "Alerts: date of the first alerted pixel of the patch"),
    ("hl_pre2021", "hansen", "Share of the item with Hansen tree-cover loss in 2001-2020"),
    ("hl_post2020", "hansen", "Share with Hansen loss in 2021-2025"),
    *[(f"hl_{y}", "hansen", f"Share with Hansen loss in {y}") for y in range(2021, 2026)],
    ("tc2000", "hansen", "Mean Hansen tree cover in 2000 (%)"),
    ("buf_tc2000", "hansen", "As tc2000, item plus 100 m surroundings"),
    ("tmf_natural", "tmf", "Share in TMF transition classes 10-39 (undisturbed / degraded forest)"),
    ("tmf_plantation", "tmf", "Share in TMF plantation classes (41, 91-93)"),
    *[(f"tmf{y}_{c}", "tmf", f"Share in TMF annual change class '{c}' in December 20{y}")
      for y in ("20", "25") for c in ("undisturbed", "degraded", "deforested", "regrowth")],
    *[(f"gfcV{v}", "gfc", f"Share mapped as forest by GFC2020 version {v}") for v in range(1, 5)],
    ("buf_gfcV1", "gfc", "As gfcV1, item plus 100 m surroundings"), ("buf_gfcV4", "gfc", "As gfcV4, item plus 100 m surroundings"),
    ("gfw_post2020", "gfw", "Share of the item with an integrated alert after 2020"),
    ("gfw_first_date", "gfw", "Date of the first integrated alert after 2020 in the item"),
    *[(f"dw_{w}_{c}", "dw", f"Mean Dynamic World probability of '{c}' {'before' if w == 'pre' else 'after'} the alert")
      for w, cs in (("pre", ("trees", "crops")), ("post", ("trees", "crops", "bare", "built", "water"))) for c in cs],
    *[(f"dw_{c}_{y}", "dw", f"Mean Dynamic World probability of '{c}' in {y} (plots)")
      for c, y in (("trees", 2019), ("crops", 2019), ("trees", 2025), ("crops", 2025), ("bare", 2025))],
    ("wc20_tree", "wc", "Share mapped as tree cover by WorldCover 2020"), ("wc20_crop", "wc", "Share mapped as cropland by WorldCover 2020"),
    ("h_eth", "eth", "Mean canopy height, ETH 2020 (m)"),
    ("h_meta_mean", "meta", "Mean canopy height, Meta / WRI (m)"), ("h_meta_p90", "meta", "90th percentile canopy height, Meta / WRI (m)"),
    ("ndvi_pre", "s2", "Sentinel-2 NDVI before the alert"), ("ndvi_post", "s2", "Sentinel-2 NDVI just after the alert"),
    *[(f"ndvi_rec{m}", "s2", f"Sentinel-2 NDVI {m} after the alert") for m in ("3m", "6m", "12m")],
    *[(f"ndvi_dry_{y}", "s2", f"Sentinel-2 dry-season (Jan-Apr) median NDVI {y}") for y in range(2019, 2027)],
    ("vh_pre", "s1", "Sentinel-1 VH before the alert (dB)"), ("vh_post", "s1", "Sentinel-1 VH after the alert (dB)"),
    ("modis_burned", "modis", "Share burned (MCD64A1) around the alert or in 2021-2025"),
    ("firms_px", "firms", "Number of FIRMS hot spots around the alert or in 2021-2025"),
    ("elev", "dem", "Mean elevation (m)"), ("slope", "dem", "Mean slope (degrees)"),
    ("driver_1km", "drivers", "Dominant driver of loss of the 1-km cell containing the centroid (1 permanent agriculture, "
                              "2 hard commodities, 3 shifting cultivation, 4 logging, 5 wildfire, 6 settlements and infrastructure, "
                              "7 other natural disturbances)"),
]
S2_COLS = {"ma_diem": "item_id", "ngay": "datetime", "ma_canh": "scene_id", "ve_tinh": "satellite", "o_mgrs": "mgrs_tile",
           "may_ca_canh_pct": "scene_cloud_pct", "phu_khung": "box_coverage", "ti_le_quang_khung": "box_clear_share",
           "phu_ranh": "outline_coverage", "ti_le_quang_ranh": "outline_clear_share", "ndvi_ranh": "ndvi_outline"}
S1_COLS = {"ma_diem": "item_id", "ngay": "datetime", "ma_canh": "scene_id", "ve_tinh": "satellite", "chieu_quy_dao": "orbit_pass",
           "quy_dao": "relative_orbit", "vv_db": "vv_db", "vh_db": "vh_db"}
SCENE_COLS = {"tep": "file", "ngay": "date", "ma_canh": "scene_id", "o_mgrs": "mgrs_tile", "kieu_mau": "render",
              "ti_le_quang_khung": "box_clear_share", "ti_le_quang_ranh": "outline_clear_share", "ndvi_ranh": "ndvi_outline",
              "ly_do_chon": "reason"}
RENDER = {"mau_that": "true_colour", "mau_gia_hong_ngoai": "false_colour_nir"}
REASON = {"it_may_trong_nam": "clearest_of_year", "gan_moc_cat_2020": "nearest_cutoff", "truoc_canh_bao": "before_first_alert",
          "sau_canh_bao": "after_first_alert"}


def log(*a):
    print(*a, flush=True)


def items() -> gpd.GeoDataFrame:
    m = pd.read_csv(S1 / "items_s1.csv")
    A = gpd.read_parquet(S1 / "sample_alerts.parquet").to_crs(4326)
    B = gpd.read_parquet(S1 / "sample_plots.parquet").to_crs(4326)
    g = pd.concat([A[["item_id", "area_ha", "first_date", "geometry"]], B[["item_id", "area_ha", "geometry"]]], ignore_index=True)
    g = gpd.GeoDataFrame(m.merge(g, on="item_id", how="left"), geometry="geometry", crs=4326)
    assert len(g) == 1780 and g.geometry.notna().all()
    prov = gpd.read_file(GADM, layer="ADM_ADM_1")
    prov = prov[prov.VARNAME_1.isin(PROV)][["VARNAME_1", "geometry"]]
    pts = gpd.GeoDataFrame(g[["item_id"]], geometry=gpd.points_from_xy(g.lon, g.lat), crs=4326).to_crs(32648)
    j = gpd.sjoin_nearest(pts, prov.to_crs(32648), how="left").drop_duplicates("item_id").set_index("item_id")
    g["province_2020"] = g.item_id.map(j.VARNAME_1)
    return g


def windows(kind, first_date, spec):
    t0 = pd.Timestamp(first_date) if kind == "alert" else None
    out = []
    for lab, a, b in spec[kind]:
        s, e = ((t0 + timedelta(days=a)), (t0 + timedelta(days=b))) if isinstance(a, int) else (pd.Timestamp(a), pd.Timestamp(b))
        out.append({"label": lab, "start": str(s.date()), "end": str(min(e, LAST_IMAGE).date()), "after_last_image": bool(s > LAST_IMAGE)})
    return out


def rings_px(geom, box):
    x0, y0, x1, y1 = box
    polys = list(getattr(geom, "geoms", [geom]))
    return [[[round((x - x0) / (x1 - x0) * TILE, 2), round((y1 - y) / (y1 - y0) * TILE, 2)] for x, y in p.exterior.coords] for p in polys]


def layout(g, half_of, spec) -> dict:
    out = {}
    for r in g.itertuples():
        c = r.geometry.centroid
        h = half_of(r.kind)
        box = (c.x - h, c.y - h, c.x + h, c.y + h)
        wins = windows(r.kind, r.first_date, spec)
        out[r.item_id] = {"box_wsen_deg": [round(v, 7) for v in box], "tile_px": TILE, "title_band_px": TITLE,
                          "sheet_px": [len(wins) * TILE, 2 * (TILE + TITLE) + 240], "rows": ROWS, "windows": wins,
                          "outline_px_in_tile": rings_px(r.geometry, box)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "release" / "kaggle" / "eudr-ch")
    ap.add_argument("--skip-images", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    g = items()
    batch = sorted(d for d in os.listdir(PKG) if (PKG / d).is_dir())
    assert set(batch) <= set(g.item_id)
    # ---- items and outlines
    n_str = g.groupby("stratum").item_id.transform("size")
    it = pd.DataFrame({
        "item_id": g.item_id, "kind": g.kind, "stratum": g.stratum, "stratum_population": g["pop"].astype("Int64"),
        "n_sampled_in_stratum": n_str, "wave": g.wave, "lon": g.lon.round(6), "lat": g.lat.round(6),
        "area_ha": g.area_ha.round(4), "first_alert_date": g.first_date.where(g.kind == "alert"), "province_2020": g.province_2020,
        "block_0p2deg": g.block, "split": g.split, "has_scene_series": g.item_id.isin(batch)})
    it.to_csv(out / "items.csv", index=False)
    g[["item_id", "kind", "area_ha", "geometry"]].assign(area_ha=lambda d: d.area_ha.round(4)).to_file(out / "outlines.geojson", driver="GeoJSON")
    log("items", len(it), "with scene series", int(it.has_scene_series.sum()))
    # ---- map features (open licences only) + data dictionary
    f = pd.read_parquet(S1 / "legal_features.parquet").set_index("item_id")
    dr = pd.read_csv(IS1 / "gfw_drivers.csv").set_index("item_id").driver
    f["driver_1km"] = dr.reindex(f.index).astype("Int64")
    cols = [c for c, _, _ in FEATURES]
    missing = [c for c in cols if c not in f.columns]
    assert not missing, missing
    mf = f.loc[g.item_id, cols].reset_index()
    mf.to_parquet(out / "map_features.parquet", index=False)
    pd.DataFrame([{"column": c, "description": d, "source": SRC[s][0] if s else "sample design",
                   "licence": SRC[s][1] if s else "CC BY 4.0"} for c, s, d in FEATURES]).to_csv(out / "map_features_dictionary.csv", index=False)
    log("map features", mf.shape)
    # ---- NDVI series (all items, Sentinel-2 L1C, clear scenes, one value per day)
    nd = pd.concat([pd.read_parquet(p) for p in sorted((IS1 / "ndvi_series").glob("wave_*.parquet")) if p.stem != "wave_-1"])
    nd = nd[nd.item_id.isin(set(g.item_id))][["item_id", "date", "ndvi"]].assign(source="sheet_series")
    # items whose sheet series was not kept: the same series from the per-scene table (outline clear share >= 0.5,
    # mean per day), which reproduces the sheet series exactly where both exist
    s2n = pd.read_csv(PKG / "tat_ca_s2_canh.csv", encoding="utf-8-sig", usecols=["ma_diem", "ngay", "ti_le_quang_ranh", "ndvi_ranh"])
    s2n = s2n[~s2n.ma_diem.isin(set(nd.item_id)) & (s2n.ti_le_quang_ranh >= 0.5) & s2n.ndvi_ranh.notna()]
    s2n["date"] = pd.to_datetime(s2n.ngay, format="mixed").dt.normalize()
    fill = s2n.groupby(["ma_diem", "date"]).ndvi_ranh.mean().reset_index().rename(columns={"ma_diem": "item_id", "ndvi_ranh": "ndvi"})
    nd = pd.concat([nd, fill.assign(source="scene_table")], ignore_index=True).sort_values(["item_id", "date"])
    nd.to_parquet(out / "ndvi_s2_l1c_series.parquet", index=False)
    log("ndvi series", len(nd), nd.item_id.nunique(), "items")
    # ---- per-scene tables (items with scene series)
    s2 = pd.read_csv(PKG / "tat_ca_s2_canh.csv", encoding="utf-8-sig").rename(columns=S2_COLS)
    ls = pd.read_csv(PKG / "tat_ca_landsat.csv", encoding="utf-8-sig").rename(columns=S2_COLS)
    s1 = pd.read_csv(PKG / "tat_ca_s1_radar.csv", encoding="utf-8-sig").rename(columns=S1_COLS)
    for name, d in (("sentinel2_scenes", s2), ("landsat_scenes", ls), ("sentinel1_scenes", s1)):
        d["datetime"] = pd.to_datetime(d.datetime, format="mixed")
        d = d[["item_id"] + [c for c in d.columns if c != "item_id"]]
        assert d.item_id.isin(set(batch)).all()
        d.to_parquet(out / f"{name}.parquet", index=False)
        log(name, d.shape)
    # ---- sheets (all items) and layouts
    lay = {"interpreter": layout(g, lambda k: INTERP_HALF, INTERP_WINS), "standard": layout(g, lambda k: STD_HALF[k], STD_WINS)}
    for kind_dir, src in (("interpreter", IS1 / "interp_sheets"), ("standard", IS1 / "ref_sheets")):
        d = out / "sheets" / kind_dir
        d.mkdir(parents=True, exist_ok=True)
        (out / "sheets" / f"layout_{kind_dir}.json").write_text(json.dumps(lay[kind_dir]), encoding="utf-8")
        if not a.skip_images:
            for i in g.item_id:
                shutil.copy2(src / f"{i}.png", d / f"{i}.png")
        log("sheets", kind_dir, len(list(d.glob("*.png"))))
    # ---- rendered single-date scenes (items with scene series)
    for k, i in enumerate(batch):
        d = out / "scenes" / i
        d.mkdir(parents=True, exist_ok=True)
        sc = pd.read_csv(PKG / i / "canh" / "danh_sach.csv", encoding="utf-8-sig").rename(columns=SCENE_COLS)
        sc["render"] = sc.render.map(RENDER)
        sc["reason"] = sc.reason.map(lambda v: ";".join(REASON[x.strip()] for x in str(v).replace(",", ";").split(";")))
        assert sc.render.notna().all(), i
        sc.to_csv(d / "scenes.csv", index=False)
        if not a.skip_images:
            for fn in sc.file:
                shutil.copy2(PKG / i / "canh" / fn, d / fn)
        if k % 200 == 0:
            log("scenes", k, i)
    log("scene folders", len(batch))
    print("dataset ->", out)


if __name__ == "__main__":
    main()
