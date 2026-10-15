"""Pipeline test with the OFFICIAL 2020 forest-status lots (diễn biến rừng 2020, public WFS of the forestry
department's map) for the 5 Central Highlands provinces.

    python -m src.dossier.official_map build     # pages -> data/processed/vnforest/dbr2020_tn.parquet
    python -m src.dossier.official_map grid      # 500 m systematic points, official class + open maps (GEE)
    python -m src.dossier.official_map report    # 4-case overlay, simulated zoning, silver-label check -> runs/dossier/
    python -m src.dossier.official_map extra     # GFC2020 V1-V4, forest types, natural forest, persistence, rubber on the grid
    python -m src.dossier.official_map versions  # everything that depends on the GFC2020 version

Official class per lot: nggocr 1 = natural forest, 2 = plantation forest, 3 = not forest (DNN agricultural land,
DT1/DT2/DTR bare/regrowth, DK other, MN water...). The lots cover land under forestry planning (and some lots outside).
Same design as the GIZ Quảng Trị pilot (EEAS 2026): systematic 500 m points, 4 cases (both forest / only national /
only GFC2020 / neither), OA/UA/PA, here for the whole Central Highlands and for several open maps.
Simulated zoning on the official base: green = official forest and the open forest maps agree (>= 3 of 4: GFC, TMF,
Hansen, WorldCover) and no loss after 2020; red = official forest with loss after 2020 and a commodity now; yellow =
official forest otherwise. Shares are over official forest area, as published (green 54.2 / yellow 45 / red 0.8).
"""
from __future__ import annotations

import glob
import gzip
import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape

from ..alerts.gee_common import ee_init, to_gdf
from ..common.paths import PROCESSED, RAW, RUNS
from .sample import COFFEE, OUT as DOSSIER

SRC = RAW / "vnforest" / "Xanh_DBR_2020"
DST = PROCESSED / "vnforest"
PROVS = ["daklak", "daknong", "gialai", "kontum", "lamdong"]
KEEP = ["matinh", "tinh", "mahuyen", "huyen", "maxa", "xa", "tk", "khoanh", "lo", "dtich", "nggocr", "ldlr", "maldlr",
        "malr3", "mdsd", "thanhrung", "namtr", "captuoi", "churung", "sldlr"]
CLASS = {1: "natural", 2: "plantation", 3: "nonforest"}


def build():
    DST.mkdir(parents=True, exist_ok=True)
    parts = []
    for p in PROVS:
        files = sorted(glob.glob(str(SRC / p / "page_*.geojson.gz")))
        rows, geoms = [], []
        for f in files:
            for ft in json.load(gzip.open(f, "rt", encoding="utf-8"))["features"]:
                pr = ft["properties"]
                rows.append({k: pr.get(k) for k in KEEP})
                geoms.append(shape(ft["geometry"]) if ft.get("geometry") else None)
        g = gpd.GeoDataFrame(rows, geometry=geoms, crs=4326)
        g["prov"] = p
        print(p, len(files), "pages", len(g), "lots", "done" if (SRC / p / "_DONE").exists() else "(partial)", flush=True)
        parts.append(g)
    g = pd.concat(parts, ignore_index=True)
    g["cls"] = g.nggocr.map(CLASS).fillna("nonforest")
    g = g[g.geometry.notna()]
    g.to_parquet(DST / "dbr2020_tn.parquet")
    print("lots:", len(g), "| area by class (ha):", g.groupby("cls").dtich.sum().round(0).to_dict())


def official_at(points: gpd.GeoDataFrame, lots: gpd.GeoDataFrame) -> pd.DataFrame:
    j = gpd.sjoin(points, lots[["cls", "ldlr", "nggocr", "mdsd", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    return j[["cls", "ldlr", "mdsd"]].rename(columns={"cls": "off_cls", "ldlr": "off_ldlr", "mdsd": "off_mdsd"})


def open_maps_image(ee):
    gfc = ee.Image("JRC/GFC2020_subtypes/V1").select("Map").mask().gt(0).unmask(0).rename("gfc")
    tmf = ee.ImageCollection("projects/JRC/TMF/v1_2024/AnnualChanges").mosaic()
    t20, t24 = tmf.select("Dec2020"), tmf.select("Dec2024")
    tmf_f = t20.eq(1).Or(t20.eq(2)).Or(t20.eq(4)).unmask(0).rename("tmf")
    h = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
    ly = h.select("lossyear").unmask(0)
    han = h.select("treecover2000").gte(30).And(ly.eq(0).Or(ly.gt(20))).unmask(0).rename("han")
    wc = ee.ImageCollection("ESA/WorldCover/v100").first().select("Map").eq(10).unmask(0).rename("wc")
    # loss after the cut-off: Hansen loss 2021-2025, or TMF forest in 2020 -> deforested by 2024, or undisturbed in 2020
    # -> degraded by 2024. (Degraded already in 2020 and still degraded in 2024 is NOT new loss.)
    tmf_loss = tmf_f.And(t24.eq(3)).Or(t20.eq(1).And(t24.eq(2)))
    loss = ly.gte(21).And(ly.lte(25)).Or(tmf_loss).unmask(0).rename("loss")
    raw = ee.Image.cat([t20.unmask(0).rename("tmf20"), t24.unmask(0).rename("tmf24"), ly.rename("lossyear")])
    cof = ee.ImageCollection(COFFEE).filterDate("2024-01-01", "2025-01-01").mosaic().select("probability").unmask(0).rename("coffee24")
    cof20 = ee.ImageCollection(COFFEE).filterDate("2020-01-01", "2021-01-01").mosaic().select("probability").unmask(0).rename("coffee20")
    dw_now = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate("2024-01-01", "2026-01-01").select("label").mode().rename("dw_now")
    return ee.Image.cat([gfc, tmf_f, han, wc, loss, cof, cof20, dw_now, raw])


GFC_VERSIONS = {"V1": True, "V2": True, "V3": False, "V4": False}  # True = ImageCollection on GEE


def extra_image(ee):
    """Layers added after the first runs. Every GFC2020 version: results depend strongly on it (on coffee points V1 calls
    ~60% forest, V4 ~2%), and GFC2020_subtypes/V1 used above behaves like V3. Plus JRC forest-type code, Nature Trace
    natural-forest probability and forest typology, FDP forest persistence and rubber probability (2020, 2024)."""
    b = []
    for v, coll in GFC_VERSIONS.items():
        im = ee.ImageCollection(f"JRC/GFC2020/{v}").mosaic() if coll else ee.Image(f"JRC/GFC2020/{v}")
        b.append(im.select("Map").unmask(0).gt(0).rename(f"gfc{v}"))
    b.append(ee.Image("JRC/GFC2020_subtypes/V1").select("Map").unmask(0).rename("gft_code"))  # 1 natregen, 10 primary, 20 planted
    b.append(ee.ImageCollection("projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection").mosaic()
             .select("B0").unmask(0).rename("ntf"))
    ty = ee.ImageCollection("projects/nature-trace/assets/forest_typology/forest_typology_2020_v1_0_collection").mosaic()
    b.append(ty.select(["PrimaryForest", "NaturallyRegeneratingForest", "PlantedForest", "PlantationForest", "TreeCropsAndAgroforestry"],
                       ["ty_primary", "ty_natregen", "ty_planted", "ty_plantation", "ty_treecrop"]).unmask(0))
    b.append(ee.Image("projects/forestdatapartnership/assets/community_forests/ForestPersistence_2020").select("value").unmask(0).rename("persist"))
    rub = ee.ImageCollection("projects/forestdatapartnership/assets/rubber/model_2026a")
    for y in (2020, 2024):
        b.append(rub.filterDate(f"{y}-01-01", f"{y + 1}-01-01").mosaic().select("probability").unmask(0).rename(f"rubber{y % 100}"))
    return ee.Image.cat(b)


def extra():
    """Sample extra_image() on the existing grid (same points, joined by index) -> grid500_tn_extra.parquet."""
    pts = gpd.read_parquet(DST / "grid500_tn.parquet")[["geometry"]]
    ee = ee_init()
    img = extra_image(ee)
    vals = []
    for i in range(0, len(pts), 4000):
        chunk = pts.iloc[i:i + 4000]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([p.x, p.y]), {"i": int(k)}) for k, p in zip(chunk.index, chunk.geometry)])
        vals.append(pd.DataFrame(to_gdf(img.sampleRegions(fc, scale=10, tileScale=8, geometries=False)).drop(columns="geometry", errors="ignore")))
        print("sampled", i + len(chunk), flush=True)
    v = pd.concat(vals).set_index("i").sort_index()
    v.to_parquet(DST / "grid500_tn_extra.parquet")
    print(v.describe().T[["mean", "max"]].round(3).to_string())


def species_at(points: gpd.GeoDataFrame, lots: gpd.GeoDataFrame) -> pd.Series:
    """Plantation species group of the official lot under each point (None when not a plantation lot)."""
    pats = [("cao su", r"cao su"), ("dieu", r"điều|đào lộn"), ("keo", r"keo"), ("thong", r"thông"), ("bach dan", r"bạch đàn")]
    sp = lots.sldlr.fillna("").str.lower()
    grp = pd.Series("khac/khong ghi", index=lots.index)
    for k, pat in reversed(pats):
        grp[sp.str.contains(pat, regex=True)] = k
    grp[lots.cls != "plantation"] = None
    j = gpd.sjoin(points[["geometry"]], lots.assign(sp=grp)[["sp", "geometry"]], how="left", predicate="within")
    return j[~j.index.duplicated(keep="first")].sp


def versions():
    """Everything that depends on the GFC2020 version, V1..V4 -> runs/dossier/official_map_versions.json."""
    g = gpd.read_parquet(DST / "grid500_tn.parquet").dropna(subset=["gfc"]).join(pd.read_parquet(DST / "grid500_tn_extra.parquet"))
    g = g.dropna(subset=["gfcV4"])
    g["off_forest"] = g.off_cls.isin(["natural", "plantation"]).astype(int)
    lots = gpd.read_parquet(DST / "dbr2020_tn.parquet", columns=["cls", "sldlr", "geometry"])
    g["sp"] = species_at(g, lots)
    HA = 25.0
    cof = g.coffee24 >= 0.3
    rub = g.rubber24 >= 0.5
    res = {"n_points": len(g), "coffee24_ge_0.3_ha": int(cof.sum() * HA), "rubber24_ge_0.5_ha": int(rub.sum() * HA),
           "subtypesV1_equals": {v: round(float((g.gfc == g[f"gfc{v}"]).mean()), 4) for v in GFC_VERSIONS},
           "by_version": {}}
    for v in GFC_VERSIONS:
        f = g[f"gfc{v}"].astype(int)
        case = pd.Series(np.select([(g.off_forest == 1) & (f == 1), (g.off_forest == 1) & (f == 0), (g.off_forest == 0) & (f == 1)],
                                   ["both_forest", "only_national", "only_gfc"], "neither"), index=g.index)
        of = g.off_forest == 1
        agree = f + g.tmf + g.han + g.wc
        commodity = (g.coffee24 >= 0.5) | (g.dw_now == 4)
        red = of & (g.loss == 1) & commodity
        green = of & ~red & (agree >= 3) & (g.loss == 0)
        yel = of & ~red & ~green
        flag = pd.Series(np.select([of & (f == 1), of, f == 1, g.loss == 1], ["both", "official_only", "gfc_only", "loss_only"], "clean"), index=g.index)
        res["by_version"][v] = {
            "forest_share_all": round(float(f.mean()), 3),
            "four_cases": case.value_counts(normalize=True).round(3).to_dict(),
            "only_gfc_ha": int((case == "only_gfc").sum() * HA),
            "only_gfc_coffee_share": round(float(cof[case == "only_gfc"].mean()), 3),
            "coffee_called_forest": round(float(f[cof].mean()), 3),
            "coffee_called_forest_by_prov": f[cof].groupby(g.prov[cof]).mean().round(3).to_dict(),
            "coffee_flags": flag[cof].value_counts(normalize=True).round(3).to_dict(),
            "coffee_flagged_ha": int((flag[cof] != "clean").sum() * HA),
            "rubber_called_forest": round(float(f[rub].mean()), 3),
            "official_plantation_detected": round(float(f[g.off_cls == "plantation"].mean()), 3),
            "official_natural_detected": round(float(f[g.off_cls == "natural"].mean()), 3),
            "official_plantation_detected_by_species": f[g.off_cls == "plantation"].groupby(g.sp[g.off_cls == "plantation"]).mean().round(3).to_dict(),
            "zoning_on_official_forest": {"green": round(float(green[of].mean()) * 100, 1), "yellow": round(float(yel[of].mean()) * 100, 1),
                                          "red": round(float(red[of].mean()) * 100, 1)},
            "yellow_ha": int(yel.sum() * HA),
            "yellow_composition": {"official_class": g.off_cls[yel].value_counts(normalize=True).round(3).to_dict(),
                                   "coffee24_ge_0.3": round(float(cof[yel].mean()), 3), "rubber24_ge_0.5": round(float(rub[yel].mean()), 3),
                                   "loss_without_commodity": round(float((g.loss[yel] == 1).mean()), 3),
                                   "species_if_plantation": g.sp[yel & (g.off_cls == "plantation")].value_counts(normalize=True).round(3).to_dict()},
        }
        # zoning over the UNION of forest (national OR GFC says forest): disagreement in either direction is yellow
        dom = of | (f == 1)
        red_u = dom & (g.loss == 1) & commodity
        for rule, green_u in (("strict", dom & ~red_u & of & (f == 1) & (agree >= 3) & (g.loss == 0)),
                              ("relaxed", dom & ~red_u & of & (f == 1) & (g.loss == 0))):
            yel_u = dom & ~red_u & ~green_u
            res["by_version"][v][f"zoning_union_{rule}"] = {
                "domain_ha": int(dom.sum() * HA), "green": round(float(green_u[dom].mean()) * 100, 1),
                "yellow": round(float(yel_u[dom].mean()) * 100, 1), "red": round(float(red_u[dom].mean()) * 100, 1),
                "yellow_coffee_share": round(float(cof[yel_u].mean()), 3), "coffee_in_yellow_ha": int((yel_u & cof).sum() * HA),
                "yellow_only_gfc_share": round(float((~of & (f == 1))[yel_u].mean()), 3),
                "yellow_by_prov": yel_u[dom].groupby(g.prov[dom]).mean().round(3).to_dict()}
    # what each disagreement mechanism is made of (FDP coffee / rubber, Nature Trace typology argmax, natural-forest prob)
    ty = g[["ty_primary", "ty_natregen", "ty_plantation", "ty_treecrop"]]
    forty = pd.Series(np.where(ty.max(axis=1) < 50, "none", ty.idxmax(axis=1).str.replace("ty_", "")), index=g.index)
    of = g.off_forest == 1

    def comp(m):
        c, r = cof[m], rub[m] & ~cof[m]
        return {"area_ha": int(m.sum() * HA), "fdp_coffee": round(float(c.mean()), 3), "fdp_rubber": round(float(r.mean()), 3),
                "forty_argmax": forty[m].value_counts(normalize=True).round(3).to_dict(),
                "natural_forest_prob_ge_0.5": round(float((g.ntf[m] >= 125).mean()), 3),
                "official": g.off_cls[m].value_counts(normalize=True).round(3).to_dict()}
    res["mechanisms"] = {"V1_forest_removed_in_V4_on_national_nonforest": comp(~of & (g.gfcV1 == 1) & (g.gfcV4 == 0)),
                         "V4_forest_on_national_nonforest": comp(~of & (g.gfcV4 == 1)),
                         "national_forest_not_V4": comp(of & (g.gfcV4 == 0))}
    res["published_13_prov"] = {"green": 54.2, "yellow": 45.0, "red": 0.8, "source": "Vietnam Law Magazine 06/01/2026"}
    (RUNS / "dossier").mkdir(parents=True, exist_ok=True)
    (RUNS / "dossier" / "official_map_versions.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


PROV_KEYS = {"dak lak": "daklak", "dak nong": "daknong", "gia lai": "gialai", "kon tum": "kontum", "lam dong": "lamdong"}


def grid():
    """500 m points over the WHOLE area of the 5 provinces (GADM, pre-2025 units), not only on the lots: most coffee
    lies outside the forestry lots, where the national map implicitly says 'not forest' (off_cls = outside_lots)."""
    lots = gpd.read_parquet(DST / "dbr2020_tn.parquet")
    prov = gpd.read_file("gazetteer/old_units.gpkg", layer="province")
    prov = prov[prov.province_key.isin(PROV_KEYS)].assign(prov=lambda d: d.province_key.map(PROV_KEYS)).to_crs(32648)
    xmin, ymin, xmax, ymax = prov.total_bounds
    xs, ys = np.meshgrid(np.arange(xmin + 250, xmax, 500), np.arange(ymin + 250, ymax, 500))
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs.ravel(), ys.ravel()), crs=32648)
    pts = gpd.sjoin(pts, prov[["prov", "geometry"]], how="inner", predicate="within")
    pts = pts[~pts.index.duplicated()].drop(columns="index_right").to_crs(4326).reset_index(drop=True)
    pts = pts.join(official_at(pts, lots))
    pts["in_lots"] = pts.off_cls.notna().astype(int)
    pts["off_cls"] = pts.off_cls.fillna("outside_lots")
    print("grid points:", len(pts), "| on lots:", int(pts.in_lots.sum()), flush=True)
    ee = ee_init()
    img = open_maps_image(ee)
    vals = []
    for i in range(0, len(pts), 4000):
        chunk = pts.iloc[i:i + 4000]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([p.x, p.y]), {"i": int(k)}) for k, p in zip(chunk.index, chunk.geometry)])
        vals.append(pd.DataFrame(to_gdf(img.sampleRegions(fc, scale=10, tileScale=8, geometries=False)).drop(columns="geometry", errors="ignore")))
        print("sampled", i + len(chunk), flush=True)
    v = pd.concat(vals).set_index("i")
    out = pts.join(v)
    out.to_parquet(DST / "grid500_tn.parquet")
    print(out[["gfc", "tmf", "han", "wc"]].mean().round(3).to_dict())


def report():
    g = gpd.read_parquet(DST / "grid500_tn.parquet").dropna(subset=["gfc"])
    g["off_forest"] = g.off_cls.isin(["natural", "plantation"]).astype(int)
    g["coffee20_hi"] = (g.coffee20 >= 0.5).astype(int)
    HA = 25.0  # one 500 m point stands for 25 ha
    res = {"n_points": len(g), "n_on_lots": int(g.in_lots.sum()), "area_ha_5prov": len(g) * HA}
    # 4 cases, as in the GIZ pilot, over the whole area (outside the lots = national 'not forest')
    case = np.select([(g.off_forest == 1) & (g.gfc == 1), (g.off_forest == 1) & (g.gfc == 0), (g.off_forest == 0) & (g.gfc == 1)],
                     ["both_forest", "only_national", "only_gfc"], "neither")
    g["case"] = case
    res["four_cases_TN"] = g.case.value_counts(normalize=True).round(3).to_dict()
    res["four_cases_on_lots_only"] = g[g.in_lots == 1].case.value_counts(normalize=True).round(3).to_dict()
    res["four_cases_by_prov"] = pd.crosstab(g.prov, g.case, normalize="index").round(3).to_dict("index")
    # A-E breakdown (PHAN_TICH_BAI_TOAN.md §3): only_gfc split by coffee (FDP 2020 prob >= 0.5) and inside/outside lots
    oc = g[g.case == "only_gfc"]
    res["only_gfc_breakdown"] = {
        "points": len(oc), "area_ha": len(oc) * HA,
        "C1_coffee_share": round(float(oc.coffee20_hi.mean()), 3),
        "C1_coffee_area_ha": int(oc.coffee20_hi.sum() * HA),
        "outside_lots_share": round(float((oc.in_lots == 0).mean()), 3),
        "by_official_class": oc.off_cls.value_counts(normalize=True).round(3).to_dict(),
        "coffee_share_outside_lots": round(float(oc[oc.in_lots == 0].coffee20_hi.mean()), 3),
        "coffee_share_on_lots_nonforest": round(float(oc[oc.in_lots == 1].coffee20_hi.mean()), 3) if (oc.in_lots == 1).any() else None}
    dnn = g[g.off_ldlr == "DNN"]
    res["E_agri_land_in_lots"] = {"points": len(dnn), "area_ha": len(dnn) * HA,
                                  "coffee20_share": round(float(dnn.coffee20_hi.mean()), 3),
                                  "coffee24_share": round(float((dnn.coffee24 >= 0.5).mean()), 3),
                                  "gfc_forest_share": round(float(dnn.gfc.mean()), 3)}
    res["coffee20_where"] = g[g.coffee20_hi == 1].off_cls.value_counts(normalize=True).round(3).to_dict()
    res["coffee20_called_forest"] = {m: round(float(g.loc[g.coffee20_hi == 1, m].mean()), 3) for m in ("gfc", "tmf", "han", "wc")}
    # FDP 2020 probabilities run low (>= 0.5 gives 132k ha vs ~640k ha in the statistics); 2024 >= 0.3 matches the
    # statistics (646k ha), so the coffee-centric view uses coffee24 >= 0.3 (perennial crop; 2.5% of it has loss after 2020)
    res["coffee_area_ha_by_threshold"] = {f"{y}_{t}": int((g[f"coffee{y}"] >= t).sum() * HA) for y in (20, 24) for t in (0.3, 0.5, 0.7)}
    c = g[g.coffee24 >= 0.3]
    flag = pd.Series(np.select([(c.off_forest == 1) & (c.gfc == 1), c.off_forest == 1, c.gfc == 1, c.loss == 1],
                               ["both", "official_only", "gfc_only", "loss_only"], "clean"), index=c.index)
    res["coffee24_flags"] = {"area_ha": len(c) * HA, "shares": flag.value_counts(normalize=True).round(3).to_dict(),
                             "flagged_ha": int((flag != "clean").sum() * HA),
                             "by_prov": pd.crosstab(c.prov, flag, normalize="index").round(3).to_dict("index"),
                             "called_forest_by": {m: round(float(c[m].mean()), 3) for m in ("gfc", "tmf", "han", "wc")},
                             "where": c.off_cls.value_counts(normalize=True).round(3).to_dict(),
                             "on_DNN_share": round(float((c.off_ldlr == "DNN").mean()), 3)}
    # accuracy of each open map against the official map (official = reference), on the lots only (outside the lots
    # the national map has no real statement), and by official class
    for m in ("gfc", "tmf", "han", "wc"):
        gl = g[g.in_lots == 1]
        y, p = gl.off_forest, gl[m].astype(int)
        tp, fp, fn, tn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum())
        res[f"{m}_vs_official"] = {"OA": round((tp + tn) / len(gl), 3), "UA_forest": round(tp / max(tp + fp, 1), 3),
                                   "PA_forest": round(tp / max(tp + fn, 1), 3),
                                   "detect_natural": round(float(gl.loc[gl.off_cls == "natural", m].mean()), 3),
                                   "detect_plantation": round(float(gl.loc[gl.off_cls == "plantation", m].mean()), 3),
                                   "forest_on_DNN_agri": round(float(gl.loc[gl.off_ldlr == "DNN", m].mean()), 3),
                                   "forest_outside_lots": round(float(g.loc[g.in_lots == 0, m].mean()), 3)}
    # simulated zoning on the official forest base
    of = g[g.off_forest == 1].copy()
    agree = of[["gfc", "tmf", "han", "wc"]].sum(axis=1)
    commodity = (of.coffee24 >= 0.5) | (of.dw_now == 4)
    red = (of.loss == 1) & commodity
    green = ~red & (agree >= 3) & (of.loss == 0)
    res["zoning_on_official_forest"] = {"green": round(float(green.mean()) * 100, 1), "yellow": round(float((~red & ~green).mean()) * 100, 1),
                                        "red": round(float(red.mean()) * 100, 1), "published_13_prov": {"green": 54.2, "yellow": 45.0, "red": 0.8}}
    yel = ~red & ~green
    res["yellow_composition"] = {"area_ha": int(yel.sum() * HA),
                                 "by_official_class": of[yel].off_cls.value_counts(normalize=True).round(3).to_dict(),
                                 "n_open_maps_saying_forest": agree[yel].value_counts(normalize=True).sort_index().round(3).to_dict(),
                                 "loss_without_commodity": round(float((of[yel].loss == 1).mean()), 3),
                                 "coffee24_ge_0.3": round(float((of[yel].coffee24 >= 0.3).mean()), 3),
                                 "gfc_forest": round(float(of[yel].gfc.mean()), 3)}
    res["zoning_by_prov"] = of.assign(z=np.select([red, green], ["red", "green"], "yellow")).groupby("prov").z.value_counts(normalize=True).unstack().round(3).to_dict("index")
    # silver-label check on the 140 dossier items (plots: forest at the cut-off; alerts: plantation harvest vs other)
    lots = gpd.read_parquet(DST / "dbr2020_tn.parquet")
    items = pd.concat([gpd.read_parquet(DOSSIER / "sample_alerts.parquet")[["item_id", "geometry"]],
                       gpd.read_parquet(DOSSIER / "sample_plots.parquet")[["item_id", "geometry"]]]).to_crs(4326)
    cen = gpd.GeoDataFrame(items[["item_id"]], geometry=items.geometry.representative_point(), crs=4326).reset_index(drop=True)
    cen = cen.join(official_at(cen, lots))
    lab = pd.read_csv("annotation/dossier_reference/labels_ai_silver.csv")
    tri = pd.read_parquet(DOSSIER / "triage.parquet")[["item_id", "kind", "gfc_forest", "gfc_planted", "tmf_plantation"]]
    d = cen.merge(lab[["item_id", "driver", "cutoff", "cleared"]], on="item_id").merge(tri, on="item_id")
    P = d[d.kind == "plot"].copy()
    P["silver_forest"] = P.cutoff.isin(["natural_forest", "tree_plantation"]).map({True: 1, False: 0})
    P.loc[P.cutoff.isin(["mixed", "unclear"]), "silver_forest"] = np.nan
    Pd = P.dropna(subset=["silver_forest"])
    res["plots_forest2020_agreement_with_silver"] = {
        "n": len(Pd), "official": round(float((Pd.off_cls.isin(["natural", "plantation"]).astype(int) == Pd.silver_forest).mean()), 3),
        "gfc2020": round(float(((Pd.gfc_forest >= 0.5).astype(int) == Pd.silver_forest).mean()), 3),
        "official_class_of_silver_coffee": P[P.cutoff == "coffee_or_orchard"].off_cls.fillna("outside_lots").value_counts().to_dict()}
    A = d[(d.kind == "alert") & (d.driver != "unsure")].copy()
    A["silver_C2"] = (A.driver == "C2").astype(int)
    res["alerts_plantation_signal_vs_silver_C2"] = {
        "n": len(A), "silver_C2": int(A.silver_C2.sum()),
        "official_plantation_hit": round(float(A.loc[A.silver_C2 == 1, "off_cls"].eq("plantation").mean()), 3) if A.silver_C2.any() else None,
        "official_plantation_false": round(float(A.loc[A.silver_C2 == 0, "off_cls"].eq("plantation").mean()), 3),
        "global_plantation_hit": round(float(((A.gfc_planted >= 0.5) | (A.tmf_plantation >= 0.5))[A.silver_C2 == 1].mean()), 3) if A.silver_C2.any() else None,
        "global_plantation_false": round(float(((A.gfc_planted >= 0.5) | (A.tmf_plantation >= 0.5))[A.silver_C2 == 0].mean()), 3),
        "official_class_counts": A.off_cls.fillna("outside_lots").value_counts().to_dict()}
    (RUNS / "dossier").mkdir(parents=True, exist_ok=True)
    (RUNS / "dossier" / "official_map_report.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    {"build": build, "grid": grid, "report": report, "extra": extra, "versions": versions}[sys.argv[1]]()
