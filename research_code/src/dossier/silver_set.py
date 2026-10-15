"""Silver set s1 (11/10/2026): more items, silver labels from a larger VLM kept apart from the system under test.

    python -m src.dossier.silver_set sample                 # items + DEV/TEST split, before any label exists
    python -m src.dossier.silver_set sheets [--waves 0-14]  # interpreter sheets (labeler) + standard sheets (system)
    python -m src.dossier.silver_set label [--hours 6]      # Qwen3.5-35B-A3B on both GPUs of the lab server, wave by wave
    python -m src.dossier.silver_set truth                  # five answers per item -> silver truth table
    python -m src.dossier.silver_set evidence [--waves ...] # GEE evidence for the system under test -> triage.parquet
    python -m src.dossier.silver_set report                 # the pre-registered endpoints and verdict (after eval)

Separation (the system under test must not be graded on its own information):
- the silver labeler (Qwen3.5-35B-A3B, GPTQ-Int4, both GPUs) sees ONLY an interpreter sheet: Sentinel-2 true and
  false colour at the cut-off (dry 2019, dry 2020, 11/2020-04/2021) and from before the alert to five years after
  it, plus the NDVI series. No evidence card, no map layer, no stratum, no output of the system under test. Its
  instructions are the human guide (annotation/GAN_NHAN_EUDR_V2.md, section 5), answered nine times with the
  options in different orders (answer-order bias was measured in v3), greedy decoding; a label needs 5 of 9;
- the system under test (legal_vlm v2, Qwen3.5-9B, frozen on 11/10/2026) gets what dot0 got: the standard sheet
  (-12..+20 months around the alert) and the evidence card;
- DEV / TEST by 0.2 degree blocks, drawn with a fixed seed from coordinates alone before any label exists: only
  DEV may train or tune (LightGBM baseline); TEST is scored once. dot0 (140 items) shaped the v2 design and is
  never TEST.
Labeler and system share a model family, so correlated errors remain possible: human labels stay the reference.
Items: GFW integrated-alert patches first alerted 2021-2024 (>= 0.3 ha; year x area-bin strata) and 1 ha coffee
plots (FDP coffee 2024 >= 0.7; strata B1-B3 as in sample.py), processed in waves that keep every stratum balanced,
so any number of finished waves is a stratified sample (weights = stratum population / items processed).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("DOSSIER_SET", "s1")  # this module always works on the silver set

import geopandas as gpd  # noqa: E402
import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from ..alerts.gee_common import ee_init, to_gdf  # noqa: E402
from ..common.config import load  # noqa: E402
from ..common.paths import GAZETTEER, INTERIM, PROCESSED, RUNS  # noqa: E402
from . import ref_sheets as R  # noqa: E402
from .sample import COFFEE, OUT, TN  # noqa: E402

assert OUT.name == "s1", OUT
WORK = INTERIM / "dossier" / "s1"
INTERP = WORK / "interp_sheets"           # labeler only
SILVER_RAW = WORK / "silver_raw"          # answers of the labeler, one file per GPU-server batch
SEED = 2026
YEARS = range(2021, 2025)
AREA_BINS = [(0.3, 0.5), (0.5, 1.0), (1.0, np.inf)]
N_ALERT_CELL = 100                        # per year x area bin -> 1200 alerts
PLOT_N = {1: 240, 2: 220, 3: 140}         # B1 forest 2020 + loss 2021-25, B2 forest 2020 no loss, B3 not forest no loss
WAVE_ALERT, WAVE_PLOT = 8, {1: 16, 2: 14, 3: 10}  # per stratum and wave: 96 alerts + 40 plots
BLOCK, DEV_SHARE = 0.2, 0.3
LABELER = "Qwen3.5-35B-A3B-GPTQ-Int4"
N_ORDERS = 9  # amendment A1 (PREREG_s1.md): 5 -> 9 after the smoke test showed answers depend on option order
HALF = 0.0045                             # interpreter view ~1 km wide (standard sheets keep their own size)
ALERT_IW = [("dry 2019", "2019-01-01", "2019-04-30"), ("dry 2020", "2020-01-01", "2020-04-30"),
            ("CUT-OFF 11/20-04/21", "2020-11-01", "2021-04-30"), ("-4..-1 mo", -120, -20), ("ALERT 0..+2 mo", 0, 60),
            ("+3..+6 mo", 90, 180), ("+10..+14 mo", 300, 420), ("+2 yr", 700, 820), ("+3 yr", 1065, 1185),
            ("dry 2026", "2026-01-01", "2026-04-30")]
PLOT_IW = [(f"dry {y}", f"{y}-01-01", f"{y}-04-30") for y in (2017, 2018, 2019, 2020)] + \
          [("CUT-OFF 11/20-04/21", "2020-11-01", "2021-04-30")] + [(f"dry {y}", f"{y}-01-01", f"{y}-04-30") for y in range(2022, 2027)]


# ------------------------------------------------------------------------------------------------ sample + split
def sample_alerts() -> gpd.GeoDataFrame:
    p = gpd.read_parquet(PROCESSED / "alerts" / "patches_nrt.parquet")
    fd = pd.to_datetime(p.first_date)
    p = p[p.province_key.isin(TN) & (fd >= "2021-01-01") & (fd <= "2024-12-31") & (p.area_ha >= 0.3)].copy()
    p["fy"] = pd.to_datetime(p.first_date).dt.year
    rng = np.random.default_rng(SEED)
    parts = []
    for y in YEARS:
        for lo, hi in AREA_BINS:
            s = p[(p.fy == y) & (p.area_ha >= lo) & (p.area_ha < hi)]
            t = s.sample(min(N_ALERT_CELL, len(s)), random_state=int(rng.integers(1e9))).copy()
            t["stratum"], t["pop"], t["rank"] = f"{y}_{lo}-{hi if np.isfinite(hi) else 'inf'}", len(s), np.arange(len(t))
            parts.append(t)
    a = pd.concat(parts, ignore_index=True)
    a["wave"] = a["rank"] // WAVE_ALERT
    a = a.sort_values(["wave", "stratum", "rank"]).reset_index(drop=True)
    a["item_id"], a["kind"] = [f"A{1001 + i}" for i in range(len(a))], "alert"
    return gpd.GeoDataFrame(a.drop(columns=["fy"]), geometry="geometry", crs=p.crs)


def sample_plots() -> gpd.GeoDataFrame:
    ee = ee_init()
    c = load("gee")
    prov = gpd.read_file(GAZETTEER / "old_units.gpkg", layer="province")
    region = ee.Geometry(prov[prov.province_key.isin(TN)].to_crs(4326).union_all().simplify(0.01).__geo_interface__)
    cof24 = ee.ImageCollection(COFFEE).filterBounds(region).filterDate("2024-01-01", "2025-01-01").mosaic().select("probability").gte(0.7)
    forest20 = ee.Image(c["context"]["jrc_gfc2020_subtypes"]).select("Map").mask().gt(0).unmask(0)
    ly = ee.Image(c["forest_masks"]["hansen"]).select("lossyear").unmask(0)
    loss_post = ly.gte(21).And(ly.lte(25))
    cls = (ee.Image(0).where(forest20.And(loss_post), 1).where(forest20.And(loss_post.Not()), 2)
           .where(forest20.Not().And(loss_post.Not()), 3).updateMask(cof24).rename("stratum").toInt())
    fc = cls.stratifiedSample(numPoints=0, classBand="stratum", region=region, scale=30, seed=SEED, classValues=list(PLOT_N),
                              classPoints=list(PLOT_N.values()), geometries=True, tileScale=16)
    pts = to_gdf(fc).to_crs(32648)
    old = gpd.read_parquet(PROCESSED / "dossier" / "sample_plots.parquet").to_crs(32648)  # dot0 plots stay out
    pts = pts[pts.geometry.distance(old.union_all()) > 300].copy()
    pts["rank"] = pts.groupby("stratum").cumcount()
    pts["wave"] = [r // WAVE_PLOT[int(s)] for r, s in zip(pts["rank"], pts.stratum)]
    g = gpd.GeoDataFrame({"stratum": "B" + pts.stratum.astype(int).astype(str), "rank": pts["rank"].values, "wave": pts.wave.values},
                         geometry=pts.geometry.buffer(50, cap_style="square").values, crs=32648).to_crs(4326)
    g = g.sort_values(["wave", "stratum", "rank"]).reset_index(drop=True)
    g["item_id"], g["kind"], g["area_ha"], g["pop"] = [f"P{1001 + i}" for i in range(len(g))], "plot", 1.0, np.nan
    return g


def do_sample():
    OUT.mkdir(parents=True, exist_ok=True)
    A, B = sample_alerts(), sample_plots()
    A.to_parquet(OUT / "sample_alerts.parquet")
    B.to_parquet(OUT / "sample_plots.parquet")
    cols = ["item_id", "kind", "stratum", "pop", "wave"]
    it = pd.concat([A[cols + ["geometry"]], B[cols + ["geometry"]]], ignore_index=True)
    c = gpd.GeoSeries(it.geometry, crs=4326).to_crs(32648).centroid.to_crs(4326)
    it["lon"], it["lat"] = c.x.round(6), c.y.round(6)
    it["block"] = np.floor(it.lon / BLOCK).astype(int).astype(str) + "_" + np.floor(it.lat / BLOCK).astype(int).astype(str)
    blocks = sorted(it.block.unique())
    random.Random(SEED).shuffle(blocks)
    dev = set(blocks[:round(DEV_SHARE * len(blocks))])
    it["split"] = np.where(it.block.isin(dev), "dev", "test")
    it.drop(columns="geometry").to_csv(OUT / "items_s1.csv", index=False)
    print(f"alerts {len(A)} | plots {len(B)} | waves {int(it.wave.max()) + 1} | blocks {len(blocks)} (dev {len(dev)}) |",
          it.groupby(["kind", "split"]).size().to_dict())


def items(waves=None) -> gpd.GeoDataFrame:
    """s1 items, plus the 140 dot0 items as wave -1 (silver labels there are checked against the earlier silver set)."""
    parts = []
    for d, w in ((OUT, None), (PROCESSED / "dossier", -1)):
        A = gpd.read_parquet(d / "sample_alerts.parquet").to_crs(4326)
        B = gpd.read_parquet(d / "sample_plots.parquet").to_crs(4326)
        if w is not None:
            A["wave"], B["wave"] = w, w
        parts += [A[["item_id", "kind", "wave", "area_ha", "first_date", "geometry"]], B[["item_id", "kind", "wave", "area_ha", "geometry"]]]
    g = pd.concat(parts, ignore_index=True)
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=4326)
    return g[g.wave.isin(waves)] if waves is not None else g


# ------------------------------------------------------------------------------------------------ sheets
# Earth Engine calls in flight across all threads: the project is in restricted mode (monthly noncommercial quota
# used up, 11/10/2026), where more parallel calls only buy 429s
GEE_SLOTS = threading.BoundedSemaphore(int(os.environ.get("GEE_SLOTS", "4")))


def chip1(ee, geom, start, end, bands, vis, half) -> Image.Image | None:
    """One Sentinel-2 median chip with the outline: one Earth Engine call; an empty window gives None. A 429 (too
    many requests) waits and retries the same chip instead of failing the whole sheet."""
    if pd.Timestamp(start) > R.TODAY:
        return None
    region = ee.Geometry(geom.centroid.buffer(half).envelope.__geo_interface__)
    img = R.s2(ee, region, str(start), str(min(pd.Timestamp(end), R.TODAY).date())).median().select(bands).visualize(**vis)
    outline = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(ee.Geometry(geom.__geo_interface__))]), 1, 2)
    for k in range(8):
        with GEE_SLOTS:
            try:
                url = img.blend(outline.visualize(palette=["ffff00"])).getThumbURL({"region": region, "dimensions": 192, "format": "png"})
                r = httpx.get(url, timeout=180)
            except Exception as e:  # noqa: BLE001
                if "did not match any bands" in str(e) or "no bands" in str(e).lower():
                    return None  # no clear scene in the window
                if "429" not in str(e) and "Too Many" not in str(e):
                    raise
                r = None
        if r is not None and r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        if r is not None and r.status_code != 429:
            raise RuntimeError(f"thumbnail {r.status_code}")
        time.sleep(min(120, 5 * 2 ** k) * (0.75 + 0.5 * random.random()))  # back off, with jitter
    raise RuntimeError("thumbnail 429 after 8 tries")


def strip(ee, geom, frames, half) -> list[Image.Image | None]:
    """All chips of a sheet in ONE Earth Engine call (a filmstrip) instead of one call each (amendment A2: the
    project is in restricted mode). Same median composite, visualisation, outline and box as chip1; a window with
    no scene comes back white and becomes None ("no image"), as size() == 0 did."""
    region = ee.Geometry(geom.centroid.buffer(half).envelope.__geo_interface__)
    outline = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(ee.Geometry(geom.__geo_interface__))]), 1, 2).visualize(palette=["ffff00"])
    white = ee.Image.constant([255, 255, 255]).toUint8().rename(["vis-red", "vis-green", "vis-blue"])
    imgs, idx = [], []
    for n, (start, end, bands, vis) in enumerate(frames):
        if pd.Timestamp(start) > R.TODAY:
            continue
        col = R.s2(ee, region, str(start), str(min(pd.Timestamp(end), R.TODAY).date()))
        imgs.append(ee.Image(ee.Algorithms.If(col.size().gt(0), col.median().select(bands).visualize(**vis).blend(outline), white)))
        idx.append(n)
    out: list[Image.Image | None] = [None] * len(frames)
    if not imgs:
        return out
    for k in range(8):
        with GEE_SLOTS:
            try:
                url = ee.ImageCollection(imgs).getFilmstripThumbURL({"region": region, "dimensions": 192, "format": "png"})
                r = httpx.get(url, timeout=300)
            except Exception as e:  # noqa: BLE001
                if "429" not in str(e) and "Too Many" not in str(e):
                    raise
                r = None
        if r is not None and r.status_code == 200:
            break
        if r is not None and r.status_code != 429:
            raise RuntimeError(f"filmstrip {r.status_code}")
        time.sleep(min(120, 5 * 2 ** k) * (0.75 + 0.5 * random.random()))
    else:
        raise RuntimeError("filmstrip 429 after 8 tries")
    film = Image.open(io.BytesIO(r.content)).convert("RGB")
    vertical = film.height >= film.width
    step = (film.height if vertical else film.width) / len(imgs)
    for j, n in enumerate(idx):
        box = (0, round(j * step), film.width, round((j + 1) * step)) if vertical else (round(j * step), 0, round((j + 1) * step), film.height)
        fr = film.crop(box)
        if min(fr.getextrema(), key=lambda e: e[0])[0] >= 250:  # uniformly white: no scene in the window
            continue
        out[n] = fr
    return out


def compose(ee, geom, kind, t0, wins, half, ts, path: Path, both_marks: bool):
    W, H, top = 192, 192, 18
    sheet = Image.new("RGB", (len(wins) * W, 2 * (H + top) + 240), "white")
    d, font = ImageDraw.Draw(sheet), ImageFont.load_default()
    cells = [(row, col, lab, name, *(((t0 + timedelta(days=a)).date(), (t0 + timedelta(days=b)).date()) if isinstance(a, int) else (a, b)),
              bands, vis) for row, (bands, vis, name) in enumerate(R.ROWS) for col, (lab, a, b) in enumerate(wins)]
    ims = strip(ee, geom, [(c[4], c[5], c[6], c[7]) for c in cells], half)  # amendment A2: one call for every chip
    for (row, col, lab, name, *_), im in zip(cells, ims):
        x, y = col * W, row * (H + top)
        d.text((x + 3, y + 3), f"{lab} | {name[:5]}", fill="black", font=font)
        if im is not None:
            sheet.paste(im.resize((W, H)), (x, y + top))
        else:
            d.text((x + 40, y + top + 90), "no image", fill="red", font=font)
    cut, alert = [("EUDR cut-off", pd.Timestamp("2020-12-31"))], [("first alert", pd.Timestamp(t0))] if kind == "alert" else []
    marks = cut + alert if both_marks else (alert or cut)  # standard sheet: exactly as dot0 (ref_sheets.make)
    sheet.paste(R.chart(ts, marks, len(wins) * W), (0, 2 * (H + top)))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def one_item(ee, r, which: str, cached: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """which = "interp" (the labeler's view; labelling can start on it) or "standard" (the frozen sheet of the system
    under test: same windows and size as dot0); the NDVI series of the first pass is reused by the second."""
    t0 = pd.Timestamp(r.first_date) if r.kind == "alert" else None
    fi, fs = INTERP / f"{r.item_id}.png", R.REF / f"{r.item_id}.png"
    if (which == "interp" and fi.exists()) or (which == "standard" and (fs.exists() or r.wave == -1)):
        return None  # dot0 standard sheets exist already (data/interim/dossier/ref_sheets)
    if cached is not None and (cached.item_id == r.item_id).any():
        ts = cached[cached.item_id == r.item_id].drop(columns="item_id")
    else:
        with GEE_SLOTS:
            ts = R.ndvi_series(ee, r.geometry)
    if which == "interp":
        compose(ee, r.geometry, r.kind, t0, ALERT_IW if r.kind == "alert" else PLOT_IW, HALF, ts, fi, True)
    else:
        wins = [(lab, a, b) for lab, a, b in (R.ALERT_WINS if r.kind == "alert" else R.PLOT_WINS)]
        compose(ee, r.geometry, r.kind, t0, wins, 0.0036 if r.kind == "alert" else 0.0027, ts, fs, False)
    return ts.assign(item_id=r.item_id) if not ts.empty else None


def do_sheets(waves, workers: int = 16, which: str = "interp"):
    ee = ee_init()
    g = items()
    for w in waves:
        part = g[g.wave == w]
        if not len(part):
            continue
        tag = "dot0" if w == -1 else f"{w:02d}"
        while which == "standard" and w != -1 and not (INTERP / f"wave_{tag}.done").exists():
            time.sleep(60)  # wait for the interpreter pass of this wave: its NDVI series is reused here
        t0 = time.time()
        cf = WORK / "ndvi_series" / f"wave_{w}.parquet"
        cached = pd.read_parquet(cf) if which == "standard" and cf.exists() else None

        def job(r):
            for k in range(4):
                try:
                    return one_item(ee, r, which, cached)
                except Exception as e:  # noqa: BLE001 - transient Earth Engine / HTTP errors
                    last = e
                    time.sleep(10 * (k + 1))
            print("FAILED", r.item_id, repr(last)[:160], flush=True)
            return None
        with ThreadPoolExecutor(workers) as ex:
            series = [s for s in ex.map(job, list(part.itertuples())) if s is not None]
        if series and which == "interp":
            cf.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(series, ignore_index=True).to_parquet(cf)
        folder = INTERP if which == "interp" else R.REF
        ok = sum((folder / f"{i}.png").exists() for i in part.item_id)
        (folder / f"wave_{tag}.done").write_text(json.dumps({"items": len(part), "sheets": ok, "s": round(time.time() - t0)}))
        print(f"wave {w}: {ok}/{len(part)} {which} sheets, {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------ silver labeler
LAND_OPTS = [("rừng_tự_nhiên", "rừng thường xanh, rừng khộp, hỗn giao, tre nứa, kể cả rừng nghèo hay đang phục hồi; tán không đều, cây to nhỏ "
              "xen nhau, không có hàng lối, liền với khu rừng lớn; rừng khộp tán thưa, nền cỏ, rụng lá mùa khô"),
             ("rừng_trồng_lấy_gỗ", "keo, bạch đàn, thông trồng để lấy gỗ; lô hình học, tán đều tuổi và dày, có các ô khai thác theo luân kỳ"),
             ("cà_phê_hoặc_nông_lâm_kết_hợp", "cà phê (kể cả dưới cây che bóng), hồ tiêu, vườn có cây che bóng; hàng bụi đều, lối đi, nhà, ao"),
             ("cây_lâu_năm_khác", "cao su (hàng thẳng rất đều, tán đồng nhất, rụng lá khoảng tháng 1-2), điều, sầu riêng, cây ăn quả, mắc ca"),
             ("cây_hằng_năm_hoặc_đất_trống", "lúa, ngô, mì, nương rẫy, đất trống; màu thay đổi theo mùa, đất lộ"),
             ("xây_dựng_hoặc_mặt_nước", "nhà, đường, sân, hồ"),
             ("không_rõ", "chỉ khi thật sự không thể xác định")]
CHANGE_OPTS = [("không_thay_đổi", "không mất tán thật: biến động chỉ do mây, bóng, rụng lá mùa khô rồi xanh lại; NDVI quay về mức cũ"),
               ("khai_thác_rừng_trồng_rồi_trồng_lại", "rừng trồng bị chặt theo luân kỳ, sau đó trồng lại hoặc tái sinh cây lấy gỗ"),
               ("cháy_hoặc_thiên_tai_rồi_tái_sinh", "cháy, sạt lở, bão; đất không bị dọn để trồng cây nông nghiệp"),
               ("khai_thác_chọn", "các lỗ tán nhỏ rải rác, phần tán còn lại giữ nguyên"),
               ("chuyển_sang_nông_nghiệp", "rừng (tự nhiên hoặc trồng) bị dọn sạch, sau đó đất trống kéo dài trên 6 tháng, nương rẫy, hoặc hàng "
                "cây trồng mới (cà phê, sầu riêng, cao su, mì); NDVI không hồi phục về mức rừng"),
               ("chuyển_sang_hạ_tầng", "đường, hồ thủy điện, mỏ, công trình"),
               ("tái_canh_nông_nghiệp", "đất đã là nông nghiệp năm 2020, nay bị chặt để tái canh hoặc đổi cây trồng"),
               ("không_rõ", "không xác định được")]
CANOPY = ["dưới_10", "10_40", "40_70", "trên_70", "không_rõ"]
SCHEMA_S = {"type": "object", "additionalProperties": False,
            "properties": {"quan_sat": {"type": "string"},
                           "su_dung_dat_2020": {"enum": [c for c, _ in LAND_OPTS]},
                           "do_che_phu_2020": {"enum": CANOPY},
                           "cay_cao_tren_5m": {"enum": ["có", "không", "không_rõ"]},
                           "cay_che_bong": {"enum": ["có", "không", "không_rõ", "không_áp_dụng"]},
                           "thay_doi_sau_2020": {"enum": [c for c, _ in CHANGE_OPTS]},
                           "thoi_diem_thay_doi": {"type": "string"},
                           "do_tin_cay": {"enum": ["cao", "vừa", "thấp"]}},
            "required": ["quan_sat", "su_dung_dat_2020", "do_che_phu_2020", "cay_cao_tren_5m", "cay_che_bong", "thay_doi_sau_2020",
                         "thoi_diem_thay_doi", "do_tin_cay"]}


def silver_sys(item_id: str, k: int) -> str:
    """The human guide (section 5) as instructions; order k > 0 shuffles both option lists (fixed per item and k)."""
    land, chg = list(LAND_OPTS), list(CHANGE_OPTS)
    if k:
        rng = random.Random(f"{item_id}|{k}")
        rng.shuffle(land)
        rng.shuffle(chg)
    opt = lambda xs: "\n".join(f"- {c}: {d}" for c, d in xs)  # noqa: E731
    return ("Bạn là người giải đoán ảnh vệ tinh, gán nhãn cho một đánh giá theo quy định chống phá rừng của EU (EUDR) ở Tây Nguyên. "
            "Bạn chỉ có trang ảnh tham chiếu: hai hàng ảnh Sentinel-2 theo các mốc ghi ở đầu mỗi ô (hàng trên màu thật; hàng dưới màu giả "
            "hồng ngoại, trong đó THỰC VẬT HIỆN MÀU ĐỎ, đất trống và công trình màu xám, nâu hoặc xanh nhạt), và chuỗi NDVI theo thời gian ở "
            "dưới cùng (vạch đỏ: mốc cắt 31/12/2020 và, với cảnh báo, ngày cảnh báo đầu tiên). Viền hoặc khung vàng là ranh đối tượng; ô "
            "'no image' là không có ảnh không mây.\n\n"
            "Trả lời hai câu hỏi về thực tế mặt đất, không kết luận về luật.\n"
            "1) Sử dụng đất tại 31/12/2020: xét các ô 'dry 2019', 'dry 2020', 'CUT-OFF 11/20-04/21'; với cảnh báo, xét đất trước khi mất tán. "
            "Nếu trong khung có nhiều loại đất, chọn loại chiếm trên 50% diện tích của đối tượng.\n" + opt(land) + "\n\n"
            "2) Thay đổi sau 31/12/2020:\n" + opt(chg) + "\n\n"
            "Trường hợp dễ nhầm: cà phê dưới cây che bóng KHÔNG phải rừng tự nhiên, kể cả khi tán cao và dày; cao su, điều, cây ăn quả là cây "
            "lâu năm khác, không phải rừng trồng lấy gỗ; keo chặt rồi trồng lại keo là khai thác rừng trồng, keo chặt rồi trồng cà phê là "
            "chuyển sang nông nghiệp; mảng cảnh báo nhỏ giữa khu rừng lớn vẫn là rừng nếu nó là một phần của khu rừng đó; rừng khộp và tre "
            "nứa rụng lá theo mùa, hãy so ảnh cùng mùa trước khi kết luận mất tán; cháy rồi dọn sạch và trồng cây đều hàng trong khoảng 12 "
            "tháng là chuyển sang nông nghiệp.\n\n"
            "Cũng ước lượng: độ che phủ cây năm 2020 (dưới_10, 10_40, 40_70, trên_70, không_rõ); có cây cao trên 5 m năm 2020 không; có cây "
            "che bóng không (chỉ khi là cà phê, còn lại ghi không_áp_dụng); tháng đầu tiên thấy thay đổi dạng YYYY-MM (ghi không_rõ nếu không "
            "xác định được tháng, để chuỗi rỗng nếu không thay đổi); độ tin cậy (cao: dấu hiệu rõ ở cả hai câu; vừa: còn điểm chưa chắc; "
            "thấp: thiếu ảnh, nhiều mây hoặc phải đoán).\n"
            "Viết phần quan_sat trước (mô tả ngắn từng giai đoạn trên ảnh), rồi mới chọn đáp án.")


def silver_payload(ids: list[str]) -> dict:
    g = items().set_index("item_id")
    out, images = [], {}
    for i in ids:
        r = g.loc[i]
        images[i] = "data:image/png;base64," + __import__("base64").b64encode((INTERP / f"{i}.png").read_bytes()).decode()
        what = (f"Đối tượng: mảng cảnh báo mất tán cây, phát hiện lần đầu {str(r.first_date)[:10]}, diện tích {float(r.area_ha):.2f} ha."
                if r.kind == "alert" else "Đối tượng: thửa 1 ha trong vùng trồng cà phê.")
        for k in range(N_ORDERS):
            out.append({"id": f"q35|{i}|{k}", "max_tokens": 900, "messages": [
                {"role": "system", "content": silver_sys(i, k)},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"ref:{i}"}},
                                             {"type": "text", "text": what + " Trả lời theo hướng dẫn."}]}]})
    return {"items": out, "images": images, "schema": SCHEMA_S, "max_tokens": 900, "llm_kw": {"tensor_parallel_size": 2}}


def labelled_ids() -> set[str]:
    return {json.loads(x)["id"].split("|")[1] for f in SILVER_RAW.glob("batch_*.jsonl") for x in open(f, encoding="utf-8") if x.strip()}


def do_label(hours: float, waves_per_batch: int = 2, poll: int = 120, smoke: int = 0):
    """Waves whose interpreter sheets are done go to the GPU server in batches; stops when the time budget is spent."""
    from ..extract.remote_lab import Lab, run_payload
    t_end = time.time() + hours * 3600
    SILVER_RAW.mkdir(parents=True, exist_ok=True)
    g = items()
    while time.time() < t_end:
        done_w = sorted(-1 if f.stem == "wave_dot0" else int(f.stem[5:]) for f in INTERP.glob("wave_*.done"))
        have = labelled_ids()
        todo = [i for i in g[g.wave.isin(done_w)].item_id if i not in have and (INTERP / f"{i}.png").exists()]
        if smoke:  # a first look: any items whose sheet exists, same settings as the real batches
            todo = [i for i in g.item_id if i not in have and (INTERP / f"{i}.png").exists()][:smoke]
        all_w = set(g.wave.unique())
        if not todo:
            if set(done_w) >= all_w:
                print("all waves labelled", flush=True)
                return
            time.sleep(poll)
            continue
        per_wave = len(g) / len(all_w)
        ids = todo[: int(waves_per_batch * per_wave * 1.05) + 1] if not smoke else todo
        lab = Lab()
        try:
            used = {int(a): int(b) for a, b in (ln.split(",") for ln in lab.retry(lambda: lab.ssh(
                "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"), "gpu check").strip().splitlines())}
            if used.get(0, 9e9) > 1500 or used.get(1, 9e9) > 1500:  # someone else is on a GPU: wait, never share it
                print(f"GPUs busy {used}; waiting", flush=True)
                time.sleep(poll * 2)
                continue
            n = len(list(SILVER_RAW.glob("batch_*.jsonl")))
            t0 = time.time()
            try:
                rows, summ = run_payload(lab, silver_payload(ids), f"silver_{n:03d}", LABELER, "0,1", timeout=5 * 3600, gpu_mem=0.85,
                                         max_model_len=8192)
            except (RuntimeError, TimeoutError) as e:  # the batch is wiped on the server; it is tried again next round
                print(f"batch {n} failed, retrying later: {str(e)[:300]}", flush=True)
                time.sleep(poll)
                continue
        finally:
            lab.close()
        with open(SILVER_RAW / f"batch_{n:03d}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"batch {n}: {len(ids)} items, {len(rows)} answers, {time.time() - t0:.0f} s | {summ}", flush=True)
        if smoke:
            return


# ------------------------------------------------------------------------------------------------ silver truth
def do_truth():
    from .human_labels import law
    rows = [json.loads(x) for f in sorted(SILVER_RAW.glob("batch_*.jsonl")) for x in open(f, encoding="utf-8") if x.strip()]
    by = {}
    for r in rows:
        _, i, k = r["id"].split("|")
        try:
            j = json.loads(r["raw"])
        except (json.JSONDecodeError, TypeError):
            j = None
        by.setdefault(i, {})[int(k)] = j
    meta = pd.read_csv(OUT / "items_s1.csv").set_index("item_id")
    out = []
    for i, ans in sorted(by.items()):
        js = [j for j in ans.values() if j]
        laws = [law(j["su_dung_dat_2020"], j["do_che_phu_2020"], j["cay_cao_tren_5m"], j["thay_doi_sau_2020"]) for j in js]
        kind = meta.kind.get(i) if i in meta.index else ("alert" if i.startswith("A") else "plot")

        def maj(vals, need=N_ORDERS // 2 + 1):  # a strict majority of the orders
            v = pd.Series([x for x in vals if x is not None], dtype=object)
            if v.empty:
                return None
            top = v.value_counts()
            return top.index[0] if top.iloc[0] >= need else None
        pair = pd.Series([f"{j['su_dung_dat_2020']}|{j['thay_doi_sau_2020']}" for j in js]).value_counts()
        agree = float(pair.iloc[0] / N_ORDERS) if len(pair) else 0.0
        chg = maj([z["change"] for z in laws])
        out.append({"item_id": i, "kind": kind, "y1": maj([z["y1"] for z in laws]), "y2": maj([z["y2"] for z in laws]),
                    "cause": chg if kind == "alert" and chg not in (None, "unclear") else None,
                    "conf": "high" if agree >= 0.8 else ("medium" if agree >= 0.6 else "low"),
                    "y1_strict": maj([z["y1_strict"] for z in laws]), "y2_strict": maj([z["y2_strict"] for z in laws]),
                    "land": maj([j["su_dung_dat_2020"] for j in js]), "change": maj([j["thay_doi_sau_2020"] for j in js]),
                    "agree": round(agree, 2), "n_answers": len(js), "status": "silver_q35",
                    "split": meta.split.get(i, "dot0") if i in meta.index else "dot0",
                    "wave": meta.wave.get(i) if i in meta.index else None, "stratum": meta.stratum.get(i) if i in meta.index else None})
    t = pd.DataFrame(out).astype({c: "Int64" for c in ("y1", "y2", "y1_strict", "y2_strict")})
    f = WORK / "silver_q35_truth.csv"
    t.to_csv(f, index=False)
    s = t[t.split != "dot0"]
    print(f"silver truth: {len(t)} items ({len(s)} in s1: {s.split.value_counts().to_dict()}) -> {f}")
    print("deforestation yes/no/unknown:", s.y2.value_counts(dropna=False).to_dict(), "| agreement:", s.conf.value_counts().to_dict())
    return t


# ------------------------------------------------------------------------------------------------ evidence for the system
def do_evidence(waves):
    from . import evidence as E
    ee = ee_init()
    A = gpd.read_parquet(OUT / "sample_alerts.parquet")
    B = gpd.read_parquet(OUT / "sample_plots.parquet")
    f = OUT / "evidence.parquet"
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["item_id"])
    rows = [have]
    for g, sig in ((A, E.alert_signals), (B, E.plot_signals)):
        g = g[g.wave.isin(waves) & ~g.item_id.isin(have.item_id)]
        for k in range(0, len(g), 150):  # chunks keep each Earth Engine request small
            part = g.iloc[k:k + 150]
            for tries in range(8):  # restricted mode: concurrency errors clear after a while
                try:
                    e = E.admin(part).merge(E.gee_common(ee, part), on="item_id", how="left").merge(sig(ee, part), on="item_id", how="left")
                    break
                except Exception as ex:  # noqa: BLE001
                    print("evidence retry", tries, repr(ex)[:120], flush=True)
                    time.sleep(min(600, 60 * 2 ** tries))
            else:
                print(f"evidence chunk {part.item_id.iloc[0]}..{part.item_id.iloc[-1]} given up; run again", flush=True)
                continue
            rows.append(e)
            pd.concat(rows, ignore_index=True).to_parquet(f)
            print(f"evidence {part.kind.iloc[0]} {k + len(part)}/{len(g)}", flush=True)
    ev = pd.concat(rows, ignore_index=True)
    # triage.parquet: what legal_vlm.features() reads; no news, no earlier VLM pass, no triage category in s1
    meta = pd.concat([A.drop(columns="geometry")[["item_id", "kind", "stratum", "patch_id", "area_ha", "first_date", "conf_max"]],
                      B.drop(columns="geometry")[["item_id", "kind", "stratum", "area_ha"]]], ignore_index=True)
    xy = pd.read_csv(OUT / "items_s1.csv")[["item_id", "lon", "lat"]]
    tri = meta.merge(ev.drop(columns=[c for c in ("kind", "area_ha", "first_date", "lon", "lat") if c in ev]), on="item_id", how="inner")
    tri = tri.merge(xy, on="item_id", how="left").assign(category=np.nan)
    tri.to_parquet(OUT / "triage.parquet")
    print("triage rows:", len(tri))


# ------------------------------------------------------------------------------------------------ pre-registered endpoints
def do_report(model: str = "Qwen3.5-9B"):
    """P1, P2 and S1-S5 of PREREG_s1.md from the TEST evaluation (legal_vlm eval --truth q35 --split test) and the
    verdict rule fixed there: promising = P1 and P2, mixed = one of them, not promising = neither."""
    from .human_labels import cohen
    from .legal_vlm import OUT as LOUT, _mcnemar, silver_truth
    rep = json.loads((RUNS / "dossier" / "legal_vlm_report_s1_q35_test.json").read_text(encoding="utf-8"))
    pred = pd.read_csv(LOUT / f"predictions_{model}_q35_test.csv")
    t = pd.read_csv(WORK / "silver_q35_truth.csv")
    test = t[(t.split == "test") & t.y2.notna()].set_index("item_id")
    P = pred.pivot_table(index="item_id", columns="method", values="deforestation", aggfunc="first").fillna("uncertain")
    ok = P.reindex(test.index).fillna("uncertain").apply(lambda s: s.map({"yes": 1, "no": 0})).eq(test.y2, axis=0).astype(int)
    m = rep["methods"]
    d = lambda k: m[k]["deforestation"]  # noqa: E731
    p1 = rep["robustness"]["mcnemar"]["vlm_v2vi"]["base_vs_gfcV4_hansen"]
    fg_v2, fg_gfc = d("vlm_v2vi")["false_green"], d("gfcV4_hansen")["false_green"]
    P1 = p1["diff"] > 0 and p1["p"] < 0.05
    P2 = fg_v2 is not None and fg_gfc is not None and fg_v2 < fg_gfc
    Pt = P.reindex(test.index)
    res = {
        "n_test_scored": int(len(test)), "n_test_items": int((t.split == "test").sum()),
        "silver_resolved_share_test": round(float(t[t.split == "test"].y2.notna().mean()), 3),
        "silver_deforestation_share_test": round(float((test.y2 == 1).mean()), 3),
        "P1_accuracy_v2vi_vs_gfcV4_hansen": {"v2vi": d("vlm_v2vi")["acc_all_uncertain_wrong"], "gfcV4_hansen": d("gfcV4_hansen")["acc_all_uncertain_wrong"],
                                             **p1, "met": bool(P1)},
        "P2_false_green": {"v2vi": fg_v2, "gfcV4_hansen": fg_gfc, "ci95_v2vi": d("vlm_v2vi")["ci95"]["false_green"],
                           "ci95_gfcV4_hansen": d("gfcV4_hansen")["ci95"]["false_green"], "met": bool(P2)},
        "S1_prompt_ensemble": rep["robustness"]["prompt_ensemble"].get("main"),
        "S2_gfc_version_changes_test": {"v2vi_card_V4_to_V1": int((Pt.vlm_v2vi != Pt.vlm_v2vi_v1card).sum()) if "vlm_v2vi_v1card" in Pt else None,
                                        "overlay_V1_to_V4": int((Pt.gfcV1_hansen != Pt.gfcV4_hansen).sum()), "n": int(len(Pt))},
        "S3_answer_order": rep["robustness"]["mcnemar"]["vlm_v2vi"].get("vlm_v2vi_shuf"),
        "S4_lgbm_dev_vs_v2vi": {"lgbm_dev": d("lgbm_dev") if "lgbm_dev" in m else None,
                                "mcnemar_v2vi_vs_lgbm": _mcnemar(ok["vlm_v2vi"], ok["lgbm_dev"]) if "lgbm_dev" in ok else None},
        "by_kind_v2vi": rep["by_kind"].get("vlm_v2vi"), "by_kind_gfcV4_hansen": rep["by_kind"].get("gfcV4_hansen"),
    }
    old = silver_truth().set_index("item_id")  # S5: the new labeler against the dot0 silver labels (Claude agents)
    q = t[t.split == "dot0"].set_index("item_id").join(old[["y2", "cause"]], rsuffix="_old", how="inner")
    both = q[q.y2.notna() & q.y2_old.notna()]
    grp = {"none": "no_conversion", "agricultural_renovation": "no_conversion", "plantation_rotation": "rotation",
           "fire_without_conversion": "fire_natural", "natural_disturbance": "fire_natural", "selective_logging": "logging",
           "conversion_to_agriculture": "agriculture", "conversion_to_infrastructure": "infrastructure"}
    cz = q[q.cause.notna() & q.cause_old.notna()]
    res["S5_silver_vs_dot0_silver"] = {"n_both_known": int(len(both)), "deforestation": cohen(list(zip(both.y2.astype(int), both.y2_old.astype(int)))),
                                       "cause_group": cohen(list(zip(cz.cause.map(grp), cz.cause_old.map(grp)))),
                                       "q35_resolved_share_dot0": round(float(q.y2.notna().mean()), 3) if len(q) else None}
    res["verdict"] = "khả quan" if P1 and P2 else ("lẫn lộn" if P1 or P2 else "không khả quan")
    (RUNS / "dossier" / "s1_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


def waves_arg(s: str | None) -> list[int]:
    if not s:
        return list(range(0, 15))
    out = []
    for part in s.split(","):
        if part == "dot0":
            out.append(-1)
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "sheets", "label", "truth", "evidence", "report"])
    ap.add_argument("--waves", default=None, help="e.g. 0-5, 0,2,4 or dot0")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--which", choices=["interp", "standard"], default="interp", help="sheets: the labeler's or the system's")
    ap.add_argument("--smoke", type=int, default=0, help="label: only this many items, once")
    a = ap.parse_args()
    if a.cmd == "sample":
        do_sample()
    elif a.cmd == "sheets":
        do_sheets(waves_arg(a.waves), a.workers, a.which)
    elif a.cmd == "label":
        do_label(a.hours, smoke=a.smoke)
    elif a.cmd == "truth":
        do_truth()
    elif a.cmd == "report":
        do_report()
    else:
        do_evidence(waves_arg(a.waves))


if __name__ == "__main__":
    main()
