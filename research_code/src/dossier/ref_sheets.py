"""Reference sheets for the (silver / human) interpreter: denser imagery than the VLM-under-test gets.

    python -m src.dossier.ref_sheets

Sentinel-2 L1C (COPERNICUS/S2_HARMONIZED, 2016+) masked with s2cloudless probability < 40, so the record starts
two years before the SR product used elsewhere. Each sheet = 2 image rows (true colour, false colour NIR) plus a
per-scene NDVI time series of the polygon (mean over clear pixels, scenes with >= 50 % clear pixels), which shows
short clearings that seasonal composites miss.
  alerts (6 columns, ~800 m box): -12..-7 mo | -4..-1 mo | alert 0..+2 mo | +3..+6 mo | +10..+14 mo | +16..+20 mo
  plots  (8 columns, ~600 m box): dry 2017 | dry 2018 | dry 2019 | dry 2020 | CUT-OFF 11/2020-04/2021 | dry 2022
                                  | dry 2024 | dry 2026
The sheets carry NO map layers, alert confidence, news or model output (interpreters stay blind).
Output: data/interim/dossier/ref_sheets/<item_id>.png, data/processed/dossier/ndvi_series.parquet
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import geopandas as gpd
import httpx
import matplotlib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ..alerts.gee_common import ee_init
from ..common.paths import INTERIM
from .sample import OUT, SET

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REF = INTERIM / "dossier" / SET / "ref_sheets" if SET else INTERIM / "dossier" / "ref_sheets"
ROWS = [(["B4", "B3", "B2"], {"min": 300, "max": 2800}, "true colour"),
        (["B8", "B4", "B3"], {"min": 300, "max": 4500}, "false colour NIR")]
ALERT_WINS = [("-12..-7 mo", -365, -200), ("-4..-1 mo", -120, -20), ("ALERT 0..+2 mo", 0, 60),
              ("+3..+6 mo", 90, 180), ("+10..+14 mo", 300, 420), ("+16..+20 mo", 480, 600)]
PLOT_WINS = [("dry 2017", "2016-12-01", "2017-04-30"), ("dry 2018", "2018-01-01", "2018-04-30"),
             ("dry 2019", "2019-01-01", "2019-04-30"), ("dry 2020", "2020-01-01", "2020-04-30"),
             ("CUT-OFF 11/20-04/21", "2020-11-01", "2021-04-30"), ("dry 2022", "2022-01-01", "2022-04-30"),
             ("dry 2024", "2024-01-01", "2024-04-30"), ("dry 2026", "2026-01-01", "2026-04-30")]
TODAY = pd.Timestamp("2026-10-01")


def s2(ee, region, start, end):
    l1c = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(region).filterDate(start, end)
    cp = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(region).filterDate(start, end)
    j = ee.Join.saveFirst("cp").apply(l1c, cp, ee.Filter.equals(leftField="system:index", rightField="system:index"))
    return ee.ImageCollection(j).map(lambda i: i.updateMask(ee.Image(i.get("cp")).select("probability").lt(40)))


def chip(ee, geom, start, end, bands, vis, half) -> Image.Image | None:
    if pd.Timestamp(start) > TODAY:
        return None
    region = ee.Geometry(geom.centroid.buffer(half).envelope.__geo_interface__)
    col = s2(ee, region, str(start), str(min(pd.Timestamp(end), TODAY).date()))
    if col.size().getInfo() == 0:
        return None
    img = col.median().select(bands).visualize(**vis)
    outline = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(ee.Geometry(geom.__geo_interface__))]), 1, 2)
    url = img.blend(outline.visualize(palette=["ffff00"])).getThumbURL({"region": region, "dimensions": 192, "format": "png"})
    r = httpx.get(url, timeout=180)
    return Image.open(io.BytesIO(r.content)).convert("RGB") if r.status_code == 200 else None


def ndvi_series(ee, geom) -> pd.DataFrame:
    g = ee.Geometry(geom.__geo_interface__)
    col = s2(ee, g, "2016-01-01", str(TODAY.date()))

    def f(i):
        nd = i.normalizedDifference(["B8", "B4"]).rename("nd")
        s = nd.reduceRegion(ee.Reducer.mean().combine(ee.Reducer.count(), "", True), g, 10)
        tot = ee.Image(1).rename("one").reduceRegion(ee.Reducer.count(), g, 10).get("one")
        return ee.Feature(None, {"t": i.date().millis(), "ndvi": s.get("nd_mean"), "n": s.get("nd_count"), "tot": tot})

    fc = col.map(f).filter(ee.Filter.notNull(["ndvi"]))
    rows = [x["properties"] for x in fc.getInfo()["features"]]
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d.t, unit="ms")
    d = d[d.n >= 0.5 * d.tot].groupby(d.date.dt.date).ndvi.mean().reset_index()  # one value per day (tile overlap)
    d["date"] = pd.to_datetime(d.date)
    return d


def chart(d: pd.DataFrame, marks: list[tuple[str, pd.Timestamp]], width_px: int) -> Image.Image:
    fig, ax = plt.subplots(figsize=(width_px / 100, 2.4), dpi=100)
    if not d.empty:
        ax.plot(d.date, d.ndvi, ".", ms=3, color="#2e7d32")
        ax.plot(d.date, d.ndvi.rolling(5, center=True, min_periods=1).median(), "-", lw=1, color="#1b5e20")
    for lab, t in marks:
        ax.axvline(t, color="#c62828", lw=1, ls="--")
        ax.text(t, 0.97, " " + lab, color="#c62828", fontsize=7, va="top", transform=ax.get_xaxis_transform())
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlim(pd.Timestamp("2016-01-01"), TODAY)
    ax.set_ylabel("NDVI (clear scenes)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return Image.open(buf).convert("RGB").resize((width_px, 240))


def make(ee, iid, kind, geom, t0) -> tuple[str, pd.DataFrame | None]:
    f = REF / f"{iid}.png"
    wins = ALERT_WINS if kind == "alert" else PLOT_WINS
    half = 0.0036 if kind == "alert" else 0.0027
    W, H, top = 192, 192, 18
    width = len(wins) * W
    ts = ndvi_series(ee, geom)
    if f.exists():
        return iid, ts
    sheet = Image.new("RGB", (width, 2 * (H + top) + 240), "white")
    d = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    cells = []
    for row, (bands, vis, name) in enumerate(ROWS):
        for col, w in enumerate(wins):
            if kind == "alert":
                lab, a, b = w
                start, end = (t0 + timedelta(days=a)).date(), (t0 + timedelta(days=b)).date()
            else:
                lab, start, end = w
            cells.append((row, col, lab, name, start, end, bands, vis))
    with ThreadPoolExecutor(6) as ex:  # the chips of one sheet in parallel (Earth Engine round trips dominate)
        ims = list(ex.map(lambda c: chip(ee, geom, c[4], c[5], c[6], c[7], half), cells))
    for (row, col, lab, name, *_), im in zip(cells, ims):
        x, y = col * W, row * (H + top)
        d.text((x + 3, y + 3), f"{lab} | {name[:5]}", fill="black", font=font)
        if im is not None:
            sheet.paste(im.resize((W, H)), (x, y + top))
        else:
            d.text((x + 40, y + top + 90), "no image", fill="red", font=font)
    marks = [("first alert", pd.Timestamp(t0))] if kind == "alert" else [("EUDR cut-off", pd.Timestamp("2020-12-31"))]
    sheet.paste(chart(ts, marks, width), (0, 2 * (H + top)))
    REF.mkdir(parents=True, exist_ok=True)
    sheet.save(f)
    return iid, ts


def main():
    ee = ee_init()
    A = gpd.read_parquet(OUT / "sample_alerts.parquet").to_crs(4326)
    B = gpd.read_parquet(OUT / "sample_plots.parquet").to_crs(4326)
    jobs = [(r.item_id, "alert", r.geometry, pd.Timestamp(r.first_date)) for r in A.itertuples()]
    jobs += [(r.item_id, "plot", r.geometry, None) for r in B.itertuples()]

    def one(j):
        for _ in range(3):
            try:
                return make(ee, *j)
            except Exception as e:  # noqa: BLE001 - transient EE / HTTP errors
                last = e
        print("FAILED", j[0], repr(last)[:150], flush=True)
        return j[0], None

    series = []
    with ThreadPoolExecutor(6) as ex:
        for k, (iid, ts) in enumerate(ex.map(one, jobs)):
            if ts is not None and not ts.empty:
                series.append(ts.assign(item_id=iid))
            if k % 20 == 0:
                print("ref sheet", k, iid, flush=True)
    pd.concat(series, ignore_index=True).to_parquet(OUT / "ndvi_series.parquet")
    print("done:", len(list(REF.glob("*.png"))), "sheets")


if __name__ == "__main__":
    main()
