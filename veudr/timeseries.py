"""Time-series features of one item from the per-scene tables (the features of model E1 in the paper).

t0 is the first alert for alert patches and 1 January 2021 for plots. Sentinel-2 uses clear scenes only (frame
coverage >= 0.95 and clear share in the outline >= 0.9), one value per day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _win(d: pd.DataFrame, t0, a, b, col):
    return d.loc[(d.date >= t0 + pd.Timedelta(days=a)) & (d.date < t0 + pd.Timedelta(days=b)), col]


def _amp(s: pd.DataFrame) -> float:
    """Seasonal amplitude: spread (P90 - P10) of monthly medians, needs >= 6 distinct months."""
    if len(s) < 6:
        return np.nan
    mo = s.groupby(s.date.dt.month).ndvi.median()
    return float(mo.quantile(0.9) - mo.quantile(0.1)) if len(mo) >= 6 else np.nan


def item_features(s2: pd.DataFrame, landsat: pd.DataFrame, s1: pd.DataFrame, kind: str, first_alert_date=None) -> dict:
    """s2, landsat, s1: the item's rows of sentinel2_scenes, landsat_scenes and sentinel1_scenes."""
    t0 = pd.Timestamp(first_alert_date) if kind == "alert" and pd.notna(first_alert_date) else pd.Timestamp("2021-01-01")
    f = {}
    s = s2[(s2.box_coverage >= 0.95) & (s2.outline_clear_share >= 0.9) & s2.ndvi_outline.notna()]
    s = (s.assign(date=pd.to_datetime(s.datetime).dt.normalize()).groupby("date", as_index=False).ndvi_outline.mean()
         .rename(columns={"ndvi_outline": "ndvi"}))
    pre, post = _win(s, t0, -365, 0, "ndvi"), _win(s, t0, 0, 183, "ndvi")
    f["s2_pre"] = pre.median()
    f["s2_post_min"] = post.min() if len(post) else np.nan
    f["s2_drop"] = f["s2_pre"] - f["s2_post_min"]
    f["s2_rec12"] = _win(s, t0, 270, 455, "ndvi").median()
    f["s2_rec24"] = _win(s, t0, 640, 820, "ndvi").median()
    f["s2_latest"] = s[s.date >= s.date.max() - pd.Timedelta(days=365)].ndvi.median() if len(s) else np.nan
    den = f["s2_pre"] - f["s2_post_min"]
    f["s2_recovery"] = (f["s2_latest"] - f["s2_post_min"]) / den if pd.notna(den) and den > 0.05 else np.nan
    f["s2_amp_pre"] = _amp(s[(s.date < t0) & (s.date >= t0 - pd.Timedelta(days=3 * 365))])
    f["s2_amp_post"] = _amp(s[s.date >= t0 + pd.Timedelta(days=365)])
    f["s2_n_post"] = int((s.date >= t0).sum())
    ls = landsat.assign(date=pd.to_datetime(landsat.datetime))
    ls = ls[(ls.outline_clear_share >= 0.9) & ls.ndvi_outline.notna() & (ls.date < "2016-01-01")]
    f["ls_mean_0015"], f["ls_sd_0015"] = ls.ndvi_outline.mean(), ls.ndvi_outline.std()
    f["ls_min_0015"] = ls.ndvi_outline.quantile(0.05) if len(ls) else np.nan
    r = s1.assign(date=pd.to_datetime(s1.datetime)).dropna(subset=["vh_db"])

    def vh(a, b):
        w = r[(r.date >= t0 + pd.Timedelta(days=a)) & (r.date < t0 + pd.Timedelta(days=b))]
        return w.groupby("relative_orbit").vh_db.median().median() if len(w) else np.nan
    f["s1_vh_pre"], f["s1_vh_post"] = vh(-365, 0), vh(183, 548)
    last = r[r.date >= r.date.max() - pd.Timedelta(days=365)] if len(r) else r
    f["s1_vh_latest"] = last.groupby("relative_orbit").vh_db.median().median() if len(last) else np.nan
    f["s1_vh_drop"] = f["s1_vh_pre"] - f["s1_vh_post"]
    f["s1_vh_rec"] = f["s1_vh_latest"] - f["s1_vh_post"]
    f["is_alert"] = int(kind == "alert")
    f["alert_year"] = t0.year if kind == "alert" else np.nan
    return f


def features(ds, item_ids=None) -> pd.DataFrame:
    """Features for every item with scene series (or the given ones) of a veudr.data.Dataset."""
    it = ds.items[ds.items.has_scene_series].set_index("item_id")
    ids = list(it.index if item_ids is None else item_ids)
    t = {k: ds.scene_table(k).groupby("item_id") for k in ("sentinel2", "landsat", "sentinel1")}
    rows = {i: item_features(t["sentinel2"].get_group(i), t["landsat"].get_group(i), t["sentinel1"].get_group(i),
                             it.at[i, "kind"], it.at[i, "first_alert_date"]) for i in ids}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("item_id")
