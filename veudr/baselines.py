"""Map-overlay and rule baselines of the paper, computed from the released map features and time-series features.

Each function returns a DataFrame with item_id, method, forest_2020 and deforestation ("yes" / "no" / "uncertain").
Thresholds are shares of the item's area. Baselines that need data not redistributed in EUDR-CH (the official forest
map of Vietnam, FAO Whisp run on the plot geometries) are not reproduced here; run Whisp with openforis-whisp.
"""
from __future__ import annotations

import pandas as pd


def _z(x) -> float:
    return 0.0 if pd.isna(x) else float(x)


def _yn(b) -> str:
    return "yes" if b else "no"


def map_overlays(mf: pd.DataFrame) -> pd.DataFrame:
    """mf: map_features.parquet. GFC2020 V1 / V4 + Hansen loss, JRC TMF, Dynamic World, drivers map, every alert."""
    out = []
    for r in mf.itertuples():
        alert = r.item_id.startswith("A")

        def rec(m, forest, defor):
            out.append({"item_id": r.item_id, "method": m, "forest_2020": forest, "deforestation": defor})
        for v in ("V1", "V4"):  # "GFC2020 + Hansen" overlay
            fo = _z(getattr(r, f"gfc{v}")) >= 0.5
            rec(f"gfc{v}_hansen", _yn(fo), _yn(fo and _z(r.hl_post2020) >= 0.2))
        tf = _z(r.tmf20_undisturbed) + _z(r.tmf20_degraded) + _z(r.tmf20_regrowth) >= 0.5
        rec("tmf_loss", _yn(tf), _yn(tf and _z(r.tmf25_deforested) - _z(r.tmf20_deforested) >= 0.2))
        dwf = (_z(r.dw_pre_trees) if alert else _z(r.dw_trees_2019)) >= 0.5
        conv = (_z(r.dw_post_crops) + _z(r.dw_post_bare)) if alert else (_z(r.dw_crops_2025) + _z(r.dw_bare_2025))
        rec("dw_conversion", _yn(dwf), _yn(dwf and conv >= 0.3))
        gf = _z(r.gfcV4) >= 0.5
        d = r.driver_1km
        rec("gfw_drivers", _yn(gf), _yn(gf and _z(r.hl_post2020) >= 0.2 and pd.notna(d) and int(d) in (1, 3)))
        rec("all_alerts", None, _yn(alert or _z(r.hl_post2020) >= 0.2))
    return pd.DataFrame(out)


def ndvi_rule(ts: pd.DataFrame) -> pd.DataFrame:
    """ts: veudr.timeseries.features(). Forest if Sentinel-2 NDVI before t0 >= 0.6; deforestation if NDVI dropped by
    >= 0.2 and 24 months later (else the last 12 months) is still >= 0.15 below the level before."""
    out = []
    for i, r in ts.iterrows():
        rec24 = r.s2_rec24 if pd.notna(r.s2_rec24) else r.s2_latest
        nf = pd.notna(r.s2_pre) and r.s2_pre >= 0.6
        dd = pd.notna(r.s2_drop) and r.s2_drop >= 0.2 and pd.notna(rec24) and (r.s2_pre - rec24) >= 0.15
        out.append({"item_id": i, "method": "ndvi_rule", "forest_2020": _yn(nf) if pd.notna(r.s2_pre) else "uncertain",
                    "deforestation": _yn(dd)})
    return pd.DataFrame(out)
