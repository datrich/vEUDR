"""Scoring round on human labels (prereg runs/dossier/PREREG_s1.md section 22): frozen system, preregistered ensemble
S1, exploratory E1 (time series + VLM) and the E2 baselines, on four label sets, with paired tests and cause accuracy.

    python -m src.dossier.score_round [--truth data/interim/dossier/legal_vlm/human_truth.csv]

Writes runs/dossier/score_round_report.json and runs/dossier/score_round_report.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DOSSIER_SET", "s1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..common.paths import INTERIM, PROCESSED, RUNS  # noqa: E402
from .hybrid_ts_vlm import boot_diff, fit_predict, mcnemar, scores  # noqa: E402
from .legal_vlm import ENSEMBLE, GRP_V3, baselines, extra_baselines, vlm_predictions  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "annotation" / "private" / "config.json"
DRIVERS = {1: "agriculture", 3: "agriculture", 4: "logging", 2: "other", 5: "other", 6: "other", 7: "other"}  # Sims et al. 2025 v1.2
FEAT = RUNS / "dossier" / "hybrid_ts_vlm_report.features.csv"


def e2_baselines(ts: pd.DataFrame) -> pd.DataFrame:
    """The E2 baselines exactly as preregistered (section 22)."""
    f = pd.read_parquet(PROCESSED / "dossier" / "s1" / "legal_features.parquet").set_index("item_id")
    dr = pd.read_csv(INTERIM / "dossier" / "s1" / "gfw_drivers.csv").set_index("item_id").driver
    z = lambda x: 0.0 if pd.isna(x) else float(x)  # noqa: E731
    out = []

    def rec(m, i, forest, defor):
        out.append({"method": m, "item_id": i, "forest_2020": forest, "deforestation": defor,
                    "verdict": {"yes": "non_compliant", "no": "compliant"}.get(defor, "uncertain")})
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    for i, r in f.iterrows():
        alert = r.kind == "alert"
        rec("all_alerts", i, None, yn(alert or z(r.hl_post2020) >= 0.2))
        tf = z(r.tmf20_undisturbed) + z(r.tmf20_degraded) + z(r.tmf20_regrowth) >= 0.5
        rec("tmf_loss", i, yn(tf), yn(tf and z(r.tmf25_deforested) - z(r.tmf20_deforested) >= 0.2))
        dwf = (z(r.dw_pre_trees) if alert else z(r.dw_trees_2019)) >= 0.5
        conv = (z(r.dw_post_crops) + z(r.dw_post_bare)) if alert else (z(r.dw_crops_2025) + z(r.dw_bare_2025))
        rec("dw_conversion", i, yn(dwf), yn(dwf and conv >= 0.3))
        d = dr.get(i)
        gf = z(r.gfcV4) >= 0.5
        rec("gfw_drivers", i, yn(gf), yn(gf and z(r.hl_post2020) >= 0.2 and pd.notna(d) and int(d) in (1, 3)))
    for i, r in ts.iterrows():
        rec24 = r.s2_rec24 if pd.notna(r.s2_rec24) else r.s2_latest
        nf = pd.notna(r.s2_pre) and r.s2_pre >= 0.6
        dd = pd.notna(r.s2_drop) and r.s2_drop >= 0.2 and pd.notna(rec24) and (r.s2_pre - rec24) >= 0.15
        rec("ndvi_rule", i, yn(nf) if pd.notna(r.s2_pre) else "uncertain", yn(dd))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--truth", type=Path, default=INTERIM / "dossier" / "legal_vlm" / "human_truth.csv")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    meta = pd.read_csv(PROCESSED / "dossier" / "s1" / "items_s1.csv").set_index("item_id")
    blind = set(json.loads(CFG.read_text(encoding="utf-8"))["blind"])
    t = pd.read_csv(a.truth)
    t = t[t.item_id.isin(meta.index)].copy()
    t["split"] = meta.loc[t.item_id, "split"].values
    vp, _ = vlm_predictions("Qwen3.5-9B")
    F = pd.read_csv(FEAT).set_index("item_id")
    pred = pd.concat([vp, baselines(), extra_baselines(t, t[t.split == "dev"]), e2_baselines(F)], ignore_index=True)
    # S1: preregistered ensemble, >= 2 of the 5 wordings say deforestation
    ens = vp[vp.method.isin(["vlm_" + m for m in ENSEMBLE])]
    v = ens.groupby("item_id").deforestation.apply(lambda x: (x == "yes").sum())
    vf = ens.groupby("item_id").forest_2020.apply(lambda x: (x == "yes").sum())
    s1 = pd.DataFrame({"method": "S1_ensemble", "item_id": v.index, "deforestation": np.where(v >= 2, "yes", "no"),
                       "forest_2020": np.where(vf.reindex(v.index) >= 3, "yes", "no")})
    pred = pd.concat([pred, s1], ignore_index=True)
    # E1 M1: out-of-fold on the current labels, mean of 5 block shuffles (section 21 design)
    ts_cols = [c for c in F.columns if c.startswith(("s2_", "ls_", "s1_", "is_alert", "alert_year"))]
    vl_cols = [c for c in F.columns if c.startswith(("vlm_", "v2vi_"))]
    d = t[t.y2.notna() & t.item_id.isin(F.index)].set_index("item_id")
    y = d.y2.astype(int).values
    g = meta.loc[d.index, "block"].values
    pr = np.mean([fit_predict(F.loc[d.index, ts_cols + vl_cols], y, g, "logit", s) for s in range(5)], axis=0)
    pred = pd.concat([pred, pd.DataFrame({"method": "E1_M1", "item_id": d.index, "deforestation": np.where(pr >= 0.5, "yes", "no")})],
                     ignore_index=True)
    methods = ["E1_M1", "S1_ensemble", "vlm_v2vi", "vlm_v2vi_shuf", "gfcV4_hansen", "gfcV1_hansen", "national_loss", "whisp_like",
               "whisp_fao", "rules_legal", "lgbm_dev", "tmf_loss", "dw_conversion", "ndvi_rule", "gfw_drivers", "all_alerts"]
    methods = [m for m in methods if (pred.method == m).any()]
    wide = pred.pivot_table(index="item_id", columns="method", values="deforestation", aggfunc="first")
    widef = pred.pivot_table(index="item_id", columns="method", values="forest_2020", aggfunc="first")
    rng = np.random.default_rng(42)
    rep = {"truth_file": str(a.truth), "n_truth": int(len(t)), "sets": {}}
    sets = {"all": t, "test": t[t.split == "test"], "blind": t[t.item_id.isin(blind)], "blind_test": t[t.item_id.isin(blind) & (t.split == "test")]}
    for sname, tt in sets.items():
        res = {}
        for target, W in (("y2", wide), ("y1", widef)):
            x = tt[tt[target].notna()].set_index("item_id")
            yy = x[target].astype(int)
            block = meta.loc[x.index, "block"].values
            rows = {"n": int(len(yy)), "n_pos": int(yy.sum())}
            triv = pd.Series(0 if target == "y2" else 1, index=x.index)
            rows["trivial"] = scores(yy.values, triv.values)
            for m in methods:
                if m not in W:
                    continue
                pm = W[m].reindex(x.index).map({"yes": 1, "no": 0})
                ok = pm.notna()
                if ok.sum() < 5:
                    continue
                rows[m] = scores(yy[ok].values, pm[ok].astype(int).values)
                rows[m]["coverage"] = round(float(ok.mean()), 3)
            for m in ("E1_M1", "S1_ensemble", "vlm_v2vi"):
                if m not in W or target == "y1" and m == "E1_M1":
                    continue
                for ref in ("gfcV4_hansen", "vlm_v2vi"):
                    if ref == m or ref not in W:
                        continue
                    pa, pb = W[m].reindex(x.index).map({"yes": 1, "no": 0}), W[ref].reindex(x.index).map({"yes": 1, "no": 0})
                    ok = (pa.notna() & pb.notna()).values
                    if ok.sum() < 5:
                        continue
                    rows[f"{m}_vs_{ref}"] = {"mcnemar": mcnemar(yy.values[ok], pa.values[ok].astype(int), pb.values[ok].astype(int)),
                                             "bal_acc_diff_ci95": boot_diff(yy.values[ok], pa.values[ok].astype(int), pb.values[ok].astype(int), block[ok], rng, B=500)}
            res[target] = rows
        # cause on alerts with a change: VLM (v2vi, folded to the form-v3 groups) and GFW drivers vs the human change
        c = tt[tt.cause.notna() & tt.item_id.str.startswith("A")].set_index("item_id")
        gt = c.cause.map(GRP_V3)
        cause = {"n": int(len(c))}
        for m in ("vlm_v2vi", "vlm_v2vi_shuf"):
            pc = vp[vp.method == m].set_index("item_id").change.reindex(c.index).map(GRP_V3)
            ok = pc.notna()
            cause[m] = round(float((pc[ok] == gt[ok]).mean()), 3) if ok.any() else None
        dr = pd.read_csv(INTERIM / "dossier" / "s1" / "gfw_drivers.csv").set_index("item_id").driver.reindex(c.index).map(DRIVERS)
        grp4 = gt.replace({"rotation": "logging", "cleared": "other", "none": "other"})  # drivers have no "none"/"cleared": compare on 4 groups
        ok = dr.notna() & gt.isin(["agriculture", "rotation", "other"])
        cause["gfw_drivers_agri_vs_not"] = round(float(((dr[ok] == "agriculture") == (gt[ok] == "agriculture")).mean()), 3) if ok.any() else None
        for m in ("vlm_v2vi", "vlm_v2vi_shuf"):
            pc = vp[vp.method == m].set_index("item_id").change.reindex(c.index).map(GRP_V3)
            ok2 = ok & pc.notna()
            cause[f"{m}_agri_vs_not"] = round(float(((pc[ok2] == "agriculture") == (gt[ok2] == "agriculture")).mean()), 3) if ok2.any() else None
        cause["n_agri_vs_not"] = int(ok.sum())
        res["cause_alerts"] = cause
        rep["sets"][sname] = res
    out = RUNS / "dossier" / "score_round_report.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    lines = []
    for sname, res in rep["sets"].items():
        for target in ("y2", "y1"):
            r = res[target]
            lines.append(f"\n### {sname} · {target} (n={r['n']}, positives={r['n_pos']})\n")
            lines.append("| method | bal_acc | acc | sens | spec | false_green | coverage |\n|---|---|---|---|---|---|---|")
            for m in ["trivial"] + methods:
                v = r.get(m)
                if isinstance(v, dict) and "bal_acc" in v:
                    lines.append(f"| {m} | {v['bal_acc']:.3f} | {v['acc']:.3f} | {v['sensitivity']:.3f} | {v['specificity']:.3f} | {v['false_green']:.3f} | {v.get('coverage', 1):.2f} |")
            for k, v in r.items():
                if "_vs_" in k:
                    lines.append(f"\n{k}: McNemar {v['mcnemar']}, Δbal_acc CI95 {v['bal_acc_diff_ci95']}")
        lines.append(f"\ncause ({sname}): {res['cause_alerts']}")
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("report ->", out)


if __name__ == "__main__":
    main()
