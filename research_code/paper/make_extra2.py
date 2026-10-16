"""Round-2 review analyses (N1, N3, N5) and label statistics for numbers.tex, on the current reference labels.

    DOSSIER_SET=s1 .venv/Scripts/python paper/forests_eudr/figures/make_extra2.py   -> runs/dossier/extra_analyses2.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DOSSIER_SET", "s1")
ROOT = Path(__file__).resolve().parents[1]  # research_code root
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from src.dossier.hybrid_ts_vlm import boot_diff, mcnemar, scores  # noqa: E402
from src.dossier.legal_vlm import ENSEMBLE, baselines, extra_baselines, vlm_predictions  # noqa: E402
from src.dossier.score_round import e2_baselines  # noqa: E402

meta = pd.read_csv(ROOT / "data/processed/dossier/s1/items_s1.csv").set_index("item_id")
t = pd.read_csv(ROOT / "data/interim/dossier/legal_vlm/human_truth.csv")
t = t[t.item_id.isin(meta.index)].set_index("item_id")
blind = set(json.loads((ROOT / "annotation/private/config.json").read_text(encoding="utf-8"))["blind"])
vp, _ = vlm_predictions("Qwen3.5-9B")
F = pd.read_csv(ROOT / "runs/dossier/hybrid_ts_vlm_report.features.csv").set_index("item_id")
P = pd.concat([vp, baselines(), extra_baselines(t.reset_index()), e2_baselines(F)], ignore_index=True)
ens = vp[vp.method.isin(["vlm_" + m for m in ENSEMBLE])]
W = P.pivot_table(index="item_id", columns="method", values="deforestation", aggfunc="first")
Wf = P.pivot_table(index="item_id", columns="method", values="forest_2020", aggfunc="first")
W["S1_ensemble"] = ens.groupby("item_id").deforestation.apply(lambda x: "yes" if (x == "yes").sum() >= 2 else "no")
Wf["S1_ensemble"] = ens.groupby("item_id").forest_2020.apply(lambda x: "yes" if (x == "yes").sum() >= 3 else "no")
yn = lambda s: s.map({"yes": 1, "no": 0})  # noqa: E731
rng = np.random.default_rng(42)
res = {}


def paired(y, a, b, groups):
    pa, pb = yn(a.reindex(y.index)), yn(b.reindex(y.index))
    ok = (pa.notna() & pb.notna()).values
    yy, aa, bb, gg = y.values[ok], pa.values[ok].astype(int), pb.values[ok].astype(int), groups[ok]
    return {"n": int(ok.sum()), "ba_a": round(scores(yy, aa)["bal_acc"], 3), "ba_b": round(scores(yy, bb)["bal_acc"], 3),
            "mcnemar": mcnemar(yy, aa, bb), "ba_diff_ci": boot_diff(yy, aa, bb, gg, rng, B=1000)}


# N1: forest 2020 on plots only, and alerts with the trivial "every alert is forest" rule
y1 = t[t.y1.notna()].y1.astype(int)
yp = y1[t.loc[y1.index, "kind"] == "plot"]
gp = meta.loc[yp.index, "block"].values
res["n1_plots"] = {f"{a}_vs_{b}": paired(yp, Wf[a], Wf[b], gp) for a, b in
                   (("S1_ensemble", "rules_legal"), ("S1_ensemble", "gfcV4_hansen"), ("vlm_v2vi", "rules_legal"), ("vlm_v2vi", "gfcV4_hansen"))}
ya = y1[t.loc[y1.index, "kind"] == "alert"]
res["n1_alerts"] = {"n": int(len(ya)), "forest_share": round(float(ya.mean()), 3),
                    "acc_every_alert_forest": round(float(ya.mean()), 3),
                    **{m: {k: round(v, 3) for k, v in scores(ya.values, yn(Wf[m].reindex(ya.index)).fillna(1).astype(int).values).items() if k in ("acc", "bal_acc", "sensitivity", "specificity")}
                       for m in ("S1_ensemble", "vlm_v2vi", "rules_legal", "gfcV4_hansen")}}
# N3: paired tests on deforestation, all labels and TEST
y2 = t[t.y2.notna()].y2.astype(int)
g2 = meta.loc[y2.index, "block"].values
res["n3"] = {}
for s, idx in (("all", y2.index), ("test", y2.index[meta.loc[y2.index, "split"] == "test"])):
    yy, gg = y2.loc[idx], meta.loc[idx, "block"].values
    res["n3"][s] = {f"{a}_vs_{b}": paired(yy, W[a], W[b], gg) for a, b in
                    (("ndvi_rule", "gfcV4_hansen"), ("ndvi_rule", "national_loss"), ("S1_ensemble", "gfcV4_hansen"),
                     ("S1_ensemble", "national_loss"), ("vlm_v2vi_imgonly", "gfcV4_hansen"), ("S1_ensemble", "all_alerts"), ("ndvi_rule", "all_alerts"))}
# N5: undetermined items counted as negatives (a call of deforestation there is wrong)
lab = t[t.status != "disputed"]
y3 = lab.y2.fillna(0).astype(int)  # undetermined -> 0
res["n5"] = {"n": int(len(y3)), "n_undetermined": int(lab.y2.isna().sum())}
for m in ("S1_ensemble", "vlm_v2vi", "vlm_v2vi_imgonly", "ndvi_rule", "gfcV4_hansen", "gfcV1_hansen", "national_loss", "gfw_drivers", "tmf_loss", "all_alerts", "whisp_fao"):
    p = yn(W[m].reindex(y3.index))
    ok = p.notna()
    s3 = scores(y3[ok].values, p[ok].astype(int).values)
    res["n5"][m] = {"ba_undetermined_negative": round(s3["bal_acc"], 3), "coverage": round(float(ok.mean()), 3)}
# label statistics
q = json.loads((ROOT / "runs/dossier/human_labels_qc.json").read_text(encoding="utf-8"))
res["qc"] = {"agreement": q["agreement"], "anchoring": q.get("anchoring"), "flags": q.get("flags"), "confidence": q.get("confidence"),
             "median_seconds": q.get("median_seconds_per_label"), "n_labels": q.get("n_labels"), "n_annotators": q.get("n_annotators")}
# silver labels against human labels
sv = pd.read_csv(ROOT / "data/interim/dossier/s1/silver_q35_truth.csv").set_index("item_id")
res["silver"] = {}
for nm, idx in (("all", t.index), ("blind", t.index[t.index.isin(blind)])):
    for yv in ("y1", "y2"):
        j = t.loc[idx, [yv]].join(sv[[yv]], rsuffix="_s").dropna()
        res["silver"][f"{nm}_{yv}"] = {"n": int(len(j)), "agree": round(float((j[yv] == j[yv + "_s"]).mean()), 3),
                                       "kappa": round(float(cohen_kappa_score(j[yv], j[yv + "_s"])), 3)}
# coffee plots called forest (Figure 6b)
pl = y1[t.loc[y1.index, "kind"] == "plot"]
res["plots_forest_share"] = {"n": int(len(pl)), "human": round(float(pl.mean()), 3),
                             **{m: round(float(yn(Wf[m].reindex(pl.index)).mean()), 3) for m in ("gfcV1_hansen", "gfcV4_hansen", "national_loss", "vlm_v2vi", "S1_ensemble")}}
(ROOT / "runs/dossier/extra_analyses2.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
