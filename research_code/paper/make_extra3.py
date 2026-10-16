"""Round-2 review analyses B1, B3 and B5 on the current reference labels.

    DOSSIER_SET=s1 .venv/Scripts/python paper/forests_eudr/figures/make_extra3.py
    -> runs/dossier/extra_analyses3.json, tables/tab_weighted.tex, supplementary/tab_byclass.tex

B1  design-weighted balanced accuracy on the alert population (inverse inclusion probabilities per stratum,
    stratified bootstrap). Plots are model-selected points without inclusion probabilities and are not weighted.
B3  supervised classifiers trained on DEV labels only and scored on TEST: time-series features alone, and time
    series + map features (no VLM answers).
B5  forest-2020 calls by land-use class of the reference label.
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
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.dossier.hybrid_ts_vlm import boot_diff, mcnemar, scores  # noqa: E402
from src.dossier.legal_vlm import ENSEMBLE, baselines, extra_baselines, vlm_predictions  # noqa: E402
from src.dossier.score_round import e2_baselines  # noqa: E402

TAB = ROOT / "paper/forests_eudr/tables"
SUP = ROOT / "paper/forests_eudr/supplementary"
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
rng = np.random.default_rng(7)
res = {}
NAMES = {"S1_ensemble": "VLM, five-prompt ensemble (S1)", "vlm_v2vi": "VLM, single prompt", "ndvi_rule": "NDVI trajectory rule",
         "rules_legal": "Legal rules on the card", "gfcV4_hansen": "GFC2020 V4 + Hansen loss", "gfcV1_hansen": "GFC2020 V1 + Hansen loss",
         "national_loss": "National 2020 map + loss", "whisp_fao": "Whisp (FAO), official", "gfw_drivers": "GFC2020 V4 + drivers"}


# ---- B1: design-weighted balanced accuracy, alert population
def wba(y, p, w):
    pos, neg = y == 1, y == 0
    sens = (w * (pos & (p == 1))).sum() / (w * pos).sum()
    spec = (w * (neg & (p == 0))).sum() / (w * neg).sum()
    return (sens + spec) / 2


pop = meta[meta.kind == "alert"].groupby("stratum")["pop"].first()
res["weighted"] = {}
for yv, M in (("y1", Wf), ("y2", W)):
    y = t[(t.kind == "alert") & t[yv].notna()][yv].astype(int)
    st = meta.loc[y.index, "stratum"]
    w = st.map(pop / st.value_counts())
    out = {"n": int(len(y))}
    for m in NAMES:
        if m not in M:
            continue
        p = yn(M[m].reindex(y.index))
        ok = p.notna()
        yy, pp, ww, ss = y[ok].values, p[ok].astype(int).values, w[ok].values, st[ok].values
        est = wba(yy, pp, ww)
        un = scores(yy, pp)["bal_acc"]
        boots = []
        idx_by = {s: np.where(ss == s)[0] for s in np.unique(ss)}
        for _ in range(500):
            ix = np.concatenate([rng.choice(v, len(v)) for v in idx_by.values()])
            try:
                boots.append(wba(yy[ix], pp[ix], ww[ix]))
            except ZeroDivisionError:
                pass
        out[m] = {"weighted": round(float(est), 3), "ci": [round(float(np.nanpercentile(boots, 2.5)), 3), round(float(np.nanpercentile(boots, 97.5)), 3)],
                  "unweighted": round(float(un), 3)}
    res["weighted"][yv] = out

# ---- B3: supervised time-series classifiers, DEV -> TEST
ts_cols = [c for c in F.columns if c.startswith(("s2_", "ls_", "s1_"))] + ["is_alert", "alert_year"]
map_cols = [c for c in F.columns if c.startswith("map_")]
y2 = t[t.y2.notna()].y2.astype(int)
split = meta.loc[y2.index, "split"]
test = y2[split == "test"]  # scored on the current reference (person labels)
# trained on the DEV items of the full 1000-item reference: person DEV labels are too few (38 items, 5 deforested)
ta = pd.read_csv(ROOT / "data/interim/dossier/legal_vlm/human_truth_all1000.csv").set_index("item_id")
ta = ta[ta.index.isin(meta.index) & ta.y2.notna()]
dev = ta.y2.astype(int)[meta.loc[ta.index, "split"] == "dev"]
res["supervised"] = {"n_dev": int(len(dev)), "pos_dev": int(dev.sum()), "n_test": int(len(test)), "pos_test": int(test.sum())}
bt = test[test.index.isin(blind)]
for name, cols in (("ts_only", ts_cols), ("ts_maps", ts_cols + map_cols)):
    for kind, model in (("logistic", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                   LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))),
                        ("lightgbm", make_pipeline(SimpleImputer(strategy="median"),
                                                   LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                                                                  class_weight="balanced", random_state=0, verbose=-1)))):
        X = F[cols].apply(pd.to_numeric, errors="coerce")
        model.fit(X.loc[dev.index].values, dev.values)
        pr = pd.Series(model.predict(X.loc[test.index].values), index=test.index)
        s_all = scores(test.values, pr.values)
        s_bt = scores(bt.values, pr.loc[bt.index].values)
        res["supervised"][f"{name}_{kind}"] = {"test": round(s_all["bal_acc"], 3), "test_sens": round(s_all["sensitivity"], 3),
                                                "test_spec": round(s_all["specificity"], 3), "blind_test": round(s_bt["bal_acc"], 3)}
        if (name, kind) == ("ts_only", "lightgbm"):  # paired tests on TEST against the main training-free methods
            g = meta.loc[test.index, "block"].values
            for ref in ("S1_ensemble", "ndvi_rule", "all_alerts", "gfcV4_hansen"):
                q = yn(W[ref].reindex(test.index))
                ok = q.notna().values
                res["supervised"][f"ts_lightgbm_vs_{ref}"] = {
                    "mcnemar": mcnemar(test.values[ok], pr.values[ok].astype(int), q.values[ok].astype(int)),
                    "ba_diff_ci": boot_diff(test.values[ok], pr.values[ok].astype(int), q.values[ok].astype(int), g[ok], rng, B=1000)}
for m in ("S1_ensemble", "ndvi_rule", "all_alerts", "gfcV4_hansen"):
    p = yn(W[m].reindex(test.index))
    ok = p.notna()
    res["supervised"][f"ref_{m}"] = {"test": round(scores(test[ok].values, p[ok].astype(int).values)["bal_acc"], 3)}

# ---- B5: forest-2020 calls by reference land-use class
lab = pd.read_csv(ROOT / "release/kaggle/eudr-ch/labels/labels.csv")
lab = lab[lab.label_source.str.startswith("human")]  # land-use class from the person labels (and person adjudications)
adj = lab[lab.role == "adjudication"].groupby("item_id").land_use_2020.first()
maj = lab[lab.role != "adjudication"].groupby("item_id").land_use_2020.agg(lambda s: s.mode().iloc[0])
land = adj.combine_first(maj)
shade = lab.groupby("item_id").shade_trees.agg(lambda s: s.dropna().mode().iloc[0] if s.dropna().size else "")
cls = land.copy()
cls[(land == "coffee_or_agroforestry") & (shade == "yes")] = "coffee_shaded"
cls[(land == "coffee_or_agroforestry") & (shade != "yes")] = "coffee_unshaded"
y1 = t[t.y1.notna()].y1.astype(int)
cls = cls.reindex(y1.index)
order = ["natural_forest", "wood_plantation", "coffee_shaded", "coffee_unshaded", "other_perennial_crop", "annual_crop_or_bare", "built_up_or_water", "unclear"]
res["by_class"] = {}
for c in order:
    ix = cls.index[cls == c]
    if not len(ix):
        continue
    row = {"n": int(len(ix)), "reference_forest": round(float(y1.loc[ix].mean()), 2)}
    for m in ("S1_ensemble", "vlm_v2vi", "rules_legal", "gfcV4_hansen", "gfcV1_hansen", "national_loss"):
        p = yn(Wf[m].reindex(ix))
        row[m] = round(float(p.mean()), 2) if p.notna().any() else None
    res["by_class"][c] = row
(ROOT / "runs/dossier/extra_analyses3.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))

# ---- tables
L = [r"\begin{table}[H]",
     r"\caption{Design-weighted balanced accuracy on the population of alert patches (inverse inclusion probabilities of the "
     r"12 strata; 95\% stratified bootstrap intervals), with the unweighted value. Plots are model-selected points without "
     r"inclusion probabilities and are not included.\label{tab:weighted}}",
     r"\begin{adjustwidth}{-\extralength}{0cm}", r"\newcolumntype{C}{>{\centering\arraybackslash}X}",
     r"\begin{tabularx}{\fulllength}{lCCCC}", r"\toprule",
     r" & \multicolumn{2}{c}{\textbf{Forest 2020}} & \multicolumn{2}{c}{\textbf{Deforestation}}\\",
     r"\textbf{Method} & Weighted & Unweighted & Weighted & Unweighted\\", r"\midrule"]
for m, nm in NAMES.items():
    c = []
    for yv in ("y1", "y2"):
        v = res["weighted"][yv].get(m)
        c += ["--", "--"] if v is None else [f"{v['weighted']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]", f"{v['unweighted']:.2f}"]
    L.append(f"{nm} & " + " & ".join(c) + r"\\")
L += [r"\midrule", f"Alert patches & \\multicolumn{{2}}{{c}}{{{res['weighted']['y1']['n']}}} & \\multicolumn{{2}}{{c}}{{{res['weighted']['y2']['n']}}}\\\\",
      r"\bottomrule", r"\end{tabularx}", r"\end{adjustwidth}", r"\end{table}"]
(TAB / "tab_weighted.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
CN = {"natural_forest": "Natural forest", "wood_plantation": "Wood plantation", "coffee_shaded": "Coffee with shade trees",
      "coffee_unshaded": "Coffee without shade trees", "other_perennial_crop": "Other perennial crop", "annual_crop_or_bare": "Annual crop or bare land",
      "built_up_or_water": "Built-up or water", "unclear": "Unclear"}
L = [r"\noindent Share of items called forest on 31 December 2020, by land-use class of the reference label (all items with a "
     r"determined forest-2020 status). ``Ref.'' is the share that the reference labels count as forest; S1: five-prompt ensemble; "
     r"VLM: single prompt; Rules: legal rules on the card; V4, V1: GFC2020 overlays; Nat.: national 2020 map.\par\smallskip",
     r"{\small", r"\begin{tabularx}{\textwidth}{Xrrrrrrrr}", r"\toprule",
     r"\textbf{Class} & $n$ & Ref. & S1 & VLM & Rules & V4 & V1 & Nat.\\", r"\midrule"]
for c, r in res["by_class"].items():
    f = lambda v: "--" if v is None else f"{v:.2f}"  # noqa: E731
    L.append(f"{CN[c]} & {r['n']} & {f(r['reference_forest'])} & " + " & ".join(f(r[m]) for m in
             ("S1_ensemble", "vlm_v2vi", "rules_legal", "gfcV4_hansen", "gfcV1_hansen", "national_loss")) + r"\\")
L += [r"\bottomrule", r"\end{tabularx}}"]
(SUP / "tab_byclass.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
print("tables written")
