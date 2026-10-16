"""Additional analyses requested in review (no new labels, no GPU): ablations and all VLM variants, three-class outcomes,
results by item kind, design-weighted shares, threshold sweep of the GFC2020 overlay, strict forest definition, the
primary test in full, uncertain-as-wrong scores, partial use of disputed items and the label flow.

    DOSSIER_SET=s1 .venv/Scripts/python paper/forests_eudr/figures/make_extra.py

Writes paper/forests_eudr/tables/tab_variants.tex, tab_threeclass.tex, tab_bykind.tex and runs/dossier/extra_analyses.json.
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
from scipy.stats import beta  # noqa: E402

from src.dossier.human_labels import law_form  # noqa: E402
from src.dossier.hybrid_ts_vlm import fit_predict, mcnemar, scores  # noqa: E402
from src.dossier.legal_vlm import ENSEMBLE, baselines, vlm_predictions  # noqa: E402
from src.dossier.score_round import e2_baselines  # noqa: E402

TAB = ROOT / "paper" / "forests_eudr" / "tables"
OUT = ROOT / "runs" / "dossier" / "extra_analyses.json"
CFG = ROOT / "annotation" / "private" / "config.json"
VAR = [("v2vi", "Vietnamese, primary"), ("v2vi_p2", "Vietnamese, reworded"), ("v2vi_shuf", "Vietnamese, shuffled options"),
       ("v2en", "Vietnamese, English options"), ("v2en_full", "English"), ("v2vi_sc", "Self-consistency (5 samples)"),
       ("v2vi_noimg", "Card only (no image)"), ("v2vi_imgonly", "Image only (no card)"), ("v2vi_v1card", "Card with GFC2020 V1")]


def cp(k, n):
    return (0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)))


def yn(s):
    return s.map({"yes": 1, "no": 0})


def ba(y, p):
    ok = p.notna()
    return scores(y[ok].values, p[ok].astype(int).values) | {"coverage": float(ok.mean())} if ok.sum() >= 5 else None


def main():
    meta = pd.read_csv(ROOT / "data/processed/dossier/s1/items_s1.csv").set_index("item_id")
    t = pd.read_csv(ROOT / "data/interim/dossier/legal_vlm/human_truth.csv")
    t = t[t.item_id.isin(meta.index)].set_index("item_id")
    blind = set(json.loads(CFG.read_text(encoding="utf-8"))["blind"])
    t["blind"] = t.index.isin(blind)
    t["split"] = meta.loc[t.index, "split"]
    vp, _ = vlm_predictions("Qwen3.5-9B")
    F = pd.read_csv(ROOT / "runs/dossier/hybrid_ts_vlm_report.features.csv").set_index("item_id")
    P = pd.concat([vp, baselines(), e2_baselines(F)], ignore_index=True)
    ens = vp[vp.method.isin(["vlm_" + m for m in ENSEMBLE])]
    W = P.pivot_table(index="item_id", columns="method", values="deforestation", aggfunc="first")
    Wf = P.pivot_table(index="item_id", columns="method", values="forest_2020", aggfunc="first")
    W["S1_ensemble"] = ens.groupby("item_id").deforestation.apply(lambda x: "yes" if (x == "yes").sum() >= 2 else "no")
    Wf["S1_ensemble"] = ens.groupby("item_id").forest_2020.apply(lambda x: "yes" if (x == "yes").sum() >= 3 else "no")
    res = {}

    # 1. all VLM variants + legal rule on the card + GFC V4, forest 2020 and deforestation
    y1, y2 = t[t.y1.notna()].y1.astype(int), t[t.y2.notna()].y2.astype(int)
    rows = []
    for key, name in VAR + [("S1_ensemble", "Ensemble S1 (five wordings)"), ("rules_legal", "Legal rules on the card"),
                            ("gfcV4_hansen", "GFC2020 V4 overlay")]:
        col = key if key in ("S1_ensemble", "rules_legal", "gfcV4_hansen") else "vlm_" + key
        r = {"key": key, "name": name}
        for tag, y, M in (("y1", y1, Wf), ("y2", y2, W)):
            for s, idx in (("all", y.index), ("blind", y.index[t.loc[y.index, "blind"]])):
                v = ba(y.loc[idx], yn(M[col].reindex(idx)))
                r[f"{tag}_{s}"] = None if v is None else round(v["bal_acc"], 2)
        rows.append(r)
    res["variants"] = rows
    pa, pb = yn(Wf.vlm_v2vi.reindex(y1.index)), yn(Wf.rules_legal.reindex(y1.index))
    ok = pa.notna() & pb.notna()
    res["mcnemar_y1_vlm_vs_rules"] = mcnemar(y1[ok].values, pa[ok].astype(int).values, pb[ok].astype(int).values)
    pa = yn(Wf.S1_ensemble.reindex(y1.index))
    ok = pa.notna() & pb.notna()
    res["mcnemar_y1_S1_vs_rules"] = mcnemar(y1[ok].values, pa[ok].astype(int).values, pb[ok].astype(int).values)

    # 2. three-class outcomes: share called deforestation by human verdict (undetermined = labelled, not disputed, y2 missing)
    lab = t[t.status != "disputed"]
    groups = {"compliant": lab.index[lab.y2 == 0], "non_compliant": lab.index[lab.y2 == 1],
              "undetermined": lab.index[lab.y2.isna()], "undetermined_cleared": lab.index[lab.y2.isna() & (lab.cause == "cleared_use_unseen")]}
    three = []
    for key, name in [("S1_ensemble", "VLM ensemble (S1)"), ("vlm_v2vi", "VLM, single prompt"), ("ndvi_rule", "NDVI trajectory rule"),
                      ("gfcV4_hansen", "GFC2020 V4 + loss"), ("gfcV1_hansen", "GFC2020 V1 + loss"), ("national_loss", "National map + loss"),
                      ("gfw_drivers", "GFC2020 V4 + drivers"), ("tmf_loss", "JRC TMF"), ("all_alerts", "Every alert")]:
        r = {"key": key, "name": name}
        for g, idx in groups.items():
            p = W[key].reindex(idx)
            r[g] = (int((p == "yes").sum()), int(p.isin(["yes", "no"]).sum()))
        three.append(r)
    res["three_class"] = {"n": {g: int(len(i)) for g, i in groups.items()}, "rows": three}

    # 3. by item kind
    kind = []
    for key, name in [("S1_ensemble", "VLM ensemble (S1)"), ("vlm_v2vi", "VLM, single prompt"), ("rules_legal", "Legal rules on the card"),
                      ("gfcV4_hansen", "GFC2020 V4"), ("gfcV1_hansen", "GFC2020 V1"), ("national_loss", "National map"), ("ndvi_rule", "NDVI rule")]:
        r = {"key": key, "name": name}
        for k in ("alert", "plot"):
            yy = y1[t.loc[y1.index, "kind"] == k]
            v = ba(yy, yn(Wf[key].reindex(yy.index)))
            r[f"y1_{k}"] = None if v is None else {m: round(v[m], 2) for m in ("bal_acc", "sensitivity", "specificity")}
        yy = y2[t.loc[y2.index, "kind"] == "alert"]
        v = ba(yy, yn(W[key].reindex(yy.index)))
        r["y2_alert"] = None if v is None else {m: round(v[m], 2) for m in ("bal_acc", "sensitivity", "specificity")}
        yy = y2[(t.loc[y2.index, "kind"] == "plot") & (y2 == 0)]  # plots not deforested
        r["y2_plot_false_red"] = round(float((yn(W[key].reindex(yy.index)) == 1).mean()), 2)
        kind.append(r)
    res["by_kind"] = {"n": {"y1_alert": [int((t.loc[y1.index, "kind"] == "alert").sum()), int(y1[t.loc[y1.index, "kind"] == "alert"].sum())],
                            "y1_plot": [int((t.loc[y1.index, "kind"] == "plot").sum()), int(y1[t.loc[y1.index, "kind"] == "plot"].sum())],
                            "y2_alert": [int((t.loc[y2.index, "kind"] == "alert").sum()), int(y2[t.loc[y2.index, "kind"] == "alert"].sum())],
                            "y2_plot": [int((t.loc[y2.index, "kind"] == "plot").sum()), int(y2[t.loc[y2.index, "kind"] == "plot"].sum())]},
                      "rows": kind}

    # 4. design-weighted shares of change classes on 2020-forest alerts (stratified bootstrap CI)
    a = t[(t.kind == "alert") & (t.y1 == 1) & t.cause.notna()].copy()
    a["stratum"] = meta.loc[a.index, "stratum"]
    pop = meta[meta.kind == "alert"].groupby("stratum")["pop"].first()
    rng = np.random.default_rng(42)
    classes = ["conversion_to_agriculture", "cleared_use_unseen", "plantation_rotation", "other_non_agricultural", "none"]

    def shares(d):
        w = d.stratum.map(pop / d.groupby("stratum").size())
        return {c: float((w * (d.cause == c)).sum() / w.sum()) for c in classes}
    est = shares(a)
    boots = []
    for _ in range(1000):
        b = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for _, g in a.groupby("stratum")])
        boots.append(shares(b))
    res["weighted_shares"] = {c: {"est": round(est[c], 3), "ci": [round(float(np.percentile([x[c] for x in boots], 2.5)), 3),
                                                                  round(float(np.percentile([x[c] for x in boots], 97.5)), 3)],
                                  "unweighted": round(float((a.cause == c).mean()), 3)} for c in classes}
    res["weighted_shares"]["n"] = int(len(a))

    # 5. threshold sweep of the GFC2020 V4 + Hansen overlay
    f = pd.read_parquet(ROOT / "data/processed/dossier/s1/legal_features.parquet").set_index("item_id")
    grid = []
    for ft in (0.3, 0.4, 0.5, 0.6, 0.7):
        fo = (f.gfcV4.fillna(0) >= ft).astype(int)
        b1 = scores(y1.values, fo.reindex(y1.index).values)["bal_acc"]
        for lt in (0.1, 0.2, 0.3, 0.4, 0.5):
            d = ((f.gfcV4.fillna(0) >= ft) & (f.hl_post2020.fillna(0) >= lt)).astype(int)
            grid.append({"forest_thr": ft, "loss_thr": lt, "y1_ba": round(b1, 3), "y2_ba": round(scores(y2.values, d.reindex(y2.index).values)["bal_acc"], 3)})
    res["gfc_sweep"] = {"best_y1": max(grid, key=lambda g: g["y1_ba"]), "best_y2": max(grid, key=lambda g: g["y2_ba"]),
                        "range_y1": [min(g["y1_ba"] for g in grid), max(g["y1_ba"] for g in grid)],
                        "range_y2": [min(g["y2_ba"] for g in grid), max(g["y2_ba"] for g in grid)]}

    # 6. strict forest definition (trees > 5 m required for humans)
    ys1, ys2 = t[t.y1_strict.notna()].y1_strict.astype(int), t[t.y2_strict.notna()].y2_strict.astype(int)
    res["strict"] = {"n_y1": int(len(ys1)), "n_y2": int(len(ys2)), "pos_y2": int(ys2.sum())}
    for key in ("S1_ensemble", "vlm_v2vi", "rules_legal", "gfcV4_hansen", "ndvi_rule"):
        v1, v2 = ba(ys1, yn(Wf[key].reindex(ys1.index))), ba(ys2, yn(W[key].reindex(ys2.index)))
        res["strict"][key] = {"y1": None if v1 is None else round(v1["bal_acc"], 3), "y2": None if v2 is None else round(v2["bal_acc"], 3)}

    # 7. primary test in full (blind TEST, deforestation)
    bt = y2[t.loc[y2.index, "blind"] & (t.loc[y2.index, "split"] == "test")]
    prim = {"n": int(len(bt)), "pos": int(bt.sum())}
    for key in ("vlm_v2vi", "gfcV4_hansen", "S1_ensemble"):
        p = yn(W[key].reindex(bt.index))
        acc_uw = float(((p == bt) & p.notna()).mean())
        tp, n_pos = int(((p == 1) & (bt == 1)).sum()), int(bt.sum())
        ok = p.notna()
        sc = scores(bt[ok].values, p[ok].astype(int).values)
        prim[key] = {"acc_uncertain_wrong": round(acc_uw, 3), "bal_acc": round(sc["bal_acc"], 3), "detected": [tp, n_pos],
                     "sens_ci": [round(x, 2) for x in cp(tp, n_pos)], "false_green": round(sc["false_green"], 3)}
    res["primary"] = prim

    # 8. uncertain-as-wrong balanced accuracy and coverage (all labels, deforestation)
    uw = {}
    for key in ("S1_ensemble", "vlm_v2vi", "vlm_v2vi_shuf", "E1_M1", "ndvi_rule", "gfcV4_hansen", "whisp_fao", "rules_legal", "whisp_like"):
        if key not in W and key != "E1_M1":
            continue
        if key == "E1_M1":
            continue
        p = yn(W[key].reindex(y2.index))
        cov = float(p.notna().mean())
        sens = float(((p == 1) & (y2 == 1)).sum() / (y2 == 1).sum())
        spec = float(((p == 0) & (y2 == 0)).sum() / (y2 == 0).sum())
        uw[key] = {"coverage": round(cov, 3), "ba_uncertain_wrong": round((sens + spec) / 2, 3)}
    res["uncertain_wrong_y2"] = uw

    # 9. disputed items: keep a target when all labels agree on it
    raw = pd.read_csv(ROOT / "data/goi_gan_nhan/phan_tich/nhan_nguoi_tho.csv")
    raw = raw[raw.vai_tro != "phan_xu"]
    dis = t.index[t.status == "disputed"]
    rec1 = {}
    for i in dis:
        rr = raw[raw.ma_diem == i]
        v = {law_form(r.su_dung_dat_2020, r.do_che_phu_2020, r.cay_cao_tren_5m, r.thay_doi_sau_2020)["y1"] for r in rr.itertuples()}
        if len(v) == 1 and None not in v:
            rec1[i] = v.pop()
    yb = pd.concat([y1[t.loc[y1.index, "blind"]], pd.Series(rec1, dtype=int)])
    sc_rec = {}
    for key in ("S1_ensemble", "vlm_v2vi", "rules_legal", "gfcV4_hansen", "gfcV1_hansen"):
        v = ba(yb, yn(Wf[key].reindex(yb.index)))
        sc_rec[key] = round(v["bal_acc"], 3)
    res["disputed_recovered_y1"] = {"n_recovered": len(rec1), "n_blind_y1": int(len(yb)), "pos": int(yb.sum()), "bal_acc": sc_rec}

    # 9b. cause on alerts: majority-class baseline and kappa (same items and grouping as score_round)
    from sklearn.metrics import cohen_kappa_score
    from src.dossier.legal_vlm import GRP_V3
    from src.dossier.score_round import DRIVERS
    dr = pd.read_csv(ROOT / "data/interim/dossier/s1/gfw_drivers.csv").set_index("item_id").driver
    cz = {}
    for sname, sub in (("all", t), ("blind", t[t.blind])):
        c = sub[sub.cause.notna() & sub.index.str.startswith("A")]
        gt = c.cause.map(GRP_V3)
        out = {"n5": int(len(c)), "majority5": round(float(gt.value_counts(normalize=True).iloc[0]), 3)}
        ok = gt.isin(["agriculture", "rotation", "other"]) & dr.reindex(c.index).notna()
        agri = (gt[ok] == "agriculture").astype(int)
        out["n2"], out["majority2"] = int(ok.sum()), round(float(max(agri.mean(), 1 - agri.mean())), 3)
        for key in ("vlm_v2vi", "vlm_v2vi_shuf"):
            pc = vp[vp.method == key].set_index("item_id").change.reindex(agri.index).map(GRP_V3)
            k2 = pc.notna()
            out[f"kappa2_{key}"] = round(float(cohen_kappa_score(agri[k2], (pc[k2] == "agriculture").astype(int))), 2)
        dd = (dr.reindex(agri.index).map(DRIVERS) == "agriculture").astype(int)
        out["kappa2_drivers"] = round(float(cohen_kappa_score(agri, dd)), 2)
        cz[sname] = out
    res["cause_baselines"] = cz

    # 10. label flow and effort
    res["flow"] = {"labels": int(len(raw) + 2), "by_role": raw.vai_tro.value_counts().to_dict(), "items": int(len(t)),
                   "blind_items": int(t.blind.sum()), "assisted_items": int((~t.blind).sum()),
                   "status": t.status.value_counts().to_dict(), "y1_determined": int(len(y1)), "y2_determined": int(len(y2)),
                   "median_s_blind": float(raw[raw.che_do == "mu"].so_giay.median()), "median_s_assisted": float(raw[raw.che_do == "goi_y"].so_giay.median())}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    write_tables(res)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str)[:6000])


def f2(x):
    return "--" if x is None else f"{x:.2f}"


def write_tables(res):
    L = [r"\begin{table}[H]",
         r"\caption{All nine preregistered VLM variants, the ensemble S1 and two references: balanced accuracy on all and on blind labels. "
         r"``Card only'' and ``image only'' isolate the two inputs of the VLM.\label{tab:variants}}",
         r"\newcolumntype{C}{>{\centering\arraybackslash}X}", r"\begin{tabularx}{\textwidth}{lCCCC}", r"\toprule",
         r" & \multicolumn{2}{c}{\textbf{Forest 2020}} & \multicolumn{2}{c}{\textbf{Deforestation}}\\",
         r"\textbf{Variant} & All & Blind & All & Blind\\", r"\midrule"]
    for r in res["variants"]:
        if r["key"] == "S1_ensemble":
            L.append(r"\midrule")
        L.append(f"{r['name']} & {f2(r['y1_all'])} & {f2(r['y1_blind'])} & {f2(r['y2_all'])} & {f2(r['y2_blind'])}\\\\")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    (TAB / "tab_variants.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    n = res["three_class"]["n"]
    L = [r"\begin{table}[H]",
         r"\caption{Calls of deforestation by reference verdict: number of items each method calls deforested among items the reference labels "
         rf"classify as compliant ($n={n['compliant']}$), non-compliant ($n={n['non_compliant']}$) and undetermined ($n={n['undetermined']}$, of which "
         rf"{n['undetermined_cleared']} cleared with no use seen yet). For the undetermined items no call is correct; a call of deforestation "
         r"there would exclude a plot whose use is not yet established.\label{tab:threeclass}}",
         r"\newcolumntype{C}{>{\centering\arraybackslash}X}", r"\begin{tabularx}{\textwidth}{lCCC}", r"\toprule",
         r"\textbf{Method} & \textbf{Compliant} & \textbf{Non-compliant} & \textbf{Undetermined}\\", r"\midrule"]
    for r in res["three_class"]["rows"]:
        cells = [f"{k}/{m} ({k / m:.0%})".replace("%", r"\%") if m else "--" for k, m in (r["compliant"], r["non_compliant"], r["undetermined"])]
        L.append(f"{r['name']} & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    (TAB / "tab_threeclass.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    nk = res["by_kind"]["n"]
    L = [r"\begin{table}[H]",
         r"\caption{Results by item kind, all labels: balanced accuracy (sensitivity/specificity). Alert patches are mostly forest in 2020 "
         rf"(forest 2020: {nk['y1_alert'][1]} of {nk['y1_alert'][0]}), plots mostly not ({nk['y1_plot'][1]} of {nk['y1_plot'][0]}). "
         rf"Deforestation is scored on alerts ($n={nk['y2_alert'][0]}$, {nk['y2_alert'][1]} deforested); of the {nk['y2_plot'][0]} plots with a "
         rf"determined conclusion only {nk['y2_plot'][1]} were deforested, so for plots the share of the {nk['y2_plot'][0] - nk['y2_plot'][1]} "
         rf"plots not deforested that is wrongly called deforested is given.\label{{tab:bykind}}}}",
         r"\begin{adjustwidth}{-\extralength}{0cm}", r"\newcolumntype{C}{>{\centering\arraybackslash}X}",
         r"\begin{tabularx}{\fulllength}{lCCCC}", r"\toprule",
         r"\textbf{Method} & \textbf{Forest 2020, alerts} & \textbf{Forest 2020, plots} & \textbf{Deforestation, alerts} & \textbf{Plots called deforested}\\",
         r"\midrule"]

    def c(v):
        return "--" if v is None else f"{v['bal_acc']:.2f} ({v['sensitivity']:.2f}/{v['specificity']:.2f})"
    for r in res["by_kind"]["rows"]:
        L.append(f"{r['name']} & {c(r['y1_alert'])} & {c(r['y1_plot'])} & {c(r['y2_alert'])} & {r['y2_plot_false_red']:.2f}\\\\")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{adjustwidth}", r"\end{table}"]
    (TAB / "tab_bykind.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
