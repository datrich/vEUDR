"""Exploratory analysis E1 (prereg runs/dossier/PREREG_s1.md section 21): satellite time series + answers of the VLM
variants -> deforestation after 2020, scored with spatial-block cross-validation on human labels.

    python -m src.dossier.hybrid_ts_vlm [--truth data/interim/dossier/legal_vlm/human_truth.csv]

Features come from data/goi_gan_nhan/cho_nguoi_gan/<item>/ (Sentinel-2 scene list with NDVI in the outline, Landsat,
Sentinel-1) and from the raw answers of the 9 preregistered VLM variants (no new GPU run). Model M1 = time series + VLM,
M2 = M1 + evidence-card map features; ablations TS only and VLM only. 5-fold GroupKFold on 0.2 deg blocks, repeated
with 5 block shuffles; fixed threshold 0.5 with balanced class weights. Writes runs/dossier/hybrid_ts_vlm_report.json.
This is a NEW method added after interim looks: exploratory, to be confirmed on labels collected after 15/10/2026.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("DOSSIER_SET", "s1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..common.paths import INTERIM, PROCESSED, RUNS  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data" / "goi_gan_nhan" / "cho_nguoi_gan"
SEED_CFG = ROOT / "annotation" / "private" / "config.json"
VLM = ["vlm_v2vi", "vlm_v2en", "vlm_v2vi_sc", "vlm_v2vi_noimg", "vlm_v2vi_imgonly", "vlm_v2vi_v1card", "vlm_v2vi_p2", "vlm_v2vi_shuf", "vlm_v2en_full"]
CHANGES = ["conversion_to_agriculture", "none", "selective_logging", "plantation_rotation", "fire_without_conversion"]
MAP_COLS = ["coffee_2020", "coffee_2024", "hl_pre2021", "hl_post2020", "tc2000", "tmf_natural", "tmf_plantation", "gfw_post2020",
            "dw_pre_trees", "dw_pre_crops", "dw_post_trees", "dw_post_crops", "dw_post_bare", "dw_trees_2025", "dw_crops_2025",
            "gfcV1", "gfcV4", "off_natural", "off_plantation", "h_meta_p90", "h_eth", "wc20_tree", "wc20_crop", "persistence", "natforest_p"]


# ------------------------------------------------------------------------------------------------ features
def _win(d: pd.DataFrame, t0, a_days, b_days, col):
    m = (d.ngay >= t0 + pd.Timedelta(days=a_days)) & (d.ngay < t0 + pd.Timedelta(days=b_days))
    return d.loc[m, col]


def _amp(s: pd.DataFrame, col) -> float:
    """Seasonal amplitude: spread of monthly medians (P90 - P10), needs >= 6 distinct months."""
    if len(s) < 6:
        return np.nan
    mo = s.groupby(s.ngay.dt.month)[col].median()
    return float(mo.quantile(0.9) - mo.quantile(0.1)) if len(mo) >= 6 else np.nan


def ts_features(iid: str, kind: str, first_date) -> dict:
    p = ANN / iid
    t0 = pd.Timestamp(first_date) if kind == "alert" and isinstance(first_date, str) else pd.Timestamp("2021-01-01")
    f = {}
    s = pd.read_csv(p / "s2_canh.csv", parse_dates=["ngay"])
    s = s[(s.phu_khung >= 0.95) & (s.ti_le_quang_ranh >= 0.9) & s.ndvi_ranh.notna()]
    s = s.assign(d=s.ngay.dt.normalize()).groupby("d", as_index=False).ndvi_ranh.mean().rename(columns={"d": "ngay"})
    pre, post = _win(s, t0, -365, 0, "ndvi_ranh"), _win(s, t0, 0, 183, "ndvi_ranh")
    f["s2_pre"] = pre.median()
    f["s2_post_min"] = post.min() if len(post) else np.nan
    f["s2_drop"] = f["s2_pre"] - f["s2_post_min"]
    f["s2_rec12"] = _win(s, t0, 270, 455, "ndvi_ranh").median()
    f["s2_rec24"] = _win(s, t0, 640, 820, "ndvi_ranh").median()
    f["s2_latest"] = s[s.ngay >= s.ngay.max() - pd.Timedelta(days=365)].ndvi_ranh.median() if len(s) else np.nan
    den = f["s2_pre"] - f["s2_post_min"]
    f["s2_recovery"] = (f["s2_latest"] - f["s2_post_min"]) / den if pd.notna(den) and den > 0.05 else np.nan
    f["s2_amp_pre"] = _amp(s[(s.ngay < t0) & (s.ngay >= t0 - pd.Timedelta(days=3 * 365))], "ndvi_ranh")
    f["s2_amp_post"] = _amp(s[s.ngay >= t0 + pd.Timedelta(days=365)], "ndvi_ranh")
    f["s2_n_post"] = int((s.ngay >= t0).sum())
    l = pd.read_csv(p / "landsat.csv", parse_dates=["ngay"])
    l = l[(l.ti_le_quang_ranh >= 0.9) & l.ndvi_ranh.notna() & (l.ngay < "2016-01-01")]
    f["ls_mean_0015"], f["ls_sd_0015"], f["ls_min_0015"] = l.ndvi_ranh.mean(), l.ndvi_ranh.std(), (l.ndvi_ranh.quantile(0.05) if len(l) else np.nan)
    r = pd.read_csv(p / "s1_radar.csv", parse_dates=["ngay"]).dropna(subset=["vh_db"])

    def vh(a, b):
        w = r[(r.ngay >= t0 + pd.Timedelta(days=a)) & (r.ngay < t0 + pd.Timedelta(days=b))]
        return w.groupby("quy_dao").vh_db.median().median() if len(w) else np.nan
    f["s1_vh_pre"], f["s1_vh_post"] = vh(-365, 0), vh(183, 548)
    last = r[r.ngay >= r.ngay.max() - pd.Timedelta(days=365)] if len(r) else r
    f["s1_vh_latest"] = last.groupby("quy_dao").vh_db.median().median() if len(last) else np.nan
    f["s1_vh_drop"] = f["s1_vh_pre"] - f["s1_vh_post"]
    f["s1_vh_rec"] = f["s1_vh_latest"] - f["s1_vh_post"]
    f["is_alert"] = int(kind == "alert")
    f["alert_year"] = t0.year if kind == "alert" else np.nan
    return f


def vlm_features(pred: pd.DataFrame) -> pd.DataFrame:
    v = pred[pred.method.isin(VLM)]
    g = v.groupby("item_id")
    out = pd.DataFrame({"vlm_share_defor": g.deforestation.apply(lambda x: (x == "yes").mean()),
                        "vlm_share_defor_unc": g.deforestation.apply(lambda x: (x == "uncertain").mean()),
                        "vlm_share_forest": g.forest_2020.apply(lambda x: (x == "yes").mean())})
    for c in CHANGES:
        out[f"vlm_share_{c}"] = g.change.apply(lambda x, c=c: (x == c).mean())
    b = v[v.method == "vlm_v2vi"].set_index("item_id")
    out["v2vi_defor"] = b.deforestation.map({"yes": 1, "no": 0, "uncertain": 0.5})
    out["v2vi_forest"] = b.forest_2020.map({"yes": 1, "no": 0, "uncertain": 0.5})
    return out


# ------------------------------------------------------------------------------------------------ models
def fit_predict(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, kind: str, seed: int) -> np.ndarray:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ug = np.unique(groups)
    perm = dict(zip(ug, np.random.default_rng(seed).permutation(len(ug))))
    g2 = np.array([perm[x] for x in groups])
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g2):
        if len(np.unique(y[tr])) < 2:
            oof[te] = y[tr].mean()
            continue
        if kind == "logit":
            m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                              LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        else:
            import lightgbm as lgb
            m = lgb.LGBMClassifier(num_leaves=7, n_estimators=200, learning_rate=0.05, min_child_samples=5,
                                   class_weight="balanced", random_state=seed, verbose=-1)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def scores(y, pred, prob=None) -> dict:
    from sklearn.metrics import roc_auc_score
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    tp, tn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
    sens = tp / max((y == 1).sum(), 1)
    spec = tn / max((y == 0).sum(), 1)
    fg = ((pred == 0) & (y == 1)).sum() / max((pred == 0).sum(), 1)  # share of "no deforestation" calls that are wrong
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "acc": round(float((pred == y).mean()), 3), "bal_acc": round(float((sens + spec) / 2), 3),
           "sensitivity": round(float(sens), 3), "specificity": round(float(spec), 3), "false_green": round(float(fg), 3)}
    if prob is not None and len(np.unique(y)) == 2:
        out["auc"] = round(float(roc_auc_score(y, prob)), 3)
    return out


def boot_diff(y, a, b, groups, rng, B=1000) -> list:
    """95% block-bootstrap CI of bal_acc(a) - bal_acc(b)."""
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    d = []
    for _ in range(B):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[ii])) < 2:
            continue
        d.append(scores(y[ii], a[ii])["bal_acc"] - scores(y[ii], b[ii])["bal_acc"])
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)] if d else [None, None]


def mcnemar(y, a, b) -> dict:
    from scipy.stats import binomtest
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return {"only_a": n01, "only_b": n10, "p": round(float(binomtest(n01, n01 + n10, 0.5).pvalue), 4) if n01 + n10 else 1.0}


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--truth", type=Path, default=INTERIM / "dossier" / "legal_vlm" / "human_truth.csv")
    ap.add_argument("--out", type=Path, default=RUNS / "dossier" / "hybrid_ts_vlm_report.json")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    from .legal_vlm import baselines, vlm_predictions
    t = pd.read_csv(a.truth)
    meta = pd.read_csv(PROCESSED / "dossier" / "s1" / "items_s1.csv").set_index("item_id")
    seed = json.loads((ROOT / "annotation" / "webapp" / "seed_dot12.json").read_text(encoding="utf-8"))
    items = {it["id"]: it for it in seed["items"]}
    blind = set(json.loads(SEED_CFG.read_text(encoding="utf-8"))["blind"])
    t = t[t.item_id.isin(items) & t.item_id.isin(meta.index)].copy()
    vp, _ = vlm_predictions("Qwen3.5-9B")
    bp = baselines()
    pred = pd.concat([vp, bp], ignore_index=True)
    ts = pd.DataFrame([{"item_id": i, **ts_features(i, items[i]["kind"], items[i].get("first_date"))} for i in t.item_id]).set_index("item_id")
    vf = vlm_features(pred)
    mf = pd.read_parquet(PROCESSED / "dossier" / "s1" / "legal_features.parquet").set_index("item_id")
    mf = mf[[c for c in MAP_COLS if c in mf.columns]].add_prefix("map_")
    F = ts.join(vf, how="left").join(mf, how="left")
    sets = {"TS": list(ts.columns), "VLM": list(vf.columns), "M1 (TS+VLM)": list(ts.columns) + list(vf.columns),
            "M2 (TS+VLM+maps)": list(ts.columns) + list(vf.columns) + list(mf.columns)}

    def ref(m, col):
        x = pred[pred.method == m].set_index("item_id")[col].map({"yes": 1, "no": 0})
        return x
    report = {"design": "prereg section 21 (E1), exploratory", "truth_file": str(a.truth), "targets": {}}
    rng = np.random.default_rng(42)
    for target, col in (("y2", "deforestation"), ("y1", "forest_2020")):
        d = t[t[target].notna()].set_index("item_id")
        d = d[d.index.isin(F.index)]
        y = d[target].astype(int).values
        groups = meta.loc[d.index, "block"].values
        X = F.loc[d.index]
        res = {"n": int(len(y)), "n_pos": int(y.sum()), "models": {}}
        probs = {}
        for name, cols in sets.items():
            for kind in ("logit", "lgbm"):
                reps = [fit_predict(X[cols], y, groups, kind, s) for s in range(5)]
                sc = [scores(y, (r >= 0.5).astype(int), r) for r in reps]
                key = f"{name} | {kind}"
                probs[key] = reps[0]
                res["models"][key] = {k: (round(float(np.mean([s[k] for s in sc])), 3), round(float(np.std([s[k] for s in sc])), 3))
                                      for k in sc[0] if k not in ("n", "n_pos")}
        refs = {"vlm_v2vi": ref("vlm_v2vi", col), "vlm_v2vi_shuf": ref("vlm_v2vi_shuf", col), "gfcV4_hansen": ref("gfcV4_hansen", col),
                "gfcV1_hansen": ref("gfcV1_hansen", col)}
        trivial = np.zeros(len(y), int) if target == "y2" else np.ones(len(y), int)
        subsets = {"all": np.ones(len(y), bool), "blind": d.index.isin(blind), "blind_test": d.index.isin(blind) & (meta.loc[d.index, "split"].values == "test")}
        res["compare"] = {}
        best = max((k for k in probs), key=lambda k: res["models"][k]["bal_acc"][0]) if target == "y2" else "M1 (TS+VLM) | logit"
        res["primary_model"] = "M1 (TS+VLM) | logit"
        for sname, msk in subsets.items():
            yy = y[msk]
            if len(yy) == 0:
                continue
            row = {"n": int(msk.sum()), "n_pos": int(yy.sum())}
            for mk in ("M1 (TS+VLM) | logit", "M1 (TS+VLM) | lgbm", "M2 (TS+VLM+maps) | logit", "TS | logit", best):
                pp = (probs[mk] >= 0.5).astype(int)
                row[mk] = scores(yy, pp[msk], probs[mk][msk])
            for rk, rv in refs.items():
                rr = rv.reindex(d.index).values
                ok = ~pd.isna(rr) & msk
                row[rk] = scores(y[ok], rr[ok].astype(int)) if ok.any() else None
            row["trivial"] = scores(yy, trivial[msk])
            m1 = (probs["M1 (TS+VLM) | logit"] >= 0.5).astype(int)
            for rk in ("vlm_v2vi", "gfcV4_hansen"):
                rr = refs[rk].reindex(d.index).values
                ok = ~pd.isna(rr) & msk
                row[f"M1_vs_{rk}"] = {"mcnemar": mcnemar(y[ok], m1[ok], rr[ok].astype(int)),
                                      "bal_acc_diff_ci95": boot_diff(y[ok], m1[ok], rr[ok].astype(int), groups[ok], rng, B=500)}
            res["compare"][sname] = row
        # importance: standardized logistic coefficients of M1 on all data
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        cols = sets["M1 (TS+VLM)"]
        Xi = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X[cols]))
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xi, y)
        res["m1_coefficients"] = dict(sorted(zip(cols, np.round(lr.coef_[0], 3).tolist()), key=lambda kv: -abs(kv[1])))
        res["missing_share"] = X[cols].isna().mean().round(3).to_dict()
        report["targets"][target] = res
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    F.to_csv(a.out.with_suffix(".features.csv"))
    for target, res in report["targets"].items():
        print(f"=== {target}: n={res['n']} pos={res['n_pos']}")
        for k, v in res["models"].items():
            print(f"  {k:28s} bal_acc {v['bal_acc'][0]:.3f}±{v['bal_acc'][1]:.3f}  sens {v['sensitivity'][0]:.3f}  spec {v['specificity'][0]:.3f}  fg {v['false_green'][0]:.3f}  auc {v.get('auc', (np.nan,))[0]}")
        for sname, row in res["compare"].items():
            print(f"  -- {sname}: n={row['n']} pos={row['n_pos']}")
            for k in ("M1 (TS+VLM) | logit", "M1 (TS+VLM) | lgbm", "M2 (TS+VLM+maps) | logit", "TS | logit", "vlm_v2vi", "vlm_v2vi_shuf", "gfcV4_hansen", "gfcV1_hansen", "trivial"):
                v = row.get(k)
                if v:
                    print(f"     {k:28s} n={v['n']:3d} bal_acc {v['bal_acc']:.3f} acc {v['acc']:.3f} sens {v['sensitivity']:.3f} spec {v['specificity']:.3f} fg {v['false_green']:.3f}")
            for rk in ("vlm_v2vi", "gfcV4_hansen"):
                print(f"     M1 vs {rk}: {row[f'M1_vs_{rk}']}")
    print("report ->", a.out)


if __name__ == "__main__":
    main()
