"""Scores used in the paper: balanced accuracy, sensitivity, specificity, false-green rate, exact McNemar test and a
block-bootstrap interval of the difference in balanced accuracy (0.2 degree blocks, column block_0p2deg)."""
from __future__ import annotations

import numpy as np


def scores(y, pred) -> dict:
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    tp, tn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
    sens = tp / max((y == 1).sum(), 1)
    spec = tn / max((y == 0).sum(), 1)
    fg = ((pred == 0) & (y == 1)).sum() / max((pred == 0).sum(), 1)  # share of "no deforestation" calls that are wrong
    return {"n": int(len(y)), "n_pos": int(y.sum()), "acc": float((pred == y).mean()), "bal_acc": float((sens + spec) / 2),
            "sensitivity": float(sens), "specificity": float(spec), "false_green": float(fg)}


def mcnemar(y, a, b) -> dict:
    """Exact McNemar test of methods a and b on the same items."""
    from scipy.stats import binomtest
    y, a, b = map(np.asarray, (y, a, b))
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return {"only_a": n01, "only_b": n10, "p": float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0}


def boot_diff(y, a, b, groups, seed=42, B=500) -> tuple[float, float]:
    """95% block-bootstrap interval of bal_acc(a) - bal_acc(b)."""
    rng = np.random.default_rng(seed)
    y, a, b, groups = map(np.asarray, (y, a, b, groups))
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    d = []
    for _ in range(B):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[ii])) < 2:
            continue
        d.append(scores(y[ii], a[ii])["bal_acc"] - scores(y[ii], b[ii])["bal_acc"])
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if d else (np.nan, np.nan)
