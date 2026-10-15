"""Score the map-overlay baselines and the NDVI rule against the reference labels of EUDR-CH (version 0.95+).

    python examples/score_baselines.py --data path/to/eudr-ch [--blind-only]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from veudr import baselines, timeseries
from veudr.data import Dataset
from veudr.metrics import scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--blind-only", action="store_true", help="score on items labelled without a model suggestion")
    a = ap.parse_args()
    ds = Dataset(a.data)
    ref = ds.reference_labels
    if a.blind_only:
        ref = ref[ref.item_mode == "blind"]
    pred = pd.concat([baselines.map_overlays(ds.map_features), baselines.ndvi_rule(timeseries.features(ds))])
    rows = []
    for target in ("forest_2020", "deforestation"):
        y = ref.dropna(subset=[target]).set_index("item_id")[target].astype(int)
        for m, g in pred.groupby("method"):
            p = g.set_index("item_id")[target].reindex(y.index).map({"yes": 1, "no": 0})
            ok = p.notna()
            if ok.sum() >= 5:
                s = scores(y[ok].values, p[ok].astype(int).values)
                rows.append({"target": target, "method": m, "n": int(ok.sum()), "bal_acc": round(s["bal_acc"], 3),
                             "sensitivity": round(s["sensitivity"], 3), "specificity": round(s["specificity"], 3)})
    print(pd.DataFrame(rows).sort_values(["target", "bal_acc"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
