"""Draw a labelling batch from EUDR-CH with the design of docs/LABELLING.md.

    python examples/make_labelling_batch.py --data path/to/eudr-ch --out batch/ [--n 1000] [--annotators 3] [--seed 12]

Stratified by kind x stratum (proportional), DEV/TEST kept as published; 40 calibration items labelled by every
annotator; 20 % of the other items double-coded; 30 % blind in every DEV/TEST x kind x stratum cell (all calibration
and double-coded items blind). Writes assignments.csv, form_template.csv and item_links.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FORM = ["land_use_2020", "shade_trees", "canopy_2020", "trees_over_5m", "change_after_2020", "change_month",
        "share_canopy_loss", "share_signs_of_use", "share_woody_regrowth", "hr_image_near_cutoff", "latest_image_year",
        "confidence", "note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--annotators", type=int, default=3)
    ap.add_argument("--n-calibration", type=int, default=40)
    ap.add_argument("--double-share", type=float, default=0.2)
    ap.add_argument("--blind-share", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=12)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    it = pd.read_csv(a.data / "items.csv")
    pool = it[it.has_scene_series].copy()  # items with single-date scenes for the annotators
    n = min(a.n, len(pool))
    # proportional allocation to kind x stratum
    cells = pool.groupby(["kind", "stratum"])
    take = (cells.size() / len(pool) * n).round().astype(int)
    batch = pd.concat([g.sample(min(len(g), take[k]), random_state=a.seed) for k, g in cells], ignore_index=True)
    batch = batch.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    ids = batch.item_id.to_numpy()
    calib = set(rng.choice(ids, a.n_calibration, replace=False))
    rest = [i for i in ids if i not in calib]
    double = set(rng.choice(rest, int(round(a.double_share * len(rest))), replace=False))
    # blind: calibration + double-coded first, topped up to blind-share in every split x kind x stratum cell
    blind = set(calib) | double
    for _, g in batch.groupby(["split", "kind", "stratum"]):
        need = int(round(a.blind_share * len(g))) - len(blind & set(g.item_id))
        free = [i for i in g.item_id if i not in blind]
        if need > 0 and free:
            blind |= set(rng.choice(free, min(need, len(free)), replace=False))
    rows = []
    for k, i in enumerate(ids):
        mode = "blind" if i in blind else "assisted"
        if i in calib:
            rows += [{"item_id": i, "role": "calibration", "mode": mode, "annotator": s + 1} for s in range(a.annotators)]
        else:
            first = k % a.annotators + 1
            rows.append({"item_id": i, "role": "main", "mode": mode, "annotator": first})
            if i in double:
                rows.append({"item_id": i, "role": "double", "mode": mode, "annotator": first % a.annotators + 1})
    asg = pd.DataFrame(rows).merge(batch[["item_id", "kind", "stratum", "split"]], on="item_id")
    a.out.mkdir(parents=True, exist_ok=True)
    asg.to_csv(a.out / "assignments.csv", index=False)
    asg[["item_id", "annotator", "role"]].assign(**{c: "" for c in FORM}).to_csv(a.out / "form_template.csv", index=False)
    links = batch[["item_id", "kind", "lon", "lat", "first_alert_date"]].assign(
        interpreter_sheet=lambda d: "sheets/interpreter/" + d.item_id + ".png",
        scenes_folder=lambda d: "scenes/" + d.item_id + "/",
        google_earth=lambda d: [f"https://earth.google.com/web/@{la},{lo},500a,1500d,35y,0h,0t,0r" for lo, la in zip(d.lon, d.lat)])
    links.to_csv(a.out / "item_links.csv", index=False)
    print(f"{len(batch)} items, {len(calib)} calibration, {len(double)} double-coded, {len(blind)} blind;"
          f" {len(asg)} labels for {a.annotators} annotators -> {a.out}")


if __name__ == "__main__":
    main()
