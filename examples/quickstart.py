"""Quick start: load EUDR-CH, compute the map-overlay baselines and the NDVI rule, and plot one item.

    python examples/quickstart.py --data path/to/eudr-ch [--item A1022]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from veudr import baselines, timeseries
from veudr.data import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--item", default="A1022")
    a = ap.parse_args()
    ds = Dataset(a.data)
    print(ds.items.groupby(["kind", "split"]).size().unstack())
    ov = baselines.map_overlays(ds.map_features)
    print("\nShare of items each overlay calls forest in 2020 / deforested:")
    print(ov.assign(f=ov.forest_2020.eq("yes"), d=ov.deforestation.eq("yes")).groupby("method")[["f", "d"]].mean().round(3))
    ts = timeseries.features(ds, [a.item])
    print("\n" + a.item, "time-series features:\n", ts.T.round(3).to_string())
    print(baselines.ndvi_rule(ts).to_string(index=False))
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1.6, 1]})
    ax0.imshow(Image.open(ds.sheet(a.item))), ax0.axis("off"), ax0.set_title(f"{a.item}: interpreter sheet")
    s2 = ds.scene_table("sentinel2", a.item)
    s2 = s2[(s2.box_coverage >= 0.95) & (s2.outline_clear_share >= 0.9)]
    ax1.plot(s2.datetime, s2.ndvi_outline, ".", ms=3, color="#2e7d32", label="Sentinel-2 NDVI in outline")
    ax1.axvline(pd.Timestamp("2020-12-31"), color="#c62828", ls="--", label="EUDR cut-off")
    ax1.legend(), ax1.set_ylim(-0.05, 1)
    fig.tight_layout()
    fig.savefig(f"{a.item}_quickstart.png", dpi=120)
    print(f"\nfigure -> {a.item}_quickstart.png")


if __name__ == "__main__":
    main()
