"""Readers for the EUDR-CH dataset as published on Kaggle (folder layout in docs/DATASET.md).

    from veudr.data import Dataset
    ds = Dataset("path/to/eudr-ch")
    ds.items.head(); ds.scene_table("sentinel2", "A1022")
"""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import pandas as pd


class Dataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not (self.root / "items.csv").exists():
            raise FileNotFoundError(f"{self.root} does not look like the EUDR-CH folder (items.csv missing)")

    @cached_property
    def items(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "items.csv", parse_dates=["first_alert_date"])

    @cached_property
    def outlines(self):
        import geopandas as gpd
        return gpd.read_file(self.root / "outlines.geojson")

    @cached_property
    def map_features(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "map_features.parquet")

    @cached_property
    def ndvi_series(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "ndvi_s2_l1c_series.parquet")

    @cached_property
    def _tables(self) -> dict:
        return {k: pd.read_parquet(self.root / f"{k}_scenes.parquet") for k in ("sentinel2", "landsat", "sentinel1")}

    @cached_property
    def human_labels(self) -> pd.DataFrame:
        """One row per label (version 0.95 and later): anonymised annotator, role, blind/assisted, factual answers."""
        return pd.read_csv(self.root / "labels" / "human_labels.csv")

    @cached_property
    def reference_labels(self) -> pd.DataFrame:
        """One row per labelled item: consensus status, forest_2020 and deforestation (1/0, empty = undetermined), verdict."""
        return pd.read_csv(self.root / "labels" / "reference_labels.csv")

    def scene_table(self, sensor: str, item_id: str | None = None) -> pd.DataFrame:
        """Per-scene records of one sensor ('sentinel2', 'landsat', 'sentinel1'), optionally for one item."""
        t = self._tables[sensor]
        return t if item_id is None else t[t.item_id == item_id].reset_index(drop=True)

    def sheet(self, item_id: str, kind: str = "interpreter") -> Path:
        """Path of the reference sheet ('interpreter': 10 windows, ~1 km; 'standard': the VLM's sheet)."""
        return self.root / "sheets" / kind / f"{item_id}.png"

    def sheet_layout(self, kind: str = "interpreter") -> dict:
        return json.loads((self.root / "sheets" / f"layout_{kind}.json").read_text(encoding="utf-8"))

    def scenes(self, item_id: str) -> pd.DataFrame:
        """Rendered single-date Sentinel-2 scenes of an item (items with has_scene_series only)."""
        d = self.root / "scenes" / item_id
        s = pd.read_csv(d / "scenes.csv", parse_dates=["date"])
        s["path"] = [d / f for f in s.file]
        return s
