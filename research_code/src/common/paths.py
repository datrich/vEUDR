"""Canonical project paths. Everything is relative to the repo root so runs are portable."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"
GAZETTEER = ROOT / "gazetteer"
PROMPTS = ROOT / "prompts"
ANNOTATION = ROOT / "annotation"
RUNS = ROOT / "runs"
PAPER = ROOT / "paper"

for _p in (RAW, INTERIM, PROCESSED, CACHE, RUNS):
    _p.mkdir(parents=True, exist_ok=True)
