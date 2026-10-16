"""Record the source of every stored label (person or model) and build label caches for the person-only reference.

    .venv/Scripts/python -m src.release.label_source --logs <folder with log_a2.jsonl, log_a3.jsonl, log_a4.jsonl>

A label counts as a model label when the labelling script logged it as saved for that account and item
(Claude Opus 5.5, zero-shot with the v4 labelling guide); all other labels were entered by people. Writes
data/goi_gan_nhan/phan_tich/label_source.csv and data/goi_gan_nhan/_cache/nhan_nguoi (person labels only, with the
adjudications made before the model labels were added).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/goi_gan_nhan/_cache/nhan_web_adj16oct"
MODEL = "model_claude-opus-5-5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--src", type=Path, default=SRC)
    a = ap.parse_args()
    logged = set()
    for acc in ("a2", "a3", "a4"):
        f = a.logs / f"log_{acc}.jsonl"
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("saved"):
                logged.add((acc, r["item"]))
    rows = []
    for d in sorted((a.src / "labels").iterdir()):
        for f in sorted((d / "items").glob("*.json")):
            rows.append({"account": d.name, "item_id": f.stem, "source": MODEL if (d.name, f.stem) in logged else "human"})
    for f in sorted((a.src / "adjudicated").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"account": "admin", "item_id": f.stem,
                     "source": "human_adjudication_of_model_proposal" if "Fable 5" in r.get("ghi_chu", "") else "human_adjudication"})
    s = pd.DataFrame(rows)
    out = ROOT / "data/goi_gan_nhan/phan_tich/label_source.csv"
    s.to_csv(out, index=False)
    print(s.groupby(["account", "source"]).size().to_string(), "\n->", out)
    # person-only cache
    dst = ROOT / "data/goi_gan_nhan/_cache/nhan_nguoi"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(a.src, dst)
    for acc, item in logged:
        (dst / "labels" / acc / "items" / f"{item}.json").unlink(missing_ok=True)
    for r in s[s.source == "human_adjudication_of_model_proposal"].itertuples():
        (dst / "adjudicated" / f"{r.item_id}.json").unlink()
    n = sum(1 for _ in (dst / "labels").rglob("*.json"))
    print(f"person-only cache: {n} labels, {len(list((dst / 'adjudicated').glob('*.json')))} adjudications -> {dst}")


if __name__ == "__main__":
    main()
