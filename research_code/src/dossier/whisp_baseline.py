"""FAO Whisp ("What is in that plot?") as a baseline on the same 140 items. Runs in its own venv because the Whisp
pre-release pins pandas < 3 (the project venv uses pandas 3.0.6):

    .venv\\Scripts\\python -c "from src.dossier.legal_vlm import items; items().to_file('data/interim/dossier/legal_vlm/items.geojson', driver='GeoJSON')"
    .venv_whisp\\Scripts\\python src\\dossier\\whisp_baseline.py      # -> data/interim/dossier/legal_vlm/whisp_items.csv

Whisp openforis-whisp 3.0.0a19 (alpha). Its perennial-crop risk (risk_pcrop) is the relevant flag for coffee.
"""
import os
from pathlib import Path

import ee
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("WHISP_SRC", ROOT / "data" / "interim" / "dossier" / "legal_vlm" / "items.geojson"))
DST = Path(os.environ.get("WHISP_DST", ROOT / "data" / "interim" / "dossier" / "legal_vlm" / "whisp_items.csv"))  # s1: set both


def main():
    load_dotenv(ROOT / ".env")
    ee.Initialize(project=os.environ["GEE_PROJECT"])
    import openforis_whisp as whisp
    df = whisp.whisp_formatted_stats_geojson_to_df(SRC, external_id_column="item_id")
    risk = whisp.whisp_risk(df)
    risk.to_csv(DST, index=False)
    cols = [c for c in risk.columns if c.startswith("risk") or c.startswith("Ind_")]
    print(risk.shape, cols)
    for c in [c for c in cols if c.startswith("risk")]:
        print(c, risk[c].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
