# Research code

The pipeline as it was run for the paper (Python 3.12, Google Earth Engine, vLLM), kept as used so that each step
can be checked. Comments and prompts are partly in Vietnamese. Paths are relative to a project root with
`configs/`, `data/` and `runs/`; Earth Engine needs a project id in the environment variable `GEE_PROJECT`.

| Module | Step |
|---|---|
| `src/dossier/sample.py` | Study area, coffee plots and their strata |
| `src/dossier/silver_set.py` | The 1780-item sample (alert patches, plots), waves, DEV/TEST blocks; interpreter and standard sheets; the silver labeller |
| `src/dossier/ref_sheets.py` | Standard reference sheets and the Sentinel-2 L1C NDVI series |
| `src/dossier/evidence.py`, `src/alerts/*` | Alert patches and per-item evidence from Earth Engine |
| `src/dossier/legal_vlm.py` | Evidence card, prompts and the nine VLM variants, baselines, metrics |
| `src/dossier/official_map.py` | GFC2020 versions; comparison with the official 2020 forest map (download code only, data not redistributed) |
| `src/dossier/whisp_baseline.py` | FAO Whisp on the same items (own environment) |
| `src/dossier/human_labels.py` | The frozen legal rule (`law`, SHA-256 prefix 22f32e93bb5b640d) and Cohen's kappa |
| `src/dossier/hybrid_ts_vlm.py` | Time-series features, the exploratory model E1, McNemar and block-bootstrap helpers |
| `src/dossier/score_round.py` | Scoring of all methods on the four label sets (all, TEST, blind, blind TEST) |
| `paper/make_*.py` | Every table and figure of the paper and the supplementary analyses (by kind, three verdicts, weighted scores, time-series classifiers, scores by label source, results by land-use class) |

The paper scripts read the reference labels (`data/interim/dossier/legal_vlm/human_truth.csv`, built from the released
labels), the list of blind items (`annotation/private/config.json`) and the VLM predictions; they write to
`runs/dossier/` and `paper/forests_eudr/tables/`.

Not included: the remote GPU runner used to send VLM batches to the lab server (any vLLM installation can run the
payloads that `legal_vlm.py` writes), modules of earlier project phases, and the labelling records. The clean,
documented versions of the rule, baselines, features and metrics are in `veudr/`.
