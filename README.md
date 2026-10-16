# vEUDR: verifying EUDR "deforestation-free" claims, facts first and law in code

vEUDR is the home of a study on verifying coffee plots in Vietnam's Central Highlands against the EU Deforestation
Regulation (EUDR, Regulation (EU) 2023/1115), and of the **EUDR-CH** benchmark dataset.

**Dataset:** [EUDR-CH on Kaggle](https://www.kaggle.com/datasets/somebodyknow/eudr-ch) (CC BY 4.0; version 0.96 includes reference labels for all 1000 items of the labelling batch, person and model labels marked by source) · **Paper:** *Facts First, Law in Code: Verifying
"Deforestation-Free" Coffee under the EU Deforestation Regulation in Vietnam's Central Highlands, with an Open
Benchmark* (submitted to *Forests*).

## The idea

The EUDR admits a commodity only if the plot it comes from was not deforested after 31 December 2020. Whether that
holds depends on two facts and on the law:

1. **What was the land used for on 31 December 2020?** Forest, wood plantation, coffee or agroforestry, other crops...
2. **What changed afterwards?** Conversion to agriculture, harvest and replanting of a plantation, clearing with no use
   seen yet, other change (fire with regrowth, logging, infrastructure), or none.
3. **The law** (Art. 2(3)-(4) of the Regulation, read with Section 11 of Commission guidance C(2026) 4972) turns these
   facts into a conclusion: compliant, non-compliant or undetermined.

Map overlays ("forest map 2020 + tree-cover loss = deforestation") skip the facts. In smallholder coffee landscapes,
where shaded coffee, plantation rotations and natural regrowth all change the canopy, this makes the conclusion
depend on which map, and which version of it, is used. vEUDR keeps the steps apart: people or models state the
facts from the satellite evidence, and one function applies the law, identically for every method and for the
reference labels.

```
evidence (Sentinel-2 sheets and scenes, Landsat, Sentinel-1)  ->  facts (Q1, Q2)  ->  Art. 2 rule (code)  ->  verdict
```

## What is in this repository

| Path | Content |
|---|---|
| `veudr/legal_rule.py` | The Art. 2 rule on factual answers (`apply_rule`), with English option codes |
| `veudr/data.py` | Reader for the EUDR-CH folder downloaded from Kaggle |
| `veudr/timeseries.py` | Sentinel-2, Landsat and Sentinel-1 features per item (model E1 of the paper) |
| `veudr/baselines.py` | Map-overlay baselines (GFC2020 V1 / V4 + Hansen loss, JRC TMF, Dynamic World, 1-km drivers map, every alert) and the NDVI trajectory rule |
| `veudr/metrics.py` | Balanced accuracy, false-green rate, exact McNemar test, block bootstrap |
| `docs/LABELLING.md` | How to organise reference labels for EUDR verification, with the set-up of this study as an example |
| `docs/DATASET.md` | Description of EUDR-CH (files, columns, pixel geometry, licences) |
| `examples/` | `quickstart.py` (load, baselines, plot an item), `score_baselines.py` (score baselines against the reference labels) and `make_labelling_batch.py` (draw a labelling batch) |
| `research_code/` | The pipeline as used for the paper: sampling, evidence, reference sheets, the vision-language model, Whisp |
| `scripts/build_kaggle.py`, `scripts/build_labels.py` | How the Kaggle release and its anonymised labels were assembled from the study's internal files |
| `tests/` | Tests of the legal rule |

The baselines and time-series features in `veudr` reproduce those of the paper exactly on the released data (all
1780 items; features to within 1e-14).

## Quick start

```bash
pip install -r requirements.txt
pip install -e .
# download EUDR-CH into ./eudr-ch: kaggle datasets download somebodyknow/eudr-ch --unzip -p eudr-ch
python examples/quickstart.py --data eudr-ch --item A1022
pytest -q
```

```python
from veudr import apply_rule
apply_rule("natural_forest", "10_or_more", "unclear", "conversion_to_agriculture")["verdict"]   # 'non_compliant'
apply_rule("wood_plantation", "10_or_more", "yes", "plantation_harvested_and_replanted")["verdict"]  # 'compliant'
apply_rule("natural_forest", "10_or_more", "unclear", "cleared_no_use_yet")["verdict"]       # 'undetermined'
```

## Main findings (1000 items; 328 person labels and 975 model labels, with person adjudication)

- The forest-2020 call is where map-based checks diverge. Of 294 coffee plots, GFC2020 version 1 calls 78 % forest in
  2020, version 4 calls 22 %, the reference labels 50 %. A vision-language model that reconciles the map sources of a
  multi-source evidence card with the Sentinel-2 sheets reached a balanced accuracy of 0.82-0.86 for this call (five-prompt
  ensemble; 0.87 on the person labels alone), against 0.75-0.77 for GFC2020 V4, mostly on coffee plots; switching the map
  version changed 2.4 % of its forest calls (46 % for a map overlay).
- Weighted to the alert population, only 16 % of alert patches on 2020 forest were converted to agriculture; 39 % were
  cleared with no use seen yet, which the guidance treats as undetermined.
- The deforestation conclusion is hard for every method: on all labels the best training-free methods (NDVI trajectory
  rule and five-prompt VLM ensemble, both 0.68) do not differ significantly from calling every alert deforestation (0.66).
  On the person labels alone they do better (0.75 and 0.72 against 0.65).
- The preregistered primary test (single-prompt VLM against GFC2020 V4 + Hansen loss on 152 blind test items, 17
  deforested) is mixed: higher accuracy (0.80 against 0.74, McNemar p = 0.18, not significant) and fewer false greens
  (5.8 % against 8.7 %).
- Model labels (Claude Opus 5.5 following the labelling guide) agreed with person labels about as well as two people agreed
  with each other on the facts (kappa 0.76 for forest 2020 on 44 items), but less well on the deforestation conclusion.

## Licences

Code: MIT (`LICENSE`). Dataset: CC BY 4.0, with the source products' terms (`DATA_LICENSE.md`).

## Citation

See `CITATION.cff`. The paper reference will be added on publication.
