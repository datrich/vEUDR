# vEUDR: verifying EUDR "deforestation-free" claims, facts first and law in code

vEUDR is the home of a study on verifying coffee plots in Vietnam's Central Highlands against the EU Deforestation
Regulation (EUDR, Regulation (EU) 2023/1115), and of the **EUDR-CH** benchmark dataset.

**Dataset:** [EUDR-CH on Kaggle](https://www.kaggle.com/datasets/somebodyknow/eudr-ch) (CC BY 4.0) · **Paper:** *Facts First, Law in Code: Verifying
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
| `examples/` | `quickstart.py` (load, baselines, plot an item) and `make_labelling_batch.py` (draw a labelling batch) |
| `research_code/` | The pipeline as used for the paper: sampling, evidence, reference sheets, the vision-language model, Whisp |
| `scripts/build_kaggle.py` | How the Kaggle release was assembled from the study's internal files |
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

## Main findings (interim, on 331 human labels)

- The forest-2020 call is where map-based checks diverge. Of 60 labelled coffee plots, GFC2020 version 1 calls 75 %
  forest in 2020, version 4 calls 15 %, annotators 43 %. A vision-language model reading the Sentinel-2 sheets reached
  a balanced accuracy of 0.83-0.90 for this call, against 0.77-0.81 for GFC2020 V4 and 0.79-0.82 for FAO Whisp, and
  changed 2.4 % of its calls when the map version on its evidence card was switched (46 % for a map overlay).
- Only 26 % of alert patches on 2020 forest were converted to agriculture; 27 % were cleared with no use seen yet,
  which the guidance treats as undetermined.
- For the deforestation conclusion, a five-prompt ensemble and a time-series model outperformed map overlays
  (balanced accuracy 0.72-0.74 against 0.63-0.65), on a par with a simple NDVI trajectory rule.

Numbers will be updated when labelling is complete.

## Licences

Code: MIT (`LICENSE`). Dataset: CC BY 4.0, with the source products' terms (`DATA_LICENSE.md`).

## Citation

See `CITATION.cff`. The paper reference will be added on publication.
