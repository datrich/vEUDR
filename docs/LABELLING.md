# Organising reference labels for EUDR verification

This guide describes how to collect human reference labels that can be used to check EUDR "deforestation-free"
claims, and how we set this up for the Central Highlands study with the EUDR-CH dataset. It is tool-agnostic: the
form can be a spreadsheet, a form application or an annotation platform, as long as the rules below are kept.

## 1. What the labels are for

The EUDR (Regulation (EU) 2023/1115) asks two questions about a plot:

1. Was the land **forest on 31 December 2020**?
2. Was that forest **converted to agricultural use** afterwards?

Both depend on legal definitions (Art. 2(3), 2(4); Commission guidance C(2026) 4972, Section 11). People and models
are good at describing land, less consistent at applying definitions. We therefore ask annotators for **facts only**
and apply the law with one function (`veudr.legal_rule.apply_rule`). The same function is applied to the labels and to
every method, so differences between methods are differences in reading the evidence, not in reading the law.

## 2. Unit of labelling

- The unit is the **land inside the item outline**: an alert patch or a plot.
- The surroundings are context only. A 0.3 ha patch inside a large forest is forest: the 0.5 ha threshold applies to
  the forest the patch belongs to, not to the patch.
- The reference date is 31 December 2020. For alerts, Q1 describes the land before the loss.

## 3. The questions

| Field | Options |
|---|---|
| Q1 Land use on 31 Dec 2020 | natural forest; wood plantation; coffee or agroforestry; other perennial crop; annual crop or bare land; built-up or water; unclear |
| Shade trees (if coffee) | yes; no; unclear |
| Canopy cover in 2020 | < 10 %; >= 10 %; unclear |
| Most trees taller than about 5 m | yes; no; unclear |
| Q2 Change after 31 Dec 2020 | no change; converted to agriculture; plantation harvested and replanted; cleared, no use seen yet; other change, not a conversion of forest; unclear |
| Month of the change | YYYY-MM; unknown |
| Shares (inside the outline) | canopy loss; signs of use on the cleared part; woody regrowth on the cleared part; steps of 10 % |
| High-resolution image near the cut-off | within 2019-2021; only outside 2019-2021; none |
| Year of the latest image examined | 2023 or earlier; 2024; 2025; 2026 |
| Confidence | high; medium; low |

Coffee is agricultural use even under shade trees (agroforestry is excluded from "forest", Art. 2(4)).

### Answer Q2 in a fixed order

Free choice of the change class gave low agreement in our first round (Cohen's kappa 0.38). A fixed sequence with
thresholds on the land inside the outline works better; stop at the first question that gives an answer:

| Step | Question | If | Answer |
|---|---|---|---|
| a | Is there a valid high-resolution image after 31 Dec 2020 (and after the loss, if any)? | no | unclear, low confidence |
| b | Canopy loss on at least **30 %** of the outline? | no | no change (note "loss ~x %" if any) |
| c | Was the land forest in 2020 (Q1 = natural forest or wood plantation)? | no | other change, not a conversion of forest |
| d | Is the cleared part mainly road, reservoir, mine or buildings? | yes | other change, not a conversion of forest (infrastructure) |
| e | Crop rows, planting holes or swidden fields on at least **50 %** of the cleared part in the latest image? | yes | converted to agriculture |
| f | Q1 = wood plantation, and the cleared part greens up evenly as one stand? | yes | plantation harvested and replanted |
| g | Unordered woody regrowth (crowns of 2-3 m or more) on at least **50 %** of the cleared part? | yes | other change, not a conversion of forest (regrowth) |
| | | no | cleared, no use seen yet |

Consistency rules (reject the answer and ask again): land that was not forest in 2020 cannot be "converted to
agriculture", "harvested and replanted" or "cleared, no use seen yet"; natural forest cannot be "harvested and
replanted"; if the loss started before January 2021, Q1 describes the land on 31 December 2020.

### How the answers become a legal conclusion

```python
from veudr.legal_rule import apply_rule
apply_rule("wood_plantation", "10_or_more", "yes", "plantation_harvested_and_replanted")
# {'forest_2020': 1, 'deforestation': 0, ..., 'verdict': 'compliant'}
apply_rule("natural_forest", "10_or_more", "unclear", "cleared_no_use_yet")["verdict"]
# 'undetermined': temporarily unstocked land remains forest until a use is visible
```

"Cleared, no use seen yet" is **undetermined**, not deforestation. Record the year of the latest image examined so
these items can be looked at again when newer imagery is available.

## 4. What annotators see

- **Show:** the interpreter sheet (Sentinel-2 composites in true and false colour around the cut-off and after the
  alert, NDVI series), single-date Sentinel-2 scenes, NDVI and Sentinel-1 VH charts, and high-resolution imagery
  in Google Earth (historical imagery) or Esri World Imagery Wayback, opened at the item outline.
- **Do not show:** forest maps, loss or alert products, alert confidence, crop-probability models, other annotators'
  labels or model outputs. Annotators must be blind to everything the methods under test use.
- Exception, if you use model suggestions to save time: see Section 6.

## 5. Design of the labelling campaign

1. **Sample** items with a stratified random design and keep the inclusion probabilities (stratum population and
   number sampled), so that results can be weighted to the population (Olofsson et al. 2014).
2. **Split** DEV/TEST by spatial blocks *before* any label exists. Only DEV may be used to train or tune; TEST is
   scored once.
3. **Calibrate:** all annotators label the same calibration items first (we used 20 per form version, 40 in all),
   then discuss disagreements with the guide before the main round.
4. **Double-code** a random 20 % of the remaining items with a second, independent annotator.
5. **Adjudicate** disagreements on land use, change or the forest-2020 reading; keep adjudications separate from the
   original labels and never overwrite them.
6. **Consensus rule:** adjudication if any, otherwise the agreed label, otherwise a strict majority (three or more
   annotators); items still disputed get no reference label until adjudicated.
7. **Report agreement** (Cohen's kappa on double-coded items, mean pairwise kappa on calibration items) for land use,
   change, the forest-2020 reading and the deforestation conclusion. Our target was kappa >= 0.6 for Q2 and the
   three-level conclusion.
8. **Preregister** the questions, the legal function, the metrics and the decision rules before scoring, with a hash of
   each frozen component, and record every later change as a dated amendment.

## 6. Model suggestions (optional) and how to keep them honest

Pre-filling the form with a model's answer saves time but anchors people. If you use suggestions:

- assign items to **blind** and **assisted** at random within each DEV/TEST x kind x stratum cell (we used 30 % blind),
  and make calibration and double-coded items blind;
- record the mode and the suggestion shown with each label, on the side that stores the labels (not in the
  annotator's browser);
- never suggest the imagery fields or the confidence;
- score methods on **blind labels only**, and estimate anchoring as the difference between how often annotators keep
  the suggestion on assisted items and how often they give the same answer unseen on blind items.

In our study annotators kept the suggested change class on 51 % of assisted items but chose it unseen on 34 % of blind
items; for land use the difference was 71 % against 67 %, for canopy cover and height none.

## 7. Quality checks per label

Flag, but keep as given: conversion of land that was not forest in 2020; rotation on land that was not a plantation;
a change dated before 2021; a high-resolution image only outside 2019-2021; "cleared, no use seen yet" when the
latest image examined is older than 2025 (worth a second look).

## 8. Example: the set-up for EUDR-CH

`examples/make_labelling_batch.py` reproduces the design on the Kaggle dataset:

```bash
python examples/make_labelling_batch.py --data path/to/eudr-ch --out batch/ --n 1000 --seed 12
```

It draws a stratified batch from `items.csv` (items with scene series), keeps the DEV/TEST split, picks 40 calibration
items and 20 % double-coded items, assigns 30 % blind per cell, and writes:

- `batch/assignments.csv`: item, role (calibration / main / double), mode (blind / assisted), annotator slot;
- `batch/form_template.csv`: one empty row per item and annotator with the fields of Section 3;
- `batch/item_links.csv`: paths to the interpreter sheet and scenes, the item centroid and a Google Earth link.

Collect the filled forms, then turn them into reference labels:

```python
import pandas as pd
from veudr.legal_rule import apply_rule
f = pd.read_csv("filled_forms.csv")
law = f.apply(lambda r: apply_rule(r.land_use_2020, r.canopy_2020, r.trees_over_5m, r.change_after_2020), axis=1)
labels = pd.concat([f, pd.DataFrame(list(law))], axis=1)
```

**Worked example (item A1022).** An alert patch of 1.86 ha first alerted on 13 July 2021. The interpreter sheet shows
tree cover in the dry seasons of 2019 and 2020 and a clearing after the alert; the NDVI in the outline drops after
the alert and recovers within about two years. The annotator, labelling blind, answered: Q1 wood plantation,
canopy >= 10 %, Q2 plantation harvested and replanted. Rule: forest in 2020, not deforestation, compliant. A map
overlay (GFC2020 V4 forest + Hansen loss) calls the same patch deforested.

## 9. Effort

A label took a median of 112 s with the interpreter sheet and Google Earth open. With 1000 items, 20 % double coding
and 40 calibration items per annotator, plan about 40 person-hours of labelling for three annotators, plus
adjudication.
