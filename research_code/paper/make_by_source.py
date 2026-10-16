"""Scores by label source and agreement between model and person labels.

    DOSSIER_SET=s1 .venv/Scripts/python paper/forests_eudr/figures/make_by_source.py

Needs data/goi_gan_nhan/phan_tich/label_source.csv (src.release.label_source), the full reference
data/interim/dossier/legal_vlm/human_truth.csv and the person-only reference human_truth_person.csv. Runs
src.dossier.score_round on the person-only reference and on the items labelled by the model only, and writes
runs/dossier/score_round_report_{person,modelonly}.json, runs/dossier/label_source_agreement.json and
tables/tab_bysource.tex. The full report (score_round_report.json) is produced last, on all labels.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("DOSSIER_SET", "s1")
ROOT = Path(__file__).resolve().parents[1]  # research_code root
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from src.dossier.human_labels import law_form  # noqa: E402

L = ROOT / "data/interim/dossier/legal_vlm"
RUNS = ROOT / "runs/dossier"
TAB = ROOT / "paper/forests_eudr/tables"
CACHE = ROOT / "data/goi_gan_nhan/_cache/nhan_web_adj16oct/labels"
src = pd.read_csv(ROOT / "data/goi_gan_nhan/phan_tich/label_source.csv")
items = pd.read_csv(ROOT / "data/processed/dossier/s1/items_s1.csv").set_index("item_id")
lab = src[src.account != "admin"]
person_items = set(lab[lab.source == "human"].item_id)
t = pd.read_csv(L / "human_truth.csv")
t[~t.item_id.isin(person_items)].to_csv(L / "human_truth_modelonly.csv", index=False)


def score(truth, tag):
    subprocess.run([sys.executable, "-m", "src.dossier.score_round", "--truth", str(truth)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    for ext in (".json", ".md"):
        shutil.copy(RUNS / f"score_round_report{ext}", RUNS / f"score_round_report_{tag}{ext}")


score(L / "human_truth_person.csv", "person")
score(L / "human_truth_modelonly.csv", "modelonly")
subprocess.run([sys.executable, "-m", "src.dossier.score_round"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

# agreement: model vs person on items with both, person vs person on items with two person labels
rows = []
for r in lab.itertuples():
    f = CACHE / r.account / "items" / f"{r.item_id}.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    z = law_form(d["su_dung_dat_2020"], d["do_che_phu_2020"], d["cay_cao_tren_5m"], d["thay_doi_sau_2020"])
    rows.append({"item": r.item_id, "uid": r.account, "src": "model" if r.source.startswith("model") else "human",
                 "land_use_2020": d["su_dung_dat_2020"], "change_after_2020": d["thay_doi_sau_2020"],
                 "forest_2020": z["y1"], "deforestation": z["y2"]})
A = pd.DataFrame(rows)
A = A[A.item.isin(items.index)]
K = ("land_use_2020", "change_after_2020", "forest_2020", "deforestation")


def agree(a, b):
    ok = a.notna() & b.notna()
    a, b = a[ok].astype(str), b[ok].astype(str)
    return {"n": int(ok.sum()), "agreement": round(float((a.values == b.values).mean()), 3),
            "kappa": round(float(cohen_kappa_score(a, b)), 3)}


h = A[A.src == "human"].groupby("item").first()
m = A[A.src == "model"].groupby("item").first()
both = h.index.intersection(m.index)
res = {"n_labels": A.groupby("src").size().to_dict(), "items_both": int(len(both)),
       "model_vs_person": {k: agree(h.loc[both, k], m.loc[both, k]) for k in K}}
hh = A[A.src == "human"].groupby("item").filter(lambda g: g.uid.nunique() >= 2).groupby("item").head(2)
p1, p2 = hh.groupby("item").nth(0).set_index("item"), hh.groupby("item").nth(1).set_index("item")
res["person_vs_person"] = {k: agree(p1[k], p2.loc[p1.index, k]) for k in K}
mm = A[A.src == "model"].groupby("item").filter(lambda g: g.uid.nunique() >= 2).groupby("item").head(2)
q1, q2 = mm.groupby("item").nth(0).set_index("item"), mm.groupby("item").nth(1).set_index("item")
res["model_vs_model"] = {k: agree(q1[k], q2.loc[q1.index, k]) for k in K}
(RUNS / "label_source_agreement.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))

# table: main methods on the three references
M = [("S1_ensemble", "VLM, five-prompt ensemble (S1)"), ("vlm_v2vi", "VLM, single prompt"), ("ndvi_rule", "NDVI trajectory rule"),
     ("all_alerts", "Every alert is deforestation"), ("rules_legal", "Legal rules on the card"), ("gfcV4_hansen", "GFC2020 V4 + Hansen loss"),
     ("gfcV1_hansen", "GFC2020 V1 + Hansen loss"), ("national_loss", "National 2020 map + loss"), ("whisp_fao", "Whisp (FAO), official")]
reps = {tag: json.loads((RUNS / f"score_round_report{s}.json").read_text(encoding="utf-8"))["sets"]["all"]
        for tag, s in (("all", ""), ("person", "_person"), ("model", "_modelonly"))}


def cell(tag, y, k):
    sc = reps[tag].get(y, {})
    sc = sc.get("scores", sc)
    v = sc.get(k)
    return f"{v['bal_acc']:.2f}" if isinstance(v, dict) and "bal_acc" in v else "--"


def n(tag, y):
    return reps[tag][y].get("n")


Lt = [r"\begin{table}[H]",
      r"\caption{Balanced accuracy by source of the reference labels (all items of each reference). \emph{Person}: consensus of the "
      r"labels entered by people only (items without a person label, and items on which people disagree, left out); \emph{model only}: "
      r"items labelled only by Claude Opus~5.5 (zero-shot, labelling guide v4), with person adjudication of disagreements; "
      r"\emph{all}: every label, as in the other tables.\label{tab:bysource}}",
      r"\begin{adjustwidth}{-\extralength}{0cm}", r"\newcolumntype{C}{>{\centering\arraybackslash}X}",
      r"\begin{tabularx}{\fulllength}{lCCCCCC}", r"\toprule",
      r" & \multicolumn{3}{c}{\textbf{Forest 2020}} & \multicolumn{3}{c}{\textbf{Deforestation}}\\",
      r"\textbf{Method} & Person & Model only & All & Person & Model only & All\\", r"\midrule"]
for k, name in M:
    Lt.append(f"{name} & " + " & ".join(cell(tag, y, k) for y in ("y1", "y2") for tag in ("person", "model", "all")) + r"\\")
Lt += [r"\midrule", "Items & " + " & ".join(str(n(tag, y)) for y in ("y1", "y2") for tag in ("person", "model", "all")) + r"\\",
       r"\bottomrule", r"\end{tabularx}", r"\end{adjustwidth}", r"\end{table}"]
(TAB / "tab_bysource.tex").write_text("\n".join(Lt) + "\n", encoding="utf-8")
print("table ->", TAB / "tab_bysource.tex")
