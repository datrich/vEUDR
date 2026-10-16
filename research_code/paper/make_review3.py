"""Diagnostics requested in review round 3 (label sources).

    DOSSIER_SET=s1 .venv/Scripts/python paper/forests_eudr/figures/make_review3.py  -> runs/dossier/review3_diagnostics.json

1.2  how disagreements between people were resolved; majorities formed by model labels; agreement with the model
     labels of one item merged into one source
1.3  anchoring of the model labeller on the silver suggestion (assisted vs blind items)
1.4  form version and label source: change classes of person labels by form, of model labels, and on shared items
4.3  kappa between people and between model and person on the same base of items
4.4  how the items with both a person and a model label were selected
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DOSSIER_SET", "s1")
ROOT = Path(__file__).resolve().parents[1]  # research_code root
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from src.dossier.human_labels import law_form  # noqa: E402

CACHE = ROOT / "data/goi_gan_nhan/_cache/nhan_web_adj16oct"
src = pd.read_csv(ROOT / "data/goi_gan_nhan/phan_tich/label_source.csv")
items = pd.read_csv(ROOT / "data/processed/dossier/s1/items_s1.csv").set_index("item_id")
rows = []
for r in src[src.account != "admin"].itertuples():
    d = json.loads((CACHE / "labels" / r.account / "items" / f"{r.item_id}.json").read_text(encoding="utf-8"))
    z = law_form(d["su_dung_dat_2020"], d["do_che_phu_2020"], d["cay_cao_tren_5m"], d["thay_doi_sau_2020"])
    rows.append({"item": r.item_id, "uid": r.account, "src": "model" if r.source.startswith("model") else "person",
                 "role": d.get("vai_tro", ""), "mode": d.get("che_do", ""), "form": "v4" if "v4" in str(d.get("phien_ban_huong_dan", "")) else "v3",
                 "land": d["su_dung_dat_2020"], "change": d["thay_doi_sau_2020"], "y1": z["y1"], "y2": z["y2"], "cause": z["change"]})
L = pd.DataFrame(rows)
L = L[L.item.isin(items.index)]
res = {}

# 1.2 resolution of disagreements between people; majorities
person = L[L.src == "person"]
pt = pd.read_csv(ROOT / "data/interim/dossier/legal_vlm/human_truth_person.csv").set_index("item_id")
full = pd.read_csv(ROOT / "data/interim/dossier/legal_vlm/human_truth.csv").set_index("item_id")
disp = pt.index[pt.status == "disputed"]
res["person_disputed"] = {"n": int(len(disp)), "status_in_full_reference": full.loc[disp, "status"].value_counts().to_dict(),
                          "has_model_label": int(L[(L.src == "model") & L.item.isin(disp)].item.nunique())}
maj = full.index[full.status == "majority"]
mm = L[L.item.isin(maj)].groupby("item").src.agg(lambda s: dict(s.value_counts()))
res["majority_items"] = {"n": int(len(maj)), "with_model_labels": int(sum("model" in v for v in mm)),
                         "majority_needs_model_labels": int(sum(v.get("model", 0) >= 2 or (v.get("model", 0) >= 1 and v.get("person", 0) < 2) for v in mm))}
agr = full.index[full.status == "agreed"]
ag = L[L.item.isin(agr)].groupby("item").src.agg(lambda s: tuple(sorted(set(s))))
res["agreed_items"] = {"n": int(len(agr)), **{"+".join(k): int(v) for k, v in ag.value_counts().items()}}


def kap(a, b):
    ok = a.notna() & b.notna()
    a, b = a[ok].astype(str), b[ok].astype(str)
    return {"n": int(ok.sum()), "agreement": round(float((a.values == b.values).mean()), 3),
            "kappa": round(float(cohen_kappa_score(a, b)), 3) if ok.sum() > 1 and len(set(a) | set(b)) > 1 else None}


K = ("land", "change", "y1", "y2")
# 4.3 / 4.4: pairs on the same base. Person-person: the first two person labels of an item; model-person: the first person
# and the merged model label (model labels of an item merged: the first one, they agree in ~90%)
pp = person.groupby("item").filter(lambda g: g.uid.nunique() >= 2).groupby("item").head(2)
p1, p2 = pp.groupby("item").nth(0).set_index("item"), pp.groupby("item").nth(1).set_index("item")
pm_items = sorted(set(person.item) & set(L[L.src == "model"].item))
h = person.groupby("item").first()
m = L[L.src == "model"].groupby("item").first()
res["pairs"] = {"person_person": {k: kap(p1[k], p2.loc[p1.index, k]) for k in K},
                "model_person": {k: kap(h.loc[pm_items, k], m.loc[pm_items, k]) for k in K}}
# same base: items where both pairs exist are disjoint, so also report both pairs on items with y2 determined by every label
both_det = [i for i in pm_items if pd.notna(h.at[i, "y2"]) and pd.notna(m.at[i, "y2"])]
res["pairs"]["model_person_y2_base"] = len(both_det)
pp_det = [i for i in p1.index if pd.notna(p1.at[i, "y2"]) and pd.notna(p2.at[i, "y2"])]
res["pairs"]["person_person_y2_base"] = len(pp_det)
# person-person by role of the pair: calibration vs double-coded
for role in ("hieu_chinh", "kep"):
    ids = [i for i in p1.index if (person[(person.item == i)].role == role).any()]
    res["pairs"][f"person_person_{role}"] = {k: kap(p1.loc[ids, k], p2.loc[ids, k]) for k in K}
# 4.4 selection of the shared items
res["shared_items"] = {"n": len(pm_items),
                       "person_roles": person[person.item.isin(pm_items)].role.value_counts().to_dict(),
                       "model_roles": L[(L.src == "model") & L.item.isin(pm_items)].role.value_counts().to_dict(),
                       "blind": int(sum(full.loc[pm_items, "che_do"] == "mu")),
                       "kind": items.loc[pm_items, "kind"].value_counts().to_dict()}

# 1.3 anchoring of the model labeller (as for people: keep the suggestion on assisted vs same answer unseen on blind)
sug = json.loads((ROOT / "annotation/webapp/goi_y_dot12.json").read_text(encoding="utf-8"))["suggestions"]
FIELD = {"land": "su_dung_dat_2020", "change": "thay_doi_sau_2020"}
res["anchoring"] = {}
for who in ("model", "person"):
    d = L[L.src == who]
    out = {}
    for k, f in FIELD.items():
        for mode in ("goi_y", "mu"):
            x = d[d["mode"] == mode]
            same = [(sug.get(i, {}).get(f) == v) for i, v in zip(x.item, x[k]) if sug.get(i, {}).get(f)]
            out[f"{k}_{'assisted' if mode == 'goi_y' else 'blind'}"] = {"n": len(same), "same_as_suggestion": round(sum(same) / max(len(same), 1), 3)}
    res["anchoring"][who] = out

# 1.4 form version vs label source on the change classes of forest-2020 alerts
A = L[(L.y1 == 1) & L.item.isin(items.index[items.kind == "alert"])]
def shares(x):
    v = x.cause.value_counts(normalize=True).round(3).to_dict()
    return {"n": int(len(x)), "conversion": v.get("conversion_to_agriculture", 0.0), "cleared_no_use": v.get("cleared_use_unseen", 0.0),
            "rotation": v.get("plantation_rotation", 0.0), "other": v.get("other_non_agricultural", 0.0), "none": v.get("none", 0.0)}
res["form_vs_source"] = {"person_v3": shares(A[(A.src == "person") & (A.form == "v3")]), "person_v4": shares(A[(A.src == "person") & (A.form == "v4")]),
                         "model_v4": shares(A[A.src == "model"])}
shared_v4 = sorted(set(A[(A.src == "person") & (A.form == "v4")].item) & set(A[A.src == "model"].item))
res["form_vs_source"]["shared_person_v4_items"] = {"n": len(shared_v4),
                                                   "person": shares(A[(A.src == "person") & A.item.isin(shared_v4)]),
                                                   "model": shares(A[(A.src == "model") & A.item.isin(shared_v4)].groupby("item").head(1))}
(ROOT / "runs/dossier/review3_diagnostics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
