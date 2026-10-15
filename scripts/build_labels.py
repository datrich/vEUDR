"""Human reference labels for the public EUDR-CH dataset (version 0.95): anonymised, English codes, no free text.

    .venv/Scripts/python -m src.release.build_labels [--src data/goi_gan_nhan/_cache/nhan_web_new] [--out release/kaggle/eudr-ch/labels]

Reads the raw records pulled from the labelling site (human_labels.pull_web) and the reference labels of
src/dossier/human_labels.py (data/interim/dossier/legal_vlm/human_truth.csv). Annotator accounts are replaced by
annotator_1..3 and adjudicator; free-text notes, timestamps, image dates typed by annotators and the model
suggestions shown on assisted items are left out. Items outside the s1 sample (pilot trial labels) are dropped.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("DOSSIER_SET", "s1")
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ANON = {"a1": "annotator_1", "a3": "annotator_2", "a4": "annotator_3", "admin": "adjudicator"}
LAND = {"rừng_tự_nhiên": "natural_forest", "rừng_trồng_lấy_gỗ": "wood_plantation", "cà_phê_hoặc_nông_lâm_kết_hợp": "coffee_or_agroforestry",
        "cây_lâu_năm_khác": "other_perennial_crop", "cây_hằng_năm_hoặc_đất_trống": "annual_crop_or_bare", "xây_dựng_hoặc_mặt_nước": "built_up_or_water",
        "không_rõ": "unclear"}
CHANGE = {"không_thay_đổi": "no_change", "chuyển_sang_nông_nghiệp": "conversion_to_agriculture",
          "khai_thác_rừng_trồng_rồi_trồng_lại": "plantation_harvested_and_replanted", "đã_dọn_chưa_thấy_sử_dụng": "cleared_no_use_yet",
          "thay_đổi_khác_không_phải_nông_nghiệp": "other_not_a_conversion", "không_rõ": "unclear"}
# canopy: forms before v3 had finer classes; every class from 10 % up maps to 10_or_more (as the legal rule reads them)
CANOPY = {"dưới_10": "under_10", "từ_10_trở_lên": "10_or_more", "10_40": "10_or_more", "40_70": "10_or_more", "trên_70": "10_or_more",
          "không_rõ": "unclear"}
YNU = {"có": "yes", "không": "no", "không_rõ": "unclear", "": ""}
HR = {"có_2019_2021": "within_2019_2021", "chỉ_ngoài_2019_2021": "only_outside_2019_2021", "không_có": "none", "": ""}
LATEST = {"đến_2023": "2023_or_earlier", "2024": "2024", "2025": "2025", "2026": "2026", "": ""}
CONF = {"cao": "high", "vừa": "medium", "thấp": "low"}
ROLE = {"hieu_chinh": "calibration", "chinh": "main", "kep": "double_coded", "phan_xu": "adjudication"}
MODE = {"mu": "blind", "goi_y": "assisted"}
CAUSE = {"conversion_to_agriculture": "conversion_to_agriculture", "cleared_use_unseen": "cleared_no_use_yet",
         "plantation_rotation": "plantation_harvested_and_replanted", "other_non_agricultural": "other_not_a_conversion", "none": "no_change"}


def form(v: str) -> str:
    return "v4" if "v4" in str(v) else ("v3" if "v3" in str(v) else "v2")


def month(v: str) -> str:
    return "" if v in ("", None) else ("unknown" if v == "không_rõ" else v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data/goi_gan_nhan/_cache/nhan_web_new")
    ap.add_argument("--out", type=Path, default=ROOT / "release/kaggle/eudr-ch/labels")
    a = ap.parse_args()
    items = pd.read_csv(ROOT / "data/processed/dossier/s1/items_s1.csv").set_index("item_id")
    cfg = json.loads((ROOT / "annotation/vercel_app/private/config.json").read_text(encoding="utf-8"))
    blind = set(cfg["blind"])
    recs = []
    for uid_dir in sorted((a.src / "labels").iterdir()):
        for f in sorted((uid_dir / "items").glob("*.json")):
            recs.append((uid_dir.name, json.loads(f.read_text(encoding="utf-8"))))
    for f in sorted((a.src / "adjudicated").glob("*.json")):
        recs.append(("admin", json.loads(f.read_text(encoding="utf-8"))))
    rows = []
    for uid, r in recs:
        if r["ma_diem"] not in items.index:
            continue
        rows.append({
            "item_id": r["ma_diem"], "annotator": ANON[uid], "role": ROLE[r.get("vai_tro", "phan_xu" if uid == "admin" else "chinh")],
            "item_mode": "blind" if r["ma_diem"] in blind else "assisted", "form_version": form(r.get("phien_ban_huong_dan", "")),
            "land_use_2020": LAND[r["su_dung_dat_2020"]], "shade_trees": YNU.get(r.get("cay_che_bong", ""), ""),
            "canopy_2020": CANOPY[r["do_che_phu_2020"]], "trees_over_5m": YNU[r["cay_cao_tren_5m"]],
            "change_after_2020": CHANGE[r["thay_doi_sau_2020"]], "change_month": month(r.get("thoi_diem_thay_doi", "")),
            "share_canopy_loss": r.get("ty_le_mat_tan", ""), "share_signs_of_use": r.get("ty_le_su_dung", ""),
            "share_woody_regrowth": r.get("ty_le_tai_sinh", ""),
            "hr_image_near_cutoff": HR.get(r.get("anh_gan_2020", ""), ""), "latest_image_year": LATEST.get(r.get("nam_anh_moi_nhat", ""), ""),
            "confidence": CONF.get(r.get("do_tin_cay", ""), ""), "seconds": r.get("so_giay", "")})
    lab = pd.DataFrame(rows).sort_values(["item_id", "role", "annotator"])
    a.out.mkdir(parents=True, exist_ok=True)
    lab.to_csv(a.out / "human_labels.csv", index=False)
    t = pd.read_csv(ROOT / "data/interim/dossier/legal_vlm/human_truth.csv")
    t = t[t.item_id.isin(items.index)]
    ref = pd.DataFrame({
        "item_id": t.item_id, "kind": t.kind, "status": t.status, "n_labels": t.n_labels,
        "item_mode": ["blind" if i in blind else "assisted" for i in t.item_id], "form_version": t.phieu,
        "calibration_item": t.calibration.astype(bool), "forest_2020": t.y1.astype("Int64"), "deforestation": t.y2.astype("Int64"),
        "forest_2020_strict": t.y1_strict.astype("Int64"), "deforestation_strict": t.y2_strict.astype("Int64"),
        "change_class": t.cause.map(CAUSE), "confidence": t.conf,
        "split": items.loc[t.item_id, "split"].values})
    ref["verdict"] = ref.deforestation.map({1: "non_compliant", 0: "compliant"}).fillna("undetermined")
    ref.loc[ref.status == "disputed", "verdict"] = "disputed"
    ref.to_csv(a.out / "reference_labels.csv", index=False)
    print(f"labels: {len(lab)} rows on {lab.item_id.nunique()} items ({lab.annotator.value_counts().to_dict()}); "
          f"reference: {len(ref)} items, {ref.verdict.value_counts().to_dict()} -> {a.out}")


if __name__ == "__main__":
    main()
