"""C2 (prereg section 26): the VLM answers the questions of the person labelling form, Q1 from card + sheet and Q2 from
the sheet alone, and the person rule `human_labels.law_form` turns the answers into the legal outcome.

    DOSSIER_SET=s1 .venv/Scripts/python -m src.dossier.c2_local run [--limit N]   # Ollama on this machine, resumable
    DOSSIER_SET=s1 .venv/Scripts/python -m src.dossier.c2_local report            # -> runs/dossier/c2_report.json

Model: Qwen3.5-9B through Ollama (tag qwen3.5:9b), temperature 0, seed 7, JSON schema output. Items: the items with a
person label (data/goi_gan_nhan/phan_tich/label_source.csv).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("DOSSIER_SET", "s1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .human_labels import law_form  # noqa: E402
from .legal_vlm import IMG_ALERT, IMG_PLOT, OUT, SHEETS, _order  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = OUT / "raw_c2_ollama.jsonl"
REPORT = ROOT / "runs/dossier/c2_report.json"
MODEL = "qwen3.5:9b"
URL = "http://localhost:11434/api/chat"
WORDINGS = ["vi", "vi_p2", "vi_shuf", "en", "en_shuf"]

LAND = ["rừng_tự_nhiên", "rừng_trồng_lấy_gỗ", "cà_phê_hoặc_nông_lâm_kết_hợp", "cây_lâu_năm_khác", "cây_hằng_năm_hoặc_đất_trống",
        "xây_dựng_hoặc_mặt_nước", "không_rõ"]
LAND_VI = ["rừng tự nhiên (thường xanh, khộp, hỗn giao, tre nứa), kể cả rừng nghèo, rừng phục hồi",
           "rừng trồng lấy gỗ (keo, bạch đàn, thông…), kể cả lô vừa khai thác", "cà phê, kể cả dưới cây che bóng, hoặc nông lâm kết hợp",
           "cây lâu năm khác (cao su, điều, sầu riêng, hồ tiêu, cây ăn quả, mắc ca…)", "cây hằng năm (lúa, ngô, mì, nương rẫy) hoặc đất trống",
           "nhà cửa, đường, mặt nước", "không xác định được"]
LAND_EN = ["natural forest (evergreen, deciduous dipterocarp, mixed, bamboo), including poor or regrowing forest",
           "wood plantation (acacia, eucalyptus, pine...), including stands just harvested", "coffee, also under shade trees, or agroforestry",
           "other perennial crop (rubber, cashew, durian, pepper, fruit trees, macadamia...)", "annual crop (rice, maize, cassava, swidden) or bare land",
           "houses, roads, water", "cannot be determined"]
CANOPY = ["dưới_10", "từ_10_trở_lên", "không_rõ"]
TALL = ["có", "không", "không_rõ"]
CHANGE = ["không_thay_đổi", "chuyển_sang_nông_nghiệp", "khai_thác_rừng_trồng_rồi_trồng_lại", "đã_dọn_chưa_thấy_sử_dụng",
          "thay_đổi_khác_không_phải_nông_nghiệp", "không_rõ"]
CHANGE_VI = ["mất tán dưới 30% phần trong đường bao (mây, bóng, rụng lá mùa khô rồi xanh lại không tính)",
             "trên ít nhất 50% phần bị dọn thấy hàng cây trồng, hố trồng hoặc nương rẫy",
             "rừng trồng lấy gỗ bị khai thác rồi xanh lại đều như một lô",
             "đã dọn nhưng chưa thấy cây trồng, chưa thấy tái sinh gỗ trên 50% phần bị dọn",
             "cây gỗ tái sinh không theo hàng trên ít nhất 50% phần bị dọn, hoặc đường, hồ, mỏ, công trình, hoặc cháy rồi tái sinh",
             "không đủ ảnh rõ sau khi mất tán để xác định"]
CHANGE_EN = ["canopy loss below 30% of the outline (clouds, shadows, dry-season leaf fall that greens up do not count)",
             "crop rows, planting holes or swidden fields on at least 50% of the cleared part",
             "a wood plantation harvested and then greening up evenly as one stand",
             "cleared, but no crop and no woody regrowth on 50% of the cleared part yet",
             "unordered woody regrowth on at least 50% of the cleared part; or road, reservoir, mine, buildings; or fire followed by regrowth",
             "no clear image after the loss to decide"]


def schema_q1():
    return {"type": "object", "properties": {"quan_sat": {"type": "string"}, "su_dung_dat_2020": {"enum": LAND},
                                             "do_che_phu_2020": {"enum": CANOPY}, "cay_cao_tren_5m": {"enum": TALL},
                                             "do_tin_cay": {"enum": ["cao", "vừa", "thấp"]}},
            "required": ["quan_sat", "su_dung_dat_2020", "do_che_phu_2020", "cay_cao_tren_5m", "do_tin_cay"]}


def schema_q2():
    return {"type": "object", "properties": {"quan_sat": {"type": "string"}, "thay_doi_sau_2020": {"enum": CHANGE},
                                             "do_tin_cay": {"enum": ["cao", "vừa", "thấp"]}},
            "required": ["quan_sat", "thay_doi_sau_2020", "do_tin_cay"]}


def options(codes, desc, item_id, shuffle):
    order = _order(item_id, len(codes)) if shuffle else list(range(len(codes)))
    return "\n".join(f"   - {codes[i]}: {desc[i]}" for i in order)


def system(q: str, w: str, item_id: str) -> str:
    en, shuf = w.startswith("en"), w.endswith("shuf")
    if q == "q1":
        land = options(LAND, LAND_EN if en else LAND_VI, item_id, shuf)
        if en:
            return ("You interpret satellite images for due diligence under the EU Deforestation Regulation in Vietnam's Central "
                    "Highlands. Answer question 1 of the labelling form about the land INSIDE the yellow outline on 31 December 2020 "
                    "(for an alert patch: before the canopy loss); the law is applied afterwards by the system.\n"
                    f"su_dung_dat_2020 (land use):\n{land}\n"
                    "do_che_phu_2020: tree canopy cover in 2020, dưới_10 (below 10%), từ_10_trở_lên (10% or more) or không_rõ.\n"
                    "cay_cao_tren_5m: are most trees taller than about two storeys (5 m), or able to reach that height in place? có / không / không_rõ.\n"
                    "Maps on the evidence card are secondary evidence with known errors (global maps often call tree crops forest; the "
                    "national map may be outdated); weigh them against the images. First write quan_sat (at most 50 words), then answer; "
                    "do_tin_cay: cao / vừa / thấp. Answer with the JSON schema.")
        if w == "vi_p2":
            return ("Nhiệm vụ: đọc trang ảnh vệ tinh và thẻ bằng chứng rồi trả lời câu 1 của phiếu gán nhãn cho phần đất TRONG đường "
                    "bao vàng, tại thời điểm 31/12/2020 (mảng cảnh báo: trước khi mất tán). Hệ thống tự áp dụng luật sau đó.\n"
                    "Bản đồ trên thẻ có sai số: bản đồ toàn cầu hay gọi cây trồng là rừng, bản đồ quốc gia có thể cũ; hãy đối chiếu với ảnh.\n"
                    f"Câu 1a, su_dung_dat_2020:\n{land}\n"
                    "Câu 1b, do_che_phu_2020: độ che phủ tán cây năm 2020 là dưới_10, từ_10_trở_lên hay không_rõ.\n"
                    "Câu 1c, cay_cao_tren_5m: phần lớn cây cao hơn khoảng hai tầng nhà (5 m), hoặc có thể đạt độ cao đó tại chỗ: có / không / không_rõ.\n"
                    "Viết quan_sat trước (tối đa 50 từ), rồi trả lời; do_tin_cay là cao / vừa / thấp. Trả lời đúng JSON schema.")
        return ("Bạn là chuyên gia giải đoán ảnh vệ tinh, hỗ trợ thẩm định theo Quy định chống mất rừng của EU ở Tây Nguyên. Trả lời "
                "câu 1 của phiếu gán nhãn về phần đất BÊN TRONG đường bao vàng tại 31/12/2020 (với mảng cảnh báo: trước khi mất tán); "
                "phần áp dụng luật do hệ thống thực hiện.\n"
                f"su_dung_dat_2020 (sử dụng đất):\n{land}\n"
                "do_che_phu_2020: độ che phủ tán cây năm 2020, dưới_10 (dưới 10%), từ_10_trở_lên (từ 10% trở lên) hoặc không_rõ.\n"
                "cay_cao_tren_5m: phần lớn cây cao hơn khoảng hai tầng nhà (5 m), hoặc có thể đạt độ cao đó tại chỗ? có / không / không_rõ.\n"
                "Bản đồ trên thẻ chỉ là bằng chứng phụ và có thể sai (bản đồ toàn cầu hay nhầm cây trồng là rừng; bản đồ quốc gia có "
                "thể lỗi thời); hãy đối chiếu với ảnh. Trước hết viết quan_sat (tối đa 50 từ), rồi trả lời; do_tin_cay: cao / vừa / "
                "thấp. Trả lời đúng JSON schema.")
    chg = options(CHANGE, CHANGE_EN if en else CHANGE_VI, item_id, shuf)
    if en:
        return ("You interpret satellite images for due diligence under the EU Deforestation Regulation. Answer question 2 of the "
                "labelling form: what happened to the land INSIDE the yellow outline after 31 December 2020? Work through the steps "
                "in order: (a) is there a clear image after the loss? if not, không_rõ; (b) canopy loss on at least 30% of the "
                "outline? if not, không_thay_đổi; (c) the cleared part is mainly road, reservoir, mine or buildings: "
                "thay_đổi_khác_không_phải_nông_nghiệp; (d) crop rows, planting holes or swidden on at least 50% of the cleared part: "
                "chuyển_sang_nông_nghiệp; (e) a wood plantation greening up evenly as one stand: khai_thác_rừng_trồng_rồi_trồng_lại; "
                "(f) unordered woody regrowth on at least 50% of the cleared part: thay_đổi_khác_không_phải_nông_nghiệp; (g) otherwise "
                f"đã_dọn_chưa_thấy_sử_dụng.\nOptions:\n{chg}\n"
                "Compare the same season (deciduous forest sheds leaves), and check whether NDVI recovers. First write quan_sat (at "
                "most 50 words), then answer; do_tin_cay: cao / vừa / thấp. Answer with the JSON schema.")
    if w == "vi_p2":
        return ("Nhiệm vụ: xem trang ảnh vệ tinh và cho biết câu 2 của phiếu gán nhãn, tức phần đất TRONG đường bao vàng đã thay đổi "
                "thế nào sau 31/12/2020. Đi lần lượt từng bước, dừng ở bước đầu tiên đúng:\n"
                "a) không có ảnh rõ sau khi mất tán thì không_rõ; b) mất tán dưới 30% đường bao thì không_thay_đổi; c) phần bị dọn "
                "chủ yếu là đường, hồ, mỏ, nhà thì thay_đổi_khác_không_phải_nông_nghiệp; d) có hàng cây trồng, hố trồng hoặc nương rẫy "
                "trên ít nhất 50% phần bị dọn thì chuyển_sang_nông_nghiệp; e) rừng trồng xanh lại đều như một lô thì "
                "khai_thác_rừng_trồng_rồi_trồng_lại; f) cây gỗ tái sinh không theo hàng trên ít nhất 50% phần bị dọn thì "
                "thay_đổi_khác_không_phải_nông_nghiệp; g) còn lại là đã_dọn_chưa_thấy_sử_dụng.\n"
                f"Các lựa chọn:\n{chg}\n"
                "So ảnh cùng mùa và xem NDVI có hồi lại không. Viết quan_sat trước (tối đa 50 từ), rồi trả lời; do_tin_cay là cao / "
                "vừa / thấp. Trả lời đúng JSON schema.")
    return ("Bạn là chuyên gia giải đoán ảnh vệ tinh, hỗ trợ thẩm định theo Quy định chống mất rừng của EU. Trả lời câu 2 của phiếu "
            "gán nhãn: sau 31/12/2020, phần đất BÊN TRONG đường bao vàng đã thay đổi thế nào? Làm theo thứ tự các bước: (a) có ảnh "
            "rõ sau khi mất tán không? nếu không: không_rõ; (b) mất tán trên ít nhất 30% đường bao? nếu không: không_thay_đổi; "
            "(c) phần bị dọn chủ yếu là đường, hồ, mỏ, nhà: thay_đổi_khác_không_phải_nông_nghiệp; (d) hàng cây trồng, hố trồng "
            "hoặc nương rẫy trên ít nhất 50% phần bị dọn: chuyển_sang_nông_nghiệp; (e) rừng trồng lấy gỗ xanh lại đều như một lô: "
            "khai_thác_rừng_trồng_rồi_trồng_lại; (f) cây gỗ tái sinh không theo hàng trên ít nhất 50% phần bị dọn: "
            "thay_đổi_khác_không_phải_nông_nghiệp; (g) còn lại: đã_dọn_chưa_thấy_sử_dụng.\n"
            f"Các lựa chọn:\n{chg}\n"
            "So ảnh cùng mùa (rừng khộp rụng lá theo mùa) và xem NDVI có phục hồi về mức cũ không. Trước hết viết quan_sat (tối "
            "đa 50 từ), rồi trả lời; do_tin_cay: cao / vừa / thấp. Trả lời đúng JSON schema.")


def user_text(q: str, card: dict) -> str:
    obj = ("Đối tượng: một mảng có cảnh báo mất tán cây (GFW) xuất hiện sau mốc cắt." if card["kind"] == "alert" else
           "Đối tượng: một thửa giả định trong vùng trồng cà phê.")
    img = IMG_ALERT if card["kind"] == "alert" else IMG_PLOT
    body = card["text"] if q == "q1" else card["text"].splitlines()[0]  # Q2: no map evidence, only what the object is
    return f"{img}\n\n{obj}\n\nTHẺ BẰNG CHỨNG:\n{body}"


def call(sysmsg: str, text: str, image_b64: str, schema: dict) -> tuple[str, float]:
    body = {"model": MODEL, "stream": False, "think": False, "format": schema,
            "options": {"temperature": 0, "seed": 7, "num_ctx": 8192},
            "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": text, "images": [image_b64]}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return out["message"]["content"], time.time() - t0


def person_items() -> list[str]:
    s = pd.read_csv(ROOT / "data/goi_gan_nhan/phan_tich/label_source.csv")
    meta = pd.read_csv(ROOT / "data/processed/dossier/s1/items_s1.csv").set_index("item_id")
    return sorted(set(s.loc[s.source == "human", "item_id"]) & set(meta.index))


def run(limit: int | None):
    cards = {c["item_id"]: c for c in map(json.loads, open(OUT / "cards.jsonl", encoding="utf-8")) if c["card"] == "v4"}
    done = set()
    if RAW.exists():
        done = {json.loads(x)["id"] for x in open(RAW, encoding="utf-8") if x.strip()}
    ids = person_items()[:limit] if limit else person_items()
    todo = [(w, q, i) for i in ids for w in WORDINGS for q in ("q1", "q2") if f"{w}|{q}|{i}" not in done]
    print(f"{len(ids)} items, {len(todo)} calls to make ({len(done)} done)", flush=True)
    t_start = time.time()
    with open(RAW, "a", encoding="utf-8") as fh:
        for n, (w, q, i) in enumerate(todo, 1):
            img = base64.b64encode((SHEETS / f"{i}.png").read_bytes()).decode()
            try:
                raw, sec = call(system(q, w, i), user_text(q, cards[i]), img, schema_q1() if q == "q1" else schema_q2())
            except Exception as e:  # noqa: BLE001 - keep going; the call is retried on the next run
                print("error", w, q, i, e, flush=True)
                continue
            fh.write(json.dumps({"id": f"{w}|{q}|{i}", "raw": raw, "seconds": round(sec, 2)}, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t_start
                print(f"{n}/{len(todo)}  {el / n:.1f} s/call  eta {(len(todo) - n) * el / n / 60:.0f} min", flush=True)


def predictions() -> pd.DataFrame:
    rows = {}
    for x in open(RAW, encoding="utf-8"):
        r = json.loads(x)
        w, q, i = r["id"].split("|")
        try:
            rows.setdefault((w, i), {})[q] = json.loads(r["raw"])
        except json.JSONDecodeError:
            rows.setdefault((w, i), {})[q] = None
    out = []
    for (w, i), d in rows.items():
        a, b = d.get("q1"), d.get("q2")
        if not a or not b:
            continue
        z = law_form(a.get("su_dung_dat_2020", "không_rõ"), a.get("do_che_phu_2020", "không_rõ"), a.get("cay_cao_tren_5m", "không_rõ"),
                     b.get("thay_doi_sau_2020", "không_rõ"))
        out.append({"wording": w, "item_id": i, "y1": z["y1"], "y2": z["y2"], "change": b.get("thay_doi_sau_2020"),
                    "land": a.get("su_dung_dat_2020")})
    return pd.DataFrame(out)


def report():
    from .hybrid_ts_vlm import boot_diff, mcnemar, scores
    from .legal_vlm import ENSEMBLE, vlm_predictions
    from .score_round import FEAT, e2_baselines
    p = predictions()
    meta = pd.read_csv(ROOT / "data/processed/dossier/s1/items_s1.csv").set_index("item_id")
    t = pd.read_csv(ROOT / "data/interim/dossier/legal_vlm/human_truth_person.csv").set_index("item_id")
    t = t[t.index.isin(meta.index)]
    full = p.groupby("item_id").wording.nunique()
    ids = full.index[full == len(WORDINGS)]
    g = p[p.item_id.isin(ids)].groupby("item_id")
    c2 = pd.DataFrame({"y1": g.y1.apply(lambda s: 1 if (s == 1).sum() >= 3 else (0 if (s == 0).sum() >= 3 else np.nan)),
                       # as S1: deforestation if >= 2 of 5; undetermined if >= 3 of 5 leave it undetermined; else no
                       "y2": g.y2.apply(lambda s: 1 if (s == 1).sum() >= 2 else (np.nan if s.isna().sum() >= 3 else 0))})
    single = p[p.wording == "vi"].set_index("item_id")[["y1", "y2"]]
    vp, _ = vlm_predictions("Qwen3.5-9B")
    ens = vp[vp.method.isin(["vlm_" + m for m in ENSEMBLE])]
    yn = {"yes": 1, "no": 0}
    s1 = pd.DataFrame({"y1": ens.groupby("item_id").forest_2020.apply(lambda x: 1 if (x == "yes").sum() >= 3 else 0),
                       "y2": ens.groupby("item_id").deforestation.apply(lambda x: 1 if (x == "yes").sum() >= 2 else 0)})
    F = pd.read_csv(FEAT).set_index("item_id")
    e2 = e2_baselines(F)
    every = e2[e2.method == "all_alerts"].set_index("item_id").deforestation.map(yn)
    rng = np.random.default_rng(42)
    res = {"n_items_complete": int(len(ids)), "n_calls": int(sum(1 for _ in open(RAW, encoding="utf-8"))),
           "median_seconds": float(pd.read_json(RAW, lines=True).seconds.median())}
    for yv in ("y1", "y2"):
        y = t[t[yv].notna()][yv].astype(int)
        y = y[y.index.isin(ids)]
        out = {"n": int(len(y)), "pos": int(y.sum())}
        for nm, pr in (("C2_S", c2[yv]), ("C2_single", single[yv]), ("S1", s1[yv]), ("every_alert", every if yv == "y2" else None)):
            if pr is None:
                continue
            q = pr.reindex(y.index)
            ok = q.notna()
            s = scores(y[ok].values, q[ok].astype(int).values)
            out[nm] = {"bal_acc": round(s["bal_acc"], 3), "sens": round(s["sensitivity"], 3), "spec": round(s["specificity"], 3),
                       "coverage": round(float(ok.mean()), 3)}
        refs = ("S1", "every_alert") if yv == "y2" else ("S1",)
        for ref in refs:
            a = c2[yv].reindex(y.index)
            b = (s1[yv] if ref == "S1" else every).reindex(y.index)
            ok = (a.notna() & b.notna()).values
            out[f"C2_S_vs_{ref}"] = {"mcnemar": mcnemar(y.values[ok], a.values[ok].astype(int), b.values[ok].astype(int)),
                                     "ba_diff_ci": boot_diff(y.values[ok], a.values[ok].astype(int), b.values[ok].astype(int),
                                                             meta.loc[y.index, "block"].values[ok], rng, B=1000)}
        res[yv] = out
    und = t[t.y2.isna() & (t.cause == "cleared_use_unseen")].index.intersection(ids)
    res["undetermined_recognised"] = {"n": int(len(und)), "share_C2_S_undetermined": round(float(c2.loc[und, "y2"].isna().mean()), 3) if len(und) else None}
    res["criteria"] = {"C2_1_beats_every_alert": bool(res["y2"]["C2_S_vs_every_alert"]["ba_diff_ci"][0] > 0),
                       "C2_3_forest_not_worse_than_S1_by_0.03": bool(res["y1"]["C2_S_vs_S1"]["ba_diff_ci"][0] >= -0.03)}
    REPORT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run(a.limit) if a.cmd == "run" else report()


if __name__ == "__main__":
    main()
