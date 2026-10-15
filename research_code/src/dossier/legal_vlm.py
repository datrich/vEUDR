"""EUDR legal-definition reasoning with a local VLM over a multi-source evidence card (pilot on the 140 silver items).

    python -m src.dossier.legal_vlm features   # GEE + official lots -> data/processed/dossier/legal_features.parquet
    python -m src.dossier.legal_vlm cards      # evidence cards E01.. -> data/interim/dossier/legal_vlm/cards.jsonl (+ .md)
    python -m src.dossier.legal_vlm run [--variants full,noimg,imgonly,nodef,v1swap,sc] [--gpu 0] [--gpu-mem 0.85]
    python -m src.dossier.legal_vlm eval       # vs silver labels + baselines -> runs/dossier/legal_vlm_report.{json,md}

Idea: EUDR art. 2 is split into checkable criteria (F1 area > 0.5 ha, F2 trees > 5 m, F3 canopy > 10 %, F4 not
predominantly agricultural use, incl. agricultural plantations such as coffee agroforestry and rubber, F5 temporarily
unstocked forest land, D1 conversion to agricultural use after 31/12/2020). The VLM gets the legal text, the criteria,
an evidence card (official 2020 lot, GFC2020, GFT2020, TMF 2020-2025, Hansen, canopy heights, FDP coffee / rubber,
Nature Trace, Dynamic World, spectral change, fire) and the reference sheet image, and must answer every criterion
citing evidence ids. Outputs are checked programmatically (cited ids exist, legal logic consistent) and compared
with deterministic baselines on the same card. Variants: full, noimg (card only), imgonly (image only), nodef (no
legal text), v1swap (GFC2020 V1 value instead of V4 under the same label), sc (5 samples, self-consistency).
Silver labels (Claude agents, imagery only) are a pipeline test, never a reference standard.
"""
from __future__ import annotations

import argparse
import base64
import itertools
import json
import re
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ..alerts.gee_common import ee_init, to_gdf
from ..common.paths import INTERIM, PROCESSED, RUNS
from .official_map import GFC_VERSIONS
from .sample import SET

# item set (DOSSIER_SET, see sample.py): unset = dot0, the 140 items the design was developed on
D = PROCESSED / "dossier" / SET if SET else PROCESSED / "dossier"
OUT = INTERIM / "dossier" / SET / "legal_vlm" if SET else INTERIM / "dossier" / "legal_vlm"
SHEETS = INTERIM / "dossier" / SET / "ref_sheets" if SET else INTERIM / "dossier" / "ref_sheets"
SILVER = "annotation/dossier_reference/labels_ai_silver.csv"
MODEL = "Qwen3.5-9B"
TMF25 = "projects/JRC/TMF/v1_2025/AnnualChanges"
NATFOR = "projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection"
FORTY = "projects/nature-trace/assets/forest_typology/forest_typology_2020_v1_0_collection"
PERSIST = "projects/forestdatapartnership/assets/community_forests/ForestPersistence_2020"
RUBBER = "projects/forestdatapartnership/assets/rubber/model_2026a"
META_H = "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"
ETH_H = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"
GLAD_H = "projects/sat-io/open-datasets/GLAD/GEDI_V27"

# official lot codes (diễn biến rừng, Thông tư 33/2018) seen in the Central Highlands
LDLR = {"TXG": "rừng gỗ tự nhiên thường xanh giàu", "TXB": "rừng gỗ tự nhiên thường xanh trung bình",
        "TXN": "rừng gỗ tự nhiên thường xanh nghèo", "TXK": "rừng gỗ tự nhiên thường xanh nghèo kiệt",
        "TXP": "rừng gỗ tự nhiên thường xanh phục hồi", "RLG": "rừng khộp (lá rộng rụng lá) giàu",
        "RLB": "rừng khộp (lá rộng rụng lá) trung bình", "RLN": "rừng khộp (lá rộng rụng lá) nghèo",
        "RLK": "rừng khộp (lá rộng rụng lá) nghèo kiệt", "RLP": "rừng khộp (lá rộng rụng lá) phục hồi",
        "LKG": "rừng lá kim tự nhiên giàu", "LKB": "rừng lá kim tự nhiên trung bình", "LKN": "rừng lá kim tự nhiên nghèo",
        "LKP": "rừng lá kim tự nhiên phục hồi", "HG1": "rừng hỗn giao gỗ – tre nứa (gỗ là chính)",
        "HG2": "rừng hỗn giao tre nứa – gỗ (tre nứa là chính)", "TNK": "rừng tre nứa", "LOO": "rừng lồ ô",
        "RTG": "rừng trồng gỗ", "RTK": "rừng trồng khác (cây đa mục đích: cao su, điều, đặc sản…)",
        "DTR": "đất đã trồng rừng nhưng chưa thành rừng", "DT1": "đất trống có cây gỗ tái sinh",
        "DT2": "đất trống không có cây gỗ tái sinh", "DNN": "đất nông nghiệp (trong phạm vi lô lâm nghiệp)",
        "DK": "đất khác", "MN": "mặt nước"}
MALR3 = {1: "đặc dụng", 2: "phòng hộ", 3: "sản xuất", 0: "ngoài quy hoạch lâm nghiệp"}

CRIT = ["F1_area_over_0_5ha", "F2_trees_over_5m", "F3_canopy_over_10pct", "F4_not_agricultural_use_2020",
        "F5_unstocked_forest_land", "D1_conversion_to_agriculture_after_2020"]
STATUS = ["met", "not_met", "unknown"]
LAND = ["natural_forest", "wood_plantation_forest", "coffee_agroforestry", "other_tree_crop", "rubber",
        "annual_crop_or_bare", "built_or_water", "other_wooded_land", "unclear"]
CHANGE = ["none", "plantation_rotation", "fire_without_conversion", "selective_logging", "natural_disturbance",
          "conversion_to_agriculture", "conversion_to_infrastructure", "agricultural_renovation", "unclear"]
YNU = ["yes", "no", "uncertain"]
VERDICT = ["compliant", "non_compliant", "uncertain"]
SCHEMA = {"type": "object", "properties": {
    "criteria": {"type": "object", "properties": {c: {"type": "object", "properties": {
        "status": {"enum": STATUS}, "evidence": {"type": "array", "items": {"type": "string"}}},
        "required": ["status", "evidence"]} for c in CRIT}, "required": CRIT},
    "land_use_2020": {"enum": LAND}, "forest_2020": {"enum": YNU}, "change_after_2020": {"enum": CHANGE},
    "deforestation_after_2020": {"enum": YNU}, "verdict": {"enum": VERDICT},
    "explanation_vi": {"type": "string"}},
    "required": ["criteria", "land_use_2020", "forest_2020", "change_after_2020", "deforestation_after_2020", "verdict",
                 "explanation_vi"]}

LEGAL_SYS = """Bạn là chuyên gia thẩm định tuân thủ Quy định chống mất rừng của EU (EUDR, Regulation (EU) 2023/1115). Nhiệm vụ: áp dụng ĐÚNG định nghĩa pháp lý lên bằng chứng được cung cấp cho một mảng đất ở Tây Nguyên, Việt Nam.

Định nghĩa (nguyên văn, Điều 2):
- (4) 'forest' means land spanning more than 0.5 hectares with trees higher than 5 metres and a canopy cover of more than 10 %, or trees able to reach those thresholds in situ, excluding land that is predominantly under agricultural or urban land use;
- (5) 'agricultural use' means the use of land for the purpose of agriculture, including for agricultural plantations and set-aside agricultural areas, and for rearing livestock;
- (6) 'agricultural plantation' means land with tree stands in agricultural production systems, such as fruit tree plantations, oil palm plantations, olive orchards and agroforestry systems where crops are grown under tree cover; it includes all plantations of relevant commodities other than wood; agricultural plantations are excluded from the definition of 'forest';
- (3) 'deforestation' means the conversion of forest to agricultural use, whether human-induced or not. Mốc cắt: 31/12/2020.
(Suy thoái rừng, khoản 7, chỉ áp dụng cho gỗ: không đánh giá ở đây.)

Các tiêu chí phải trả lời (met / not_met / unknown), mỗi tiêu chí ghi mã bằng chứng đã dùng (E01, E02… hoặc IMG = ảnh):
- F1_area_over_0_5ha: KHỐI RỪNG LIỀN KHỐI chứa đối tượng rộng > 0,5 ha. Một mảng cảnh báo nhỏ nằm trong khu rừng lớn hơn (xem E16 và ảnh) vẫn đạt F1; không dùng diện tích riêng của mảng cảnh báo để loại F1.
- F2_trees_over_5m: cây cao > 5 m tại 2020, hoặc cây sẽ đạt ngưỡng khi trưởng thành.
- F3_canopy_over_10pct: độ che phủ tán > 10 % tại 2020.
- F4_not_agricultural_use_2020: met nếu năm 2020 KHÔNG phải đất chủ yếu dùng cho nông nghiệp/đô thị; not_met nếu là đồn điền nông nghiệp hay đất nông nghiệp.
- F5_unstocked_forest_land: met nếu là đất rừng tạm thời không có cây (sau khai thác, cháy) nhưng vẫn là đất rừng.
- D1_conversion_to_agriculture_after_2020: met nếu sau 31/12/2020 đất rừng chuyển sang dùng cho nông nghiệp.

Các trường hợp dễ sai:
1. Cà phê trồng dưới cây che bóng (muồng, sầu riêng, bơ, mắc ca…) là đồn điền nông nghiệp: KHÔNG phải rừng, dù cây cao và tán dày.
2. Cao su, điều, cây ăn quả là đồn điền nông nghiệp: không phải rừng, kể cả khi bản đồ quốc gia ghi là "rừng trồng".
3. Rừng trồng lấy gỗ (keo, bạch đàn, thông) LÀ rừng. Khai thác rồi trồng lại cây lấy gỗ, hoặc đất rừng tạm trống sau khai thác/cháy mà không đổi mục đích, KHÔNG phải phá rừng.
4. Phá rừng chỉ khi đất là rừng tại 31/12/2020 VÀ sau đó chuyển sang nông nghiệp (cà phê, sầu riêng, tiêu, mì, nương rẫy…). Chuyển sang đường, hồ thủy điện, mỏ thì ghi conversion_to_infrastructure và deforestation_after_2020 = no.
5. Đất đã là nông nghiệp năm 2020 rồi tái canh hay chặt cây che bóng: agricultural_renovation, không phải phá rừng.
6. Bản đồ chỉ là bằng chứng có sai số: bản đồ rừng toàn cầu có thể nhầm cây trồng là rừng; bản đồ quốc gia có thể lỗi thời; cảnh báo mất tán gồm cả khai thác, cháy, rụng lá mùa khô. Cân nhắc nhiều nguồn cùng với ảnh. Khi bằng chứng mâu thuẫn hoặc không đủ, chọn uncertain, không đoán.

Kết luận: verdict = compliant nếu không có phá rừng sau 31/12/2020 (kể cả khi năm 2020 không phải rừng), non_compliant nếu có, uncertain nếu không đủ bằng chứng. Trả lời đúng JSON schema; explanation_vi ngắn gọn (tối đa 80 từ), nêu các bằng chứng quyết định."""

GENERIC_SYS = """Bạn là chuyên gia viễn thám về rừng. Dựa trên bằng chứng được cung cấp cho một mảng đất ở Tây Nguyên, Việt Nam, đánh giá: năm 2020 đây có phải rừng không, sau 31/12/2020 có bị phá rừng không, và kết luận verdict (compliant nếu không bị phá rừng sau 2020, non_compliant nếu bị phá rừng, uncertain nếu không đủ bằng chứng). Với mỗi tiêu chí trong schema ghi met / not_met / unknown và mã bằng chứng đã dùng (E01, E02… hoặc IMG = ảnh). Trả lời đúng JSON schema; explanation_vi ngắn gọn (tối đa 80 từ)."""

# ---- design v2 (10/10/2026): the VLM answers only the two semantic questions (land use at the cut-off, change after
# it), numeric thresholds F1-F3 are computed from the data and art. 2 is applied by code. v1 showed (i) English enum
# values under a Vietnamese prompt distort constrained decoding (the model starts "r(ừng)" and the only allowed value
# starting with "r" is "rubber": 84/140 items), (ii) the model ignores the instruction that F1 is about the contiguous
# forest, not the alert patch. v2vi uses Vietnamese option values, v2en the same prompt with English values.
LAND2 = {"vi": ["rừng_tự_nhiên", "rừng_trồng_lấy_gỗ", "cà_phê_hoặc_nông_lâm_kết_hợp", "cây_lâu_năm_khác",
                "cây_hằng_năm_hoặc_đất_trống", "xây_dựng_hoặc_mặt_nước", "không_rõ"],
         "en": ["natural_forest", "wood_plantation", "coffee_or_agroforestry", "other_perennial_crop", "annual_crop_or_bare",
                "built_or_water", "unclear"]}
CHANGE2 = {"vi": ["không_thay_đổi", "khai_thác_rừng_trồng_rồi_trồng_lại", "cháy_hoặc_thiên_tai_rồi_tái_sinh", "khai_thác_chọn",
                  "chuyển_sang_nông_nghiệp", "chuyển_sang_hạ_tầng", "tái_canh_nông_nghiệp", "không_rõ"],
           "en": ["no_change", "plantation_harvest_replanted", "fire_or_disaster_regrowth", "selective_logging",
                  "conversion_to_agriculture", "conversion_to_infrastructure", "agricultural_renovation", "unclear"]}
CONF2 = {"vi": ["cao", "vừa", "thấp"], "en": ["high", "medium", "low"]}
KEYS2 = {"vi": ("quan_sát", "sử_dụng_đất_2020", "bằng_chứng_sử_dụng_đất", "thay_đổi_sau_2020", "bằng_chứng_thay_đổi", "độ_chắc_chắn"),
         "en": ("observations", "land_use_2020", "land_use_evidence", "change_after_2020", "change_evidence", "confidence")}
LAND_DESC = ["rừng tự nhiên (thường xanh, khộp, hỗn giao, tre nứa), kể cả rừng nghèo, rừng phục hồi",
             "keo, bạch đàn, thông… trồng để lấy gỗ",
             "cà phê (kể cả dưới cây che bóng), hồ tiêu, vườn có cây che bóng",
             "cao su, điều, sầu riêng, cây ăn quả, mắc ca…", "lúa, ngô, mì, nương rẫy, đất trống",
             "nhà cửa, đường, mặt nước", "không xác định được"]
CHANGE_DESC = ["không mất tán thật (mây, bóng, rụng lá mùa khô rồi xanh lại)",
               "rừng trồng bị chặt theo luân kỳ, sau đó trồng lại hoặc tái sinh cây lấy gỗ",
               "cháy, lở đất, bão; không dọn để trồng cây nông nghiệp", "lỗ tán nhỏ rải rác, phần tán còn lại giữ nguyên",
               "rừng bị dọn sạch rồi đất trống kéo dài, nương rẫy, hoặc xuất hiện hàng cây trồng (cà phê, sầu riêng, cao su, mì…)",
               "đường, hồ thủy điện, mỏ, công trình", "vườn cây nông nghiệp bị chặt để tái canh hoặc đổi cây trồng",
               "không xác định được"]


LAND_DESC_EN = ["natural forest (evergreen, deciduous dipterocarp, mixed, bamboo), including poor or regrowing forest",
                "acacia, eucalyptus, pine... planted for wood", "coffee (also under shade trees), pepper, gardens with shade trees",
                "rubber, cashew, durian, fruit trees, macadamia...", "rice, maize, cassava, swidden fields, bare land",
                "houses, roads, water", "cannot be determined"]
CHANGE_DESC_EN = ["no real canopy loss (clouds, shadows, dry-season leaf fall that greens up again)",
                  "plantation felled in rotation, then replanted or regrowing wood trees",
                  "fire, landslide, storm; not cleared for agricultural crops", "small scattered canopy gaps, the rest of the canopy intact",
                  "forest cleared, then long-lasting bare land, swidden, or rows of crops (coffee, durian, rubber, cassava...)",
                  "road, hydropower reservoir, mine, construction", "agricultural garden cut for renovation or a crop change",
                  "cannot be determined"]


def _order(item_id, n):
    """Option order: fixed, or a per-item permutation (seeded by the item id) to test position bias."""
    if item_id is None:
        return list(range(n))
    rng = np.random.default_rng(int.from_bytes(item_id.encode(), "little") % (2 ** 32))
    return [int(x) for x in rng.permutation(n)]


def schema2(lang: str, item_id: str | None = None) -> dict:
    k = KEYS2[lang]
    lo, co = _order(item_id, len(LAND2[lang])), _order(item_id[::-1] if item_id else None, len(CHANGE2[lang]))
    return {"type": "object", "properties": {
        k[0]: {"type": "string"}, k[1]: {"enum": [LAND2[lang][i] for i in lo]}, k[2]: {"type": "array", "items": {"type": "string"}},
        k[3]: {"enum": [CHANGE2[lang][i] for i in co]}, k[4]: {"type": "array", "items": {"type": "string"}}, k[5]: {"enum": CONF2[lang]}},
        "required": list(k)}


def sys2(lang: str, style: str = "p1", item_id: str | None = None) -> str:
    """p1: the v2 prompt; p2: the same content reworded and reordered (robustness); en: English instructions.
    item_id given -> options listed in a per-item random order (position-bias test)."""
    k = KEYS2[lang]
    lo, co = _order(item_id, len(LAND2[lang])), _order(item_id[::-1] if item_id else None, len(CHANGE2[lang]))
    ld, cd = (LAND_DESC_EN, CHANGE_DESC_EN) if style == "en" else (LAND_DESC, CHANGE_DESC)
    land = "\n".join(f"   - {LAND2[lang][i]}: {ld[i]}" for i in lo)
    chg = "\n".join(f"   - {CHANGE2[lang][i]}: {cd[i]}" for i in co)
    conf = " / ".join(CONF2[lang])
    if style == "p2":
        return (f"Nhiệm vụ của bạn: đọc ảnh vệ tinh và thẻ bằng chứng của một mảng đất ở Tây Nguyên rồi cho biết HAI điều về thực địa. "
                f"Hệ thống sẽ tự áp dụng EUDR (Quy định chống mất rừng của EU) sau đó; bạn chỉ mô tả thực tế, không kết luận về luật.\n\n"
                f"Lưu ý khi đọc: bản đồ có sai số (bản đồ rừng toàn cầu dễ gọi cây trồng là rừng, bản đồ quốc gia có thể cũ, cảnh báo mất tán "
                f"có cả khai thác, cháy, rụng lá), nên ưu tiên ảnh theo thời gian. Mất tán thật mà sau đó đất trống kéo dài, hoặc mọc lên các "
                f"hàng cây đều, là chuyển đổi; NDVI hồi phục về mức cũ thường là không chuyển đổi. Rừng khộp, tre nứa rụng lá theo mùa: so "
                f"sánh cùng mùa.\n\n"
                f"Câu A ({k[1]}): trước 31/12/2020 (với cảnh báo: trước khi mất tán), mảng đất đang là gì? Chọn một:\n{land}\n"
                f"Câu B ({k[3]}): từ sau 31/12/2020 đến nay, mảng đất đã thay đổi thế nào? Chọn một:\n{chg}\n\n"
                f"Định dạng: viết {k[0]} trước (tối đa 60 từ, tả ảnh trước/sau và NDVI), rồi trả lời câu A và B; ghi mã bằng chứng "
                f"(E01…, IMG) vào {k[2]} và {k[4]}; {k[5]} là {conf}. Trả lời đúng JSON schema.")
    if style == "en":
        return (f"You are a satellite-image interpreter supporting due diligence under the EU Deforestation Regulation (EUDR) in "
                f"Vietnam's Central Highlands. Answer TWO questions about the facts on the ground; the system applies the law (forest "
                f"definition, area, height and canopy thresholds) afterwards, so do not draw legal conclusions yourself. The evidence "
                f"card is in Vietnamese.\n\n"
                f"1) {k[1]}: what was this land used for on 31 Dec 2020 (for an alert patch: before the canopy loss)?\n{land}\n"
                f"2) {k[3]}: what happened to the land after 31 Dec 2020?\n{chg}\n\n"
                f"Hints: compare the images before and after and check whether NDVI recovers to its earlier level. Bare land lasting "
                f"many months, or regular rows of trees appearing after clearing, indicate conversion. Deciduous dipterocarp forest and "
                f"bamboo shed leaves seasonally: compare the same season. Maps are secondary evidence with known errors: global maps "
                f"often label tree crops as forest; the national map may be outdated; loss alerts include harvest, fire and leaf fall.\n\n"
                f"First write {k[0]} (at most 60 words): what the before/after images and the NDVI series show. Then choose the answers; "
                f"put the evidence ids used (E01, E02… or IMG = the image) in {k[2]} and {k[4]}; {k[5]}: {conf}. Answer with the JSON schema.")
    return (f"Bạn là chuyên gia giải đoán ảnh vệ tinh, hỗ trợ thẩm định theo Quy định chống mất rừng của EU (EUDR) ở Tây Nguyên, Việt Nam. "
            f"Hãy trả lời HAI câu hỏi về thực tế mặt đất; phần áp dụng luật (định nghĩa rừng, ngưỡng diện tích, chiều cao, độ che phủ) do "
            f"hệ thống thực hiện, bạn không cần tự kết luận về luật.\n\n"
            f"1) {k[1]}: đất này được dùng vào việc gì tại 31/12/2020 (với mảng cảnh báo: trước khi mất tán)?\n{land}\n"
            f"2) {k[3]}: điều gì xảy ra với đất sau 31/12/2020?\n{chg}\n\n"
            f"Gợi ý: so sánh ảnh trước và sau, xem NDVI có phục hồi về mức cũ không. Đất trống kéo dài nhiều tháng, hoặc hàng cây đều "
            f"xuất hiện sau khi dọn, là dấu hiệu chuyển đổi. Rừng khộp và tre nứa rụng lá theo mùa: so ảnh cùng mùa. Bản đồ chỉ là bằng "
            f"chứng phụ và có thể sai: bản đồ toàn cầu hay nhầm cây trồng là rừng; bản đồ quốc gia có thể lỗi thời; cảnh báo mất tán gồm "
            f"cả khai thác, cháy, rụng lá.\n\n"
            f"Trước hết viết {k[0]} (tối đa 60 từ): mô tả ngắn những gì thấy trên ảnh trước và sau, và diễn biến NDVI. Sau đó chọn đáp "
            f"án; ghi mã bằng chứng đã dùng (E01, E02… hoặc IMG = ảnh) vào {k[2]} và {k[4]}; {k[5]}: {conf}. Trả lời đúng JSON schema.")


VARIANTS = {"full": dict(sys="legal", card="v4", image=True),
            "noimg": dict(sys="legal", card="v4", image=False),
            "imgonly": dict(sys="legal", card="none", image=True),
            "nodef": dict(sys="generic", card="v4", image=True),
            "v1swap": dict(sys="legal", card="v1", image=True),
            "sc": dict(sys="legal", card="v4", image=True, sampling={"n": 5, "temperature": 0.7, "top_p": 0.95, "seed": 7}),
            "v2vi": dict(sys="v2", lang="vi", card="v4", image=True),
            "v2en": dict(sys="v2", lang="en", card="v4", image=True),
            "v2vi_sc": dict(sys="v2", lang="vi", card="v4", image=True,
                            sampling={"n": 5, "temperature": 0.7, "top_p": 0.95, "seed": 7}),
            # v3 (10/10/2026): ablations and robustness of the v2 design
            "v2vi_noimg": dict(sys="v2", lang="vi", card="v4", image=False),
            "v2vi_imgonly": dict(sys="v2", lang="vi", card="none", image=True),
            "v2vi_v1card": dict(sys="v2", lang="vi", card="v1", image=True),
            "v2vi_p2": dict(sys="v2", lang="vi", style="p2", card="v4", image=True),
            "v2vi_shuf": dict(sys="v2", lang="vi", shuffle=True, card="v4", image=True),
            "v2en_full": dict(sys="v2", lang="en", style="en", card="v4", image=True)}

# per-model vLLM settings (run_batch.py "llm_kw"); None removes a default
# InternVL: vLLM 0.31 also hands mm_processor_kwargs to InternVL's video processor, which rejects max_dynamic_patch
# (and Qwen's max_pixels), so the model keeps its own tiling (up to 12 tiles + thumbnail)
MODEL_KW = {"InternVL3_5-8B": {"trust_remote_code": True, "mm_processor_kwargs": None}}
MODEL_TAG = {"Qwen3.5-9B": "", "InternVL3_5-8B": "@ivl8b", "Qwen3.5-35B-A3B-GPTQ-Int4": "@q35b"}

IMG_ALERT = ("Ảnh tham chiếu (IMG): hàng trên là ảnh Sentinel-2 màu thật, hàng giữa màu giả hồng ngoại (thực vật màu đỏ), "
             "theo các mốc so với cảnh báo đầu tiên: −12..−7 tháng, −4..−1 tháng, 0..+2 tháng, +3..+6, +10..+14, +16..+20 tháng; "
             "dưới cùng là NDVI theo thời gian (vạch đỏ = cảnh báo đầu tiên). Viền vàng = ranh mảng cảnh báo.")
IMG_PLOT = ("Ảnh tham chiếu (IMG): ảnh Sentinel-2 mùa khô 2017, 2018, 2019, 2020, giai đoạn mốc cắt (11/2020–04/2021), 2022, 2024, "
            "2026 (hàng trên màu thật, hàng giữa màu giả hồng ngoại, thực vật màu đỏ); dưới cùng NDVI theo thời gian "
            "(vạch đỏ = mốc cắt 31/12/2020). Khung vàng = ranh thửa.")
Q_ALERT = ("Đối tượng: một mảng có cảnh báo mất tán cây (GFW) xuất hiện sau mốc cắt. Xác định theo EUDR: tại 31/12/2020 mảng có "
           "phải 'forest' không; thay đổi sau 2020 là gì và có phải 'deforestation' không; kết luận verdict.")
Q_PLOT = ("Đối tượng: một thửa giả định trong vùng trồng cà phê (nguồn cung tiềm năng). Xác định theo EUDR: tại 31/12/2020 thửa có "
          "phải 'forest' không; sau 2020 có 'deforestation' không; kết luận verdict.")


# ------------------------------------------------------------------------------------------------ features
def items() -> gpd.GeoDataFrame:
    a = gpd.read_parquet(D / "sample_alerts.parquet")[["item_id", "kind", "area_ha", "geometry"]]
    p = gpd.read_parquet(D / "sample_plots.parquet")[["item_id", "kind", "area_ha", "geometry"]]
    g = gpd.GeoDataFrame(pd.concat([a, p], ignore_index=True), geometry="geometry", crs=4326)
    g["geometry"] = shapely.make_valid(g.geometry.values)
    if SET and (D / "triage.parquet").exists():  # s1: the items whose evidence is done
        g = g[g.item_id.isin(pd.read_parquet(D / "triage.parquet", columns=["item_id"]).item_id)]
    return g


def gee_image(ee):
    b = []
    for v, coll in GFC_VERSIONS.items():
        im = ee.ImageCollection(f"JRC/GFC2020/{v}").mosaic() if coll else ee.Image(f"JRC/GFC2020/{v}")
        b.append(im.select("Map").unmask(0).gt(0).rename(f"gfc{v}"))
    gft = ee.Image("JRC/GFC2020_subtypes/V1").select("Map").unmask(0)
    b += [gft.eq(1).rename("gft_natregen"), gft.eq(10).rename("gft_primary"), gft.eq(20).rename("gft_planted")]
    tmf = ee.ImageCollection(TMF25).mosaic()
    for y in (2020, 2025):
        t = tmf.select(f"Dec{y}").unmask(0)
        b += [t.eq(c).rename(f"tmf{y % 100}_{n}") for c, n in ((1, "undisturbed"), (2, "degraded"), (3, "deforested"), (4, "regrowth"))]
    b.append(ee.ImageCollection(NATFOR).mosaic().select("B0").unmask(0).divide(250).rename("natforest_p"))
    ty = ee.ImageCollection(FORTY).mosaic()
    b += [ty.select(s).unmask(0).divide(250).rename(n) for s, n in
          (("PrimaryForest", "ty_primary"), ("NaturallyRegeneratingForest", "ty_natregen"),
           ("PlantationForest", "ty_plantation"), ("TreeCropsAndAgroforestry", "ty_treecrop"))]
    b.append(ee.Image(PERSIST).select("value").unmask(0).rename("persistence"))
    rub = ee.ImageCollection(RUBBER)
    b += [rub.filterDate(f"{y}-01-01", f"{y + 1}-01-01").mosaic().select("probability").unmask(0).rename(f"rubber_{y}")
          for y in (2020, 2024)]
    wc = ee.ImageCollection("ESA/WorldCover/v100").first().select("Map")
    b += [wc.eq(10).rename("wc20_tree"), wc.eq(40).rename("wc20_crop")]
    eth = ee.Image(ETH_H).select("b1")
    b.append(eth.updateMask(eth.lte(100)).rename("h_eth"))
    gl = ee.ImageCollection(GLAD_H).mosaic().select("b1")
    b.append(gl.updateMask(gl.lte(60)).rename("h_glad"))
    return ee.Image.cat(b)


def official_cover(g: gpd.GeoDataFrame) -> pd.DataFrame:
    """Area shares of official 2020 classes inside each item, and the attributes of the lot with the largest overlap."""
    lots = gpd.read_parquet(PROCESSED / "vnforest" / "dbr2020_tn.parquet", columns=["cls", "ldlr", "sldlr", "namtr", "malr3", "geometry"])
    gi, li = lots.sindex.query(g.geometry.values, predicate="intersects")
    a, b = g.iloc[gi].reset_index(drop=True), lots.iloc[li].reset_index(drop=True)
    inter = shapely.intersection(a.geometry.values, shapely.make_valid(b.geometry.values))
    pairs = b.drop(columns="geometry").assign(item_id=a.item_id.values,
                                              ha=gpd.GeoSeries(inter, crs=4326).to_crs(32648).area.values / 1e4)
    tot = g.set_index("item_id").to_crs(32648).area / 1e4
    sh = pairs.pivot_table(index="item_id", columns="cls", values="ha", aggfunc="sum").reindex(
        columns=["natural", "plantation", "nonforest"]).fillna(0)
    sh = sh.reindex(tot.index).fillna(0).div(tot, axis=0).clip(0, 1).add_prefix("off_")
    sh["off_outside"] = (1 - sh.sum(axis=1)).clip(0, 1)
    dom = pairs.sort_values("ha").drop_duplicates("item_id", keep="last").set_index("item_id")
    dom = dom[["cls", "ldlr", "sldlr", "namtr", "malr3"]].add_prefix("lot_")
    return sh.join(dom).reset_index()


def features():
    g = items()
    ee = ee_init()
    parts = []
    for k in range(0, len(g), 200):  # small Earth Engine requests; dot0 (140 items) is a single chunk
        for tries in range(8):  # restricted Earth Engine projects answer "too many requests" now and then
            try:
                parts.append(_features_chunk(ee, g.iloc[k:k + 200]))
                break
            except Exception as e:  # noqa: BLE001
                if tries == 7:
                    raise
                print("features retry", tries, repr(e)[:120], flush=True)
                time.sleep(min(600, 60 * 2 ** tries))
        print("features", k + len(parts[-1]), "/", len(g), flush=True)
    feat = pd.concat(parts, ignore_index=True)
    tri = pd.read_parquet(D / "triage.parquet")
    keep = ["item_id", "kind", "area_ha", "first_date", "coffee_2020", "coffee_2024", "hl_pre2021", "hl_post2020",
            *[f"hl_{y}" for y in range(2021, 2026)], "tc2000", "tmf_natural", "tmf_plantation", "wdpa", "gfw_post2020",
            "gfw_first_date", "dw_pre_trees", "dw_pre_crops", "dw_post_trees", "dw_post_crops", "dw_post_bare",
            "dw_post_built", "dw_post_water", "dw_trees_2019", "dw_crops_2019", "dw_trees_2025", "dw_crops_2025", "dw_bare_2025",
            "ndvi_pre", "ndvi_post", "ndvi_rec3m", "ndvi_rec6m", "ndvi_rec12m", "vh_pre", "vh_post", "modis_burned",
            "firms_px", *[f"ndvi_dry_{y}" for y in range(2019, 2027)], "elev", "slope", "category"]
    f = tri[[c for c in keep if c in tri]].merge(feat, on="item_id", how="left")
    f.to_parquet(D / "legal_features.parquet")
    print(f.shape)
    print(f.describe().T[["mean", "min", "max"]].round(2).to_string())


def _features_chunk(ee, g: gpd.GeoDataFrame) -> pd.DataFrame:
    fc = ee.FeatureCollection(g[["item_id", "geometry"]].__geo_interface__)
    m = pd.DataFrame(to_gdf(gee_image(ee).reduceRegions(fc, ee.Reducer.mean(), scale=10, tileScale=8)).drop(columns="geometry"))
    meta = ee.ImageCollection(META_H).mosaic().select(0).rename("h_meta")
    red = ee.Reducer.mean().combine(ee.Reducer.percentile([90]), "", True)
    h = pd.DataFrame(to_gdf(meta.reduceRegions(fc, red, scale=2, tileScale=8)).drop(columns="geometry"))
    h = h.rename(columns={"mean": "h_meta_mean", "p90": "h_meta_p90", "h_meta_mean": "h_meta_mean", "h_meta_p90": "h_meta_p90"})
    off = official_cover(g)
    # surroundings: 100 m buffer (F1 is about the contiguous forest the object belongs to, not the alert patch size)
    gb = g.copy()
    gb["geometry"] = g.to_crs(32648).buffer(100).to_crs(4326).values
    fb = ee.FeatureCollection(gb[["item_id", "geometry"]].__geo_interface__)
    bimg = ee.Image.cat([gee_image(ee).select(["gfcV1", "gfcV4", "natforest_p"]),
                         ee.Image("UMD/hansen/global_forest_change_2025_v1_13").select("treecover2000").rename("tc2000")])
    bm = pd.DataFrame(to_gdf(bimg.reduceRegions(fb, ee.Reducer.mean(), scale=10, tileScale=8)).drop(columns="geometry"))
    bo = official_cover(gb)
    bm = bm.merge(bo.assign(off_forest=bo.off_natural + bo.off_plantation)[["item_id", "off_forest"]], on="item_id", how="left")
    bm = bm.rename(columns={c: f"buf_{c}" for c in bm.columns if c != "item_id"})
    return m.merge(h[["item_id", "h_meta_mean", "h_meta_p90"]], on="item_id", how="left").merge(off, on="item_id", how="left").merge(
        bm, on="item_id", how="left")


# ------------------------------------------------------------------------------------------------ cards
def _p(x) -> str:
    return "không có dữ liệu" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * float(x):.0f}%"


def _n(x, d=2) -> str:
    return "không có dữ liệu" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.{d}f}"


def card(r: pd.Series, gfc_col: str = "gfcV4") -> list[tuple[str, str]]:
    """(id, text) lines; ids are stable per item kind."""
    E = []
    if r.kind == "alert":
        E.append(("E01", f"Đối tượng: mảng cảnh báo mất tán, diện tích {_n(r.area_ha)} ha, cảnh báo đầu tiên {str(r.first_date)[:10]}."))
    else:
        E.append(("E01", f"Đối tượng: thửa giả định, diện tích {_n(r.area_ha)} ha."))
    lot = "nằm ngoài các lô lâm nghiệp" if pd.isna(r.get("lot_ldlr")) else (
        f"lô chiếm nhiều diện tích nhất: mã {r.lot_ldlr} ({LDLR.get(str(r.lot_ldlr), 'không rõ')})"
        + (f", loài cây: {r.lot_sldlr}" if isinstance(r.get("lot_sldlr"), str) and r.lot_sldlr.strip() else "")
        + (f", năm trồng {int(r.lot_namtr)}" if pd.notna(r.get("lot_namtr")) and float(r.lot_namtr) > 1900 else "")
        + (f", quy hoạch {MALR3.get(int(r.lot_malr3), 'không rõ')}" if pd.notna(r.get("lot_malr3")) else ""))
    E.append(("E02", f"Bản đồ hiện trạng rừng 2020 chính thức của Việt Nam (lô diễn biến rừng 2020): rừng tự nhiên {_p(r.off_natural)}, "
                     f"rừng trồng {_p(r.off_plantation)}, chưa có rừng {_p(r.off_nonforest)}, ngoài lô {_p(r.off_outside)} diện tích; {lot}."))
    E.append(("E03", f"Bản đồ rừng 2020 của EU (JRC GFC2020): {_p(r[gfc_col])} diện tích là rừng."))
    E.append(("E04", f"Bản đồ loại rừng 2020 của EU (JRC GFT2020, làm từ GFC2020 bản 3 nên có thể khác E03): tái sinh tự nhiên "
                     f"{_p(r.gft_natregen)}, nguyên sinh {_p(r.gft_primary)}, rừng trồng {_p(r.gft_planted)}."))
    E.append(("E05", f"JRC TMF (rừng ẩm nhiệt đới, Landsat 30 m): tháng 12/2020 nguyên vẹn {_p(r.tmf20_undisturbed)}, suy thoái {_p(r.tmf20_degraded)}, "
                     f"tái sinh {_p(r.tmf20_regrowth)}, đã mất {_p(r.tmf20_deforested)}; tháng 12/2025 nguyên vẹn {_p(r.tmf25_undisturbed)}, "
                     f"suy thoái {_p(r.tmf25_degraded)}, tái sinh {_p(r.tmf25_regrowth)}, đã mất {_p(r.tmf25_deforested)}; "
                     f"lớp con 'đồn điền cây / rừng trồng' của TMF {_p(r.tmf_plantation)}."))
    E.append(("E06", f"Hansen/UMD: độ che phủ cây năm 2000 trung bình {_n(r.tc2000, 0)}%; mất tán trước 2021 {_p(r.hl_pre2021)}; "
                     f"mất tán 2021 {_p(r.hl_2021)}, 2022 {_p(r.hl_2022)}, 2023 {_p(r.hl_2023)}, 2024 {_p(r.hl_2024)}, 2025 {_p(r.hl_2025)} diện tích."))
    E.append(("E07", f"Chiều cao tán: Meta/WRI 1 m (ảnh 2009–2020 tùy nơi) trung bình {_n(r.h_meta_mean, 1)} m, phân vị 90 {_n(r.h_meta_p90, 1)} m; "
                     f"ETH 2020 (10 m) {_n(r.h_eth, 1)} m; GLAD 2019 (30 m) {_n(r.h_glad, 1)} m."))
    E.append(("E08", f"ESA WorldCover 2020: cây gỗ {_p(r.wc20_tree)}, đất trồng trọt {_p(r.wc20_crop)} diện tích."))
    E.append(("E09", f"Mô hình cây trồng FDP (Google/FDP, xác suất 0–1): cà phê 2020 {_n(r.coffee_2020)}, cà phê 2024 {_n(r.coffee_2024)} "
                     f"(xác suất năm 2020 của mô hình này thường thấp hơn thực tế); cao su 2020 {_n(r.rubber_2020)}, cao su 2024 {_n(r.rubber_2024)}."))
    E.append(("E10", f"Nature Trace (Google) 2020: xác suất rừng tự nhiên {_n(r.natforest_p)}; phân loại kiểu (0–1): nguyên sinh {_n(r.ty_primary)}, "
                     f"tái sinh tự nhiên {_n(r.ty_natregen)}, rừng trồng {_n(r.ty_plantation)}, cây trồng/nông lâm kết hợp {_n(r.ty_treecrop)}."))
    E.append(("E11", f"FDP Forest Persistence 2020 (0–1, cao = rừng tồn tại lâu dài): {_n(r.persistence)}."))
    if r.kind == "alert":
        E.append(("E12", f"Dynamic World quanh cảnh báo: trước cảnh báo cây {_p(r.dw_pre_trees)}, cây trồng {_p(r.dw_pre_crops)}; sau cảnh báo "
                         f"cây {_p(r.dw_post_trees)}, cây trồng {_p(r.dw_post_crops)}, đất trống {_p(r.dw_post_bare)}, xây dựng {_p(r.dw_post_built)}, "
                         f"mặt nước {_p(r.dw_post_water)}."))
        E.append(("E13", f"Phổ quanh cảnh báo: NDVI trước {_n(r.ndvi_pre)}, ngay sau {_n(r.ndvi_post)}, sau 3 tháng {_n(r.ndvi_rec3m)}, "
                         f"6 tháng {_n(r.ndvi_rec6m)}, 12 tháng {_n(r.ndvi_rec12m)}; radar Sentinel-1 VH trước {_n(r.vh_pre, 1)} dB, sau {_n(r.vh_post, 1)} dB."))
    else:
        E.append(("E12", f"Dynamic World: năm 2019 cây {_p(r.dw_trees_2019)}, cây trồng {_p(r.dw_crops_2019)}; năm 2025 cây {_p(r.dw_trees_2025)}, "
                         f"cây trồng {_p(r.dw_crops_2025)}, đất trống {_p(r.dw_bare_2025)}."))
        dry = ", ".join(f"{y}: {_n(r.get(f'ndvi_dry_{y}'))}" for y in range(2019, 2027))
        E.append(("E13", f"NDVI mùa khô theo năm: {dry}."))
    E.append(("E14", f"Cháy: số điểm nóng FIRMS (VIIRS/MODIS) quanh thời điểm {'cảnh báo' if r.kind == 'alert' else '2021–2025'}: {_n(r.firms_px, 0)}."))
    E.append(("E15", f"Khu bảo tồn (WDPA): {'có' if (r.wdpa or 0) > 0.5 else 'không'}; cảnh báo GFW sau 2020 phủ {_p(r.gfw_post2020)} diện tích."))
    E.append(("E16", f"Vùng xung quanh (đối tượng + vành 100 m): bản đồ rừng 2020 của EU {_p(r['buf_' + gfc_col])} là rừng, bản đồ QG 2020 "
                     f"{_p(r.buf_off_forest)} là rừng, che phủ cây Hansen 2000 trung bình {_n(r.buf_tc2000, 0)}%, xác suất rừng tự nhiên "
                     f"(Nature Trace) {_n(r.buf_natforest_p)}."))
    return E


def cards():
    f = pd.read_parquet(D / "legal_features.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "cards.jsonl", "w", encoding="utf-8") as fh:
        for _, r in f.iterrows():
            for name, col in (("v4", "gfcV4"), ("v1", "gfcV1")):
                lines = card(r, col)
                fh.write(json.dumps({"item_id": r.item_id, "kind": r.kind, "card": name, "ids": [i for i, _ in lines],
                                     "text": "\n".join(f"{i} | {t}" for i, t in lines)}, ensure_ascii=False) + "\n")
    ex = [json.loads(x) for x in open(OUT / "cards.jsonl", encoding="utf-8")][:4:2] + \
         [json.loads(x) for x in open(OUT / "cards.jsonl", encoding="utf-8") if '"kind": "plot"' in x][:1]
    (OUT / "cards_examples.md").write_text("\n\n".join(f"### {c['item_id']} ({c['kind']}, {c['card']})\n\n```\n{c['text']}\n```" for c in ex),
                                          encoding="utf-8")
    print("cards:", len(f), "items x 2 (V4 / V1 value) ->", OUT / "cards.jsonl")


# ------------------------------------------------------------------------------------------------ run on the lab GPU
def build_payload(variants: list[str], model: str = MODEL, only: set[str] | None = None) -> dict:
    cs = {(c["item_id"], c["card"]): c for c in map(json.loads, open(OUT / "cards.jsonl", encoding="utf-8"))}
    ids = sorted({k[0] for k in cs if only is None or k[0] in only})
    images = {i: "data:image/png;base64," + base64.b64encode((SHEETS / f"{i}.png").read_bytes()).decode() for i in ids}
    out = []
    for v in variants:
        cfg = VARIANTS[v]
        for i in ids:
            c = cs[(i, cfg["card"] if cfg["card"] != "none" else "v4")]
            q = Q_ALERT if c["kind"] == "alert" else Q_PLOT
            if cfg["sys"] == "v2":
                q = ("Đối tượng: một mảng có cảnh báo mất tán cây (GFW) xuất hiện sau mốc cắt." if c["kind"] == "alert" else
                     "Đối tượng: một thửa giả định trong vùng trồng cà phê.") + " Trả lời hai câu hỏi về sử dụng đất 2020 và thay đổi sau 2020."
            if cfg["card"] == "none":  # image only: keep just E01 (what the object is)
                body = f"{q}\n\nTHẺ BẰNG CHỨNG:\n{c['text'].splitlines()[0]}"
            else:
                body = f"{q}\n\nTHẺ BẰNG CHỨNG:\n{c['text']}"
            content = []
            if cfg["image"]:
                content.append({"type": "image_url", "image_url": {"url": f"ref:{i}"}})
                body = (IMG_ALERT if c["kind"] == "alert" else IMG_PLOT) + "\n\n" + body
            content.append({"type": "text", "text": body})
            sid = i if cfg.get("shuffle") else None
            sysmsg = sys2(cfg["lang"], cfg.get("style", "p1"), sid) if cfg["sys"] == "v2" else (LEGAL_SYS if cfg["sys"] == "legal" else GENERIC_SYS)
            it = {"id": f"{v}|{i}", "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": content}]}
            if cfg["sys"] == "v2":
                it["schema"], it["max_tokens"] = schema2(cfg["lang"], sid), 900
            if "sampling" in cfg:
                it["sampling"] = cfg["sampling"]
            out.append(it)
    used = {i for it in out for p in it["messages"][1]["content"] if p.get("type") == "image_url" for i in [p["image_url"]["url"][4:]]}
    pl = {"items": out, "images": {k: v for k, v in images.items() if k in used}, "schema": SCHEMA, "max_tokens": 1500}
    if model in MODEL_KW:
        pl["llm_kw"] = MODEL_KW[model]
    return pl


def run(variants: list[str], gpu: int, gpu_mem: float, model: str, tag: str = "v1", chunk: int = 0):
    """chunk > 0: the items go in groups of `chunk` (one GPU-server batch each, raw_<model>_<tag>_cNN.jsonl); finished
    groups are skipped, so a cut run resumes."""
    if chunk:
        ids = sorted({json.loads(x)["item_id"] for x in open(OUT / "cards.jsonl", encoding="utf-8")})
        for n, k in enumerate(range(0, len(ids), chunk)):
            if not (OUT / f"raw_{model}_{tag}_c{n:02d}.jsonl").exists():
                _run(variants, gpu, gpu_mem, model, f"{tag}_c{n:02d}", set(ids[k:k + chunk]))
        return
    _run(variants, gpu, gpu_mem, model, tag, None)


def _run(variants: list[str], gpu: int, gpu_mem: float, model: str, tag: str, only: set[str] | None):
    from ..extract.remote_lab import Lab, free_gpu, run_payload
    payload = build_payload(variants, model, only)
    print(tag, "items:", len(payload["items"]), "| images:", len(payload["images"]), flush=True)
    lab = Lab()
    try:
        g = free_gpu(lab, prefer=(gpu,), limit_mib=1500)
        if g is None:
            raise SystemExit(f"GPU {gpu} is busy: not starting")
        rows, summ = run_payload(lab, payload, "legal_vlm", model, g, gpu_mem=gpu_mem, max_model_len=12288)
        left = lab.retry(lab.data_files, "check data dir")
    finally:
        lab.close()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"raw_{model}_{tag}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / f"raw_{model}_{tag}.meta.json").write_text(json.dumps({"summary": summ, "variants": variants, "gpu": g, "gpu_mem": gpu_mem,
                                                            "server_data_dir_after": left}, indent=1), encoding="utf-8")
    print("rows:", len(rows), "| summary:", summ, "| server data dir after run:", left or "empty")


# ------------------------------------------------------------------------------------------------ evaluation
def silver_truth() -> pd.DataFrame:
    """EUDR targets derived from the silver labels. Alerts: C1 = conversion to agriculture -> deforestation; C2 rotation,
    C3 selective logging, C4 infrastructure, C5 fire, C6 natural, C0 / no loss -> not deforestation (art. 2(3): only
    conversion to AGRICULTURAL use counts). Plots: natural forest / tree plantation at the cut-off -> forest; coffee or
    orchard -> not forest; mixed -> unknown; deforestation only if a forest plot was cleared."""
    s = pd.read_csv(SILVER)
    rows = []
    for r in s.itertuples():
        y1 = y2 = cause = None
        if r.kind == "alert":
            if r.loss == "no" or r.driver == "C0":
                y2, cause = 0, "none"
            elif r.driver in ("C1", "C2", "C3", "C4", "C5", "C6"):
                y2 = int(r.driver == "C1")
                cause = {"C1": "conversion_to_agriculture", "C2": "plantation_rotation", "C3": "selective_logging",
                         "C4": "conversion_to_infrastructure", "C5": "fire_without_conversion", "C6": "natural_disturbance"}[r.driver]
                y1 = 1 if r.driver in ("C1", "C2", "C3") else None
        else:
            y1 = {"natural_forest": 1, "tree_plantation": 1, "coffee_or_orchard": 0}.get(r.cutoff)
            if r.cleared == "no":
                y2 = 0
            elif r.cleared == "yes":
                y2 = 1 if y1 == 1 else (0 if y1 == 0 else None)
        rows.append({"item_id": r.item_id, "kind": r.kind, "y1": y1, "y2": y2, "cause": cause, "silver_conf": r.conf})
    return pd.DataFrame(rows)


def parse(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", raw or "", re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            return None


AGRI_LAND = {"coffee_agroforestry", "other_tree_crop", "rubber", "annual_crop_or_bare", "built_or_water", "other_wooded_land"}
FOREST_LAND = {"natural_forest", "wood_plantation_forest"}
NO_CONV = {"none", "plantation_rotation", "fire_without_conversion", "selective_logging", "natural_disturbance",
           "conversion_to_infrastructure", "agricultural_renovation"}


def legal_logic(j: dict) -> dict:
    """Art. 2 applied by code to the facts the VLM extracted (criteria statuses, land use 2020, change type), instead of
    trusting the VLM's own conclusion fields (the 9B model contradicts itself, e.g. 'conversion_to_agriculture' with
    deforestation = no)."""
    c = {k: (j.get("criteria", {}).get(k) or {}).get("status") for k in CRIT}
    land = j.get("land_use_2020")
    core = [c["F1_area_over_0_5ha"], c["F2_trees_over_5m"], c["F3_canopy_over_10pct"]]
    if all(x == "met" for x in core) or c["F5_unstocked_forest_land"] == "met":
        structural = "yes"
    elif any(x == "not_met" for x in core):
        structural = "no"
    else:
        structural = "unknown"
    if c["F4_not_agricultural_use_2020"] == "not_met" or land in AGRI_LAND or structural == "no":
        forest = "no"
    elif structural == "yes" and c["F4_not_agricultural_use_2020"] == "met" and land in FOREST_LAND | {"unclear"}:
        forest = "yes"
    else:
        forest = "uncertain"
    conv = (c["D1_conversion_to_agriculture_after_2020"] == "met") - (c["D1_conversion_to_agriculture_after_2020"] == "not_met") \
        + (j.get("change_after_2020") == "conversion_to_agriculture") - (j.get("change_after_2020") in NO_CONV)
    if forest == "no":
        defor = "no"
    elif forest == "yes":
        defor = "yes" if conv > 0 else ("no" if conv < 0 else "uncertain")
    else:
        defor = "no" if conv < 0 else "uncertain"
    return {"forest_2020": forest, "deforestation_after_2020": defor,
            "verdict": {"no": "compliant", "yes": "non_compliant"}.get(defor, "uncertain"),
            "change_after_2020": j.get("change_after_2020"), "land_use_2020": land}


_CHG = ["none", "plantation_rotation", "fire_without_conversion", "selective_logging", "conversion_to_agriculture",
        "conversion_to_infrastructure", "agricultural_renovation", "unclear"]
_LND = ["natural_forest", "wood_plantation_forest", "coffee_agroforestry", "other_tree_crop", "annual_crop_or_bare",
        "built_or_water", "unclear"]
CHANGE_CANON = {**dict(zip(CHANGE2["vi"], _CHG)), **dict(zip(CHANGE2["en"], _CHG))}
LAND_CANON = {**dict(zip(LAND2["vi"], _LND)), **dict(zip(LAND2["en"], _LND))}


def numeric_criteria(r) -> tuple[bool, bool, bool]:
    """F1-F3 from the data (design v2): contiguous forest > 0.5 ha (object or its 100 m surroundings), trees > 5 m
    (any canopy-height map), canopy > 10 %."""
    z = lambda x: 0.0 if pd.isna(x) else float(x)  # noqa: E731
    big = z(r.area_ha) >= 0.5 or max(z(r.buf_gfcV4), z(r.buf_off_forest), z(r.buf_tc2000) / 100) >= 0.5
    tall = max(z(r.h_meta_p90), z(r.h_eth), z(r.h_glad)) >= 5
    canopy = max(z(r.wc20_tree), z(r.tc2000) / 100, z(r.gfcV4), z(r.off_natural) + z(r.off_plantation)) >= 0.1
    return big, tall, canopy


def legal_v2(j: dict, lang: str, crit: tuple[bool, bool, bool]) -> dict:
    """Art. 2 by code on the two facts the VLM gives (land use at the cut-off, change after it) plus F1-F3 from data."""
    k = KEYS2[lang]
    land = LAND_CANON.get(j.get(k[1]), "unclear")
    change = CHANGE_CANON.get(j.get(k[3]), "unclear")
    if land in FOREST_LAND:
        forest = "yes" if all(crit) else "no"
    elif land == "unclear":
        forest = "uncertain"
    else:
        forest = "no"
    if forest == "no":
        defor = "no"
    elif forest == "yes":
        defor = "yes" if change == "conversion_to_agriculture" else ("uncertain" if change == "unclear" else "no")
    else:
        defor = "no" if change in NO_CONV else "uncertain"
    return {"forest_2020": forest, "deforestation_after_2020": defor,
            "verdict": {"no": "compliant", "yes": "non_compliant"}.get(defor, "uncertain"),
            "change_after_2020": change, "land_use_2020": land}


def vlm_predictions(model: str = MODEL) -> tuple[pd.DataFrame, pd.DataFrame]:
    cards_ids = {(c["item_id"], c["card"]): set(c["ids"]) for c in map(json.loads, open(OUT / "cards.jsonl", encoding="utf-8"))}
    f = pd.read_parquet(D / "legal_features.parquet").set_index("item_id")
    preds, checks = [], []
    raw = [x for fp in sorted(OUT.glob(f"raw_{model}_*.jsonl")) for x in open(fp, encoding="utf-8") if x.strip()]
    for row in map(json.loads, raw):
        v, i = row["id"].split("|")
        samples = row.get("raws") or [row["raw"]]
        js = [parse(x) for x in samples]
        ok = [j for j in js if j]
        if VARIANTS[v]["sys"] == "v2":  # facts from the VLM, law by code
            lang = VARIANTS[v]["lang"]
            sams = [legal_v2(j, lang, numeric_criteria(f.loc[i])) for j in ok]
            if not sams:
                preds.append({"method": f"vlm_{v}", "item_id": i, "forest_2020": "uncertain", "deforestation": "uncertain",
                              "verdict": "uncertain", "change": "unclear", "land": "unclear", "vote": 0.0, "valid": False})
                continue

            def vote2(key):
                top = pd.Series([s[key] for s in sams]).value_counts()
                return top.index[0], float(top.iloc[0] / len(js))
            verdict, share = vote2("verdict")
            preds.append({"method": f"vlm_{v}", "item_id": i, "forest_2020": vote2("forest_2020")[0],
                          "deforestation": vote2("deforestation_after_2020")[0], "verdict": verdict,
                          "change": vote2("change_after_2020")[0], "land": vote2("land_use_2020")[0], "vote": share,
                          "valid": True, "explanation": ok[0].get(KEYS2[lang][0], ""),
                          "conf": CONF2["en"][CONF2[lang].index(c)] if (c := ok[0].get(KEYS2[lang][5])) in CONF2[lang] else None})
            k = KEYS2[lang]
            cited = (ok[0].get(k[2]) or []) + (ok[0].get(k[4]) or [])
            checks.append({"method": f"vlm_{v}", "item_id": i, "n_cited": len(cited),
                           "citation_valid": all(e in cards_ids[(i, "v4")] | {"IMG"} for e in cited) if cited else False})
            continue
        if not ok:
            for suf in ("", "_sym"):
                preds.append({"method": f"vlm_{v}{suf}", "item_id": i, "forest_2020": "uncertain", "deforestation": "uncertain",
                              "verdict": "uncertain", "change": "unclear", "land": "unclear", "vote": 0.0, "valid": False})
            continue
        for suf, sams in (("", ok), ("_sym", [legal_logic(j) for j in ok])):
            def vote(key, sams=sams):
                top = pd.Series([j.get(key) for j in sams]).value_counts()
                return top.index[0], float(top.iloc[0] / len(js))
            verdict, share = vote("verdict")
            preds.append({"method": f"vlm_{v}{suf}", "item_id": i, "forest_2020": vote("forest_2020")[0],
                          "deforestation": vote("deforestation_after_2020")[0], "verdict": verdict,
                          "change": vote("change_after_2020")[0], "land": vote("land_use_2020")[0], "vote": share,
                          "valid": True, "explanation": ok[0].get("explanation_vi", "")})
        # faithfulness / legal-logic checks on the first sample
        j = ok[0]
        allowed = cards_ids[(i, "v1" if v == "v1swap" else "v4")] | {"IMG"}
        if VARIANTS[v]["card"] == "none":
            allowed = {"E01", "IMG"}
        if not VARIANTS[v]["image"]:
            allowed = allowed - {"IMG"}
        cited = [e for c in CRIT for e in (j.get("criteria", {}).get(c, {}).get("evidence") or [])]
        crit = {c: j.get("criteria", {}).get(c, {}).get("status") for c in CRIT}
        area = float(f.loc[i, "area_ha"])
        checks.append({"method": f"vlm_{v}", "item_id": i,
                       "citation_valid": all(e in allowed for e in cited) if cited else False,
                       "n_cited": len(cited),
                       "logic_forest_vs_F4": not (j.get("forest_2020") == "yes" and crit["F4_not_agricultural_use_2020"] == "not_met"),
                       "logic_defor_needs_forest": not (j.get("deforestation_after_2020") == "yes" and j.get("forest_2020") == "no"),
                       "logic_verdict": not ((j.get("verdict") == "non_compliant" and j.get("deforestation_after_2020") != "yes")
                                             or (j.get("verdict") == "compliant" and j.get("deforestation_after_2020") == "yes")),
                       "logic_F1_area": not ((crit["F1_area_over_0_5ha"] == "met" and area < 0.5) or
                                             (crit["F1_area_over_0_5ha"] == "not_met" and area >= 0.5 and j.get("forest_2020") == "no"
                                              and crit["F4_not_agricultural_use_2020"] != "not_met"))})
    return pd.DataFrame(preds), pd.DataFrame(checks)


def baselines() -> pd.DataFrame:
    """Deterministic decisions on the same evidence: EU-map style checks, national map, Whisp-like, legal rules, and
    the project's rules v0 (fuse.py)."""
    f = pd.read_parquet(D / "legal_features.parquet").fillna({"hl_post2020": 0, "tmf25_deforested": 0, "coffee_2024": 0,
                                                              "rubber_2020": 0, "rubber_2024": 0, "ty_treecrop": 0})
    out = []
    for r in f.itertuples():
        loss = (r.hl_post2020 >= 0.2) or (r.tmf25_deforested >= 0.2)

        def rec(m, forest, defor, verdict=None):
            out.append({"method": m, "item_id": r.item_id, "forest_2020": forest, "deforestation": defor,
                        "verdict": verdict or {"yes": "non_compliant", "no": "compliant"}.get(defor, "uncertain")})
        for v in ("V1", "V4"):  # "GFC2020 + Hansen" overlay as used by many EUDR tools
            fo = getattr(r, f"gfc{v}") >= 0.5
            rec(f"gfc{v}_hansen", "yes" if fo else "no", "yes" if fo and r.hl_post2020 >= 0.2 else "no")
        of = (r.off_natural + r.off_plantation) >= 0.5
        rec("national_loss", "yes" if of else "no", "yes" if of and loss else "no")
        votes = sum([r.gfcV4 >= 0.5, (r.tmf20_undisturbed + r.tmf20_degraded + r.tmf20_regrowth) >= 0.5,
                     (r.tc2000 or 0) >= 30 and (r.hl_pre2021 or 0) < 0.5])
        rec("whisp_like", "yes" if votes >= 2 else ("uncertain" if votes == 1 else "no"),
            "yes" if votes >= 2 and loss else ("uncertain" if votes == 1 and loss else "no"))
        # legal rules on the card (same criteria as the prompt, fixed thresholds)
        tall = max(r.h_meta_p90 if pd.notna(r.h_meta_p90) else 0, r.h_eth if pd.notna(r.h_eth) else 0) >= 5
        sp = str(r.lot_sldlr).lower() if isinstance(r.lot_sldlr, str) else ""
        agri = ((r.coffee_2024 >= 0.5 and (r.hl_pre2021 or 0) < 0.2 and not loss) or r.coffee_2020 >= 0.3 or r.rubber_2020 >= 0.5
                or r.ty_treecrop >= 0.5 or r.lot_ldlr == "DNN" or "cao su" in sp or "điều" in sp)
        treed = max(r.gfcV4, r.wc20_tree if pd.notna(r.wc20_tree) else 0, (r.tc2000 or 0) / 100, r.off_natural + r.off_plantation) >= 0.5
        # F1 is about the contiguous forest the object belongs to: the alert patch itself is often < 0.5 ha
        big = r.area_ha >= 0.5 or max(r.buf_gfcV4 or 0, r.buf_off_forest or 0, (r.buf_tc2000 or 0) / 100) >= 0.5
        forest = big and tall and treed and not agri
        plantation = r.gft_planted >= 0.5 or r.off_plantation >= 0.5 or (r.tmf_plantation or 0) >= 0.5
        z = lambda x: 0 if pd.isna(x) else x  # noqa: E731
        # signs of agricultural use after the loss: crop models / Dynamic World, or no canopy recovery a year later
        agri_now = (r.coffee_2024 >= 0.5 or z(r.dw_crops_2025) >= 0.3 or r.rubber_2024 >= 0.5
                    or z(r.dw_post_crops) + z(r.dw_post_bare) >= 0.3)
        no_recovery = pd.notna(r.ndvi_pre) and pd.notna(r.ndvi_rec12m) and r.ndvi_pre - r.ndvi_rec12m >= 0.2
        recovered = (pd.notna(r.ndvi_rec12m) and pd.notna(r.ndvi_post) and r.ndvi_rec12m - r.ndvi_post >= 0.15) or z(r.tmf25_regrowth) >= 0.3
        lost = loss or (r.kind == "alert" and pd.notna(r.ndvi_pre) and pd.notna(r.ndvi_post) and r.ndvi_pre - r.ndvi_post >= 0.15)
        if not forest:
            rec("rules_legal", "no", "no")
        elif not lost:
            rec("rules_legal", "yes", "no")
        elif (agri_now or no_recovery) and not (plantation and recovered):
            rec("rules_legal", "yes", "yes")
        elif recovered:
            rec("rules_legal", "yes", "no")
        else:
            rec("rules_legal", "yes", "uncertain")
        cat = {"GREEN": "compliant", "FALSE": "compliant", "PLANTATION": "compliant", "RED": "non_compliant"}.get(r.category, "uncertain")
        out.append({"method": "project_rules_v0", "item_id": r.item_id, "forest_2020": None,
                    "deforestation": {"compliant": "no", "non_compliant": "yes"}.get(cat, "uncertain"), "verdict": cat})
    return pd.DataFrame(out)


def extra_baselines(t: pd.DataFrame, t_train: pd.DataFrame | None = None) -> pd.DataFrame:
    """FAO Whisp (perennial-crop risk; run by whisp_baseline.py in its own venv) and LightGBM on the evidence-card
    features with spatial block cross-validation (0.2 deg blocks, 5 folds, out-of-fold predictions; trained on the
    same silver labels it is scored on, so it is an upper-bound style baseline, not an independent one)."""
    out = []
    wf = OUT / "whisp_items.csv"
    if wf.exists():
        w = pd.read_csv(wf)
        for r in w.itertuples():
            out.append({"method": "whisp_fao", "item_id": r.external_id,
                        "forest_2020": {"yes": "yes", "no": "no"}.get(r.Ind_01_treecover, "uncertain"),
                        "deforestation": {"high": "yes", "low": "no"}.get(r.risk_pcrop, "uncertain"),
                        "verdict": {"high": "non_compliant", "low": "compliant"}.get(r.risk_pcrop, "uncertain")})
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    f = pd.read_parquet(D / "legal_features.parquet")
    xy = pd.read_parquet(D / "triage.parquet")[["item_id", "lon", "lat"]]
    d = f.merge(xy, on="item_id").merge(t[["item_id", "y2"]], on="item_id").reset_index(drop=True)
    feats = [c for c in d.columns if d[c].dtype.kind in "fib" and c not in ("y2", "lon", "lat", "lot_namtr", "lot_malr3")]
    X = d[feats].astype(float)
    groups = np.floor(d.lon / 0.2).astype(int).astype(str) + "_" + np.floor(d.lat / 0.2).astype(int).astype(str)
    if t_train is not None:  # s1: fit once on DEV, predict TEST (no label of a scored item is ever seen)
        tr = f.merge(xy, on="item_id").merge(t_train[["item_id", "y2"]], on="item_id")
        tr = tr[tr.y2.notna()]
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=8, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
        m.fit(tr[feats].astype(float), tr.y2.astype(int))
        p = pd.Series(m.predict_proba(X)[:, 1], index=d.index)
        for i in d.index:
            pi = float(p[i])
            out.append({"method": "lgbm_dev", "item_id": d.item_id[i], "forest_2020": None,
                        "deforestation": "yes" if pi >= 0.5 else "no", "verdict": "non_compliant" if pi >= 0.5 else "compliant"})
            dd = "yes" if pi >= 0.65 else ("no" if pi <= 0.35 else "uncertain")
            out.append({"method": "lgbm_dev_abstain", "item_id": d.item_id[i], "forest_2020": None, "deforestation": dd,
                        "verdict": {"yes": "non_compliant", "no": "compliant"}.get(dd, "uncertain")})
        return pd.DataFrame(out)
    idx = d.index[d.y2.notna()]
    if len(idx) < 30 or groups[idx].nunique() < 5 or d.y2[idx].nunique() < 2:  # too few labels yet (partial human labels)
        return pd.DataFrame(out)
    p = pd.Series(np.nan, index=d.index)
    for tr, te in GroupKFold(n_splits=5).split(X.loc[idx], d.y2[idx], groups[idx]):
        ytr = d.y2[idx[tr]].astype(int)
        if ytr.nunique() < 2:  # a fold whose training blocks hold one class only
            p.loc[idx[te]] = float(ytr.iloc[0])
            continue
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=8, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
        m.fit(X.loc[idx[tr]], ytr)
        p.loc[idx[te]] = m.predict_proba(X.loc[idx[te]])[:, 1]
    for i in idx:
        pi = float(p[i])
        out.append({"method": "lgbm_cv", "item_id": d.item_id[i], "forest_2020": None,
                    "deforestation": "yes" if pi >= 0.5 else "no", "verdict": "non_compliant" if pi >= 0.5 else "compliant"})
        dd = "yes" if pi >= 0.65 else ("no" if pi <= 0.35 else "uncertain")
        out.append({"method": "lgbm_cv_abstain", "item_id": d.item_id[i], "forest_2020": None, "deforestation": dd,
                    "verdict": {"yes": "non_compliant", "no": "compliant"}.get(dd, "uncertain")})
    return pd.DataFrame(out)


def metrics(p: pd.DataFrame, t: pd.DataFrame, rng: np.random.Generator, B: int = 1000) -> dict:
    m = t.merge(p, on="item_id")
    res = {}
    d = m[m.y2.notna()].copy()
    d["pred"] = d.deforestation.map({"yes": 1, "no": 0})
    cov = d.pred.notna()

    def fgr(x):  # among items called compliant (no deforestation), share that are deforestation
        c = x[x.pred == 0]
        return float((c.y2 == 1).mean()) if len(c) else np.nan

    def acc(x):
        c = x[x.pred.notna()]
        return float((c.pred == c.y2).mean()) if len(c) else np.nan
    boot = {"coverage": [], "acc_covered": [], "false_green": []}
    for _ in range(B):
        s = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
        boot["coverage"].append(float(s.pred.notna().mean()))
        boot["acc_covered"].append(acc(s))
        boot["false_green"].append(fgr(s))
    ci = {k: [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)] for k, v in boot.items()}
    pos = d[d.y2 == 1]
    res["deforestation"] = {"n": len(d), "n_defor": int((d.y2 == 1).sum()), "coverage": round(float(cov.mean()), 3),
                            "acc_covered": round(acc(d), 3), "acc_all_uncertain_wrong": round(float((d.pred == d.y2).mean()), 3),
                            "recall_defor": round(float((pos.pred == 1).mean()), 3) if len(pos) else None,
                            "false_green": round(fgr(d), 3) if not np.isnan(fgr(d)) else None,
                            "false_red": round(float((d[d.pred == 1].y2 == 0).mean()), 3) if (d.pred == 1).any() else None,
                            "ci95": ci}
    e = m[m.y1.notna() & m.forest_2020.notna()].copy()
    e["pred"] = e.forest_2020.map({"yes": 1, "no": 0})
    ec = e[e.pred.notna()]
    res["forest_2020"] = {"n": len(e), "coverage": round(float(e.pred.notna().mean()), 3) if len(e) else None,
                          "acc_covered": round(float((ec.pred == ec.y1).mean()), 3) if len(ec) else None}
    if "change" in m:
        a = m[m.cause.notna() & m.change.notna()]
        grp = {"none": "no_conversion", "agricultural_renovation": "no_conversion", "plantation_rotation": "rotation",
               "fire_without_conversion": "fire_natural", "natural_disturbance": "fire_natural", "selective_logging": "logging",
               "conversion_to_agriculture": "agriculture", "conversion_to_infrastructure": "infrastructure", "unclear": "unclear"}
        a = a.assign(g_pred=a.change.map(grp), g_true=a.cause.map(grp))
        res["cause_alerts"] = {"n": len(a), "acc": round(float((a.g_pred == a.g_true).mean()), 3) if len(a) else None,
                               "confusion": pd.crosstab(a.g_true, a.g_pred).to_dict()}
    return res


ENSEMBLE = ["v2vi", "v2vi_p2", "v2vi_shuf", "v2en", "v2en_full"]  # five wordings of the v2 prompt


def _tag(method: str) -> str:  # "@ivl8b" for methods of another model, "" for the main model and the baselines
    return "@" + method.split("@", 1)[1] if "@" in method else ""


def _mcnemar(a: pd.Series, b: pd.Series) -> dict:
    """Exact McNemar on per-item correctness (1/0, "uncertain" counted wrong); diff = accuracy of b minus a."""
    from scipy.stats import binomtest
    a_only, b_only = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"diff": round(float(b.mean() - a.mean()), 3), "a_only": a_only, "b_only": b_only, "p": round(float(p), 4)}


def _rule(pred: pd.Series, y: pd.Series) -> dict:
    """A decision rule (1 / 0 / NaN = no conclusion) against the truth."""
    c = pred.notna()
    r = lambda x: None if pd.isna(x) else round(float(x), 3)  # noqa: E731
    return {"coverage": r(c.mean()), "acc_covered": r((pred[c] == y[c]).mean()) if c.any() else None,
            "acc_all": r((pred == y).mean()), "recall": r((pred[y == 1] == 1).mean()),
            "false_green": r((y[c][pred[c] == 0] == 1).mean()) if (pred[c] == 0).any() else None}


def robustness(allp: pd.DataFrame, t: pd.DataFrame) -> dict:
    """Paired tests and agreement-based rules, per model: McNemar of every v2 variant against v2vi and of v2vi against
    GFC V4 + Hansen; the five-wording prompt ensemble (pairwise agreement, >= 2/5 say deforestation, unanimity);
    how many verdicts the GFC V1 card changes, next to the overlay baseline; agreement between models."""
    P_all = allp.pivot_table(index="item_id", columns="method", values="deforestation", aggfunc="first").fillna("uncertain")
    tt = t[t.y2.notna()].set_index("item_id")
    P = P_all.reindex(tt.index).fillna("uncertain")
    num = P.apply(lambda s: s.map({"yes": 1, "no": 0}))
    ok = num.eq(tt.y2, axis=0).astype(int)
    tags = sorted({_tag(m) for m in P if m.startswith("vlm_v2vi") and _tag(m) in MODEL_TAG.values()})
    res = {"mcnemar": {}, "prompt_ensemble": {}, "gfc_card": {}, "cross_model": {}}
    for tag in tags:
        base = f"vlm_v2vi{tag}"
        if base not in P:
            continue
        res["mcnemar"][base] = {m: _mcnemar(ok[base], ok[m]) for m in P if m.startswith("vlm_v2") and _tag(m) == tag and m != base}
        res["mcnemar"][base]["base_vs_gfcV4_hansen"] = _mcnemar(ok["gfcV4_hansen"], ok[base])
        E = [f"vlm_{v}{tag}" for v in ENSEMBLE]
        if all(e in P for e in E):
            X = P[E]
            yes = (X == "yes").sum(axis=1)
            strict = (X.nunique(axis=1) == 1) & X.iloc[:, 0].isin(["yes", "no"])
            loose = X.apply(lambda r: next(iter(set(r) - {"uncertain"})) if len(set(r) - {"uncertain"}) == 1 else None, axis=1)
            res["prompt_ensemble"][tag or "main"] = {
                "pairwise_same_verdict": [round(float(min(v)), 3), round(float(max(v)), 3)] if (v := [
                    (X[a] == X[b]).mean() for a, b in itertools.combinations(E, 2)]) else None,
                "at_least_2_of_5_yes": _rule((yes >= 2).astype(float), tt.y2),
                "unanimous_5_of_5": _rule(num[E[0]].where(strict), tt.y2),
                "unanimous_ignoring_uncertain": _rule(loose.map({"yes": 1, "no": 0}), tt.y2)}
        if f"vlm_v2vi_v1card{tag}" in P_all:
            res["gfc_card"][base] = int((P_all[base] != P_all[f"vlm_v2vi_v1card{tag}"]).sum())
    res["gfc_card"]["gfcV1_hansen_vs_gfcV4_hansen"] = int((P_all.gfcV1_hansen != P_all.gfcV4_hansen).sum())
    res["gfc_card"]["n_items"] = len(P_all)
    L = allp.pivot_table(index="item_id", columns="method", values="land", aggfunc="first")
    C = allp.pivot_table(index="item_id", columns="method", values="change", aggfunc="first")
    for ta, tb in itertools.combinations(tags, 2):
        a, b = f"vlm_v2vi{ta}", f"vlm_v2vi{tb}"
        both = num[a].where(num[a] == num[b])
        res["cross_model"][f"{a} | {b}"] = {
            "same_verdict": round(float((P_all[a] == P_all[b]).mean()), 3), "mcnemar": _mcnemar(ok[a], ok[b]),
            "kappa_land": round(float(_kappa(L[a], L[b])), 3), "kappa_change": round(float(_kappa(C[a], C[b])), 3),
            "both_agree_rule": _rule(both, tt.y2)}
    return res


def _kappa(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    a, b = a[m].astype(str), b[m].astype(str)
    po = float((a == b).mean())
    pe = sum(float((a == k).mean()) * float((b == k).mean()) for k in set(a) | set(b))
    return 1.0 if pe >= 1 else (po - pe) / (1 - pe)


# human labels of form v3 (human_labels.py, prereg amendment A3) have 6 changes: the 8 answers of the models are folded
# into them to compare causes; `metrics` (frozen) keeps the 8-class grouping for the silver labels
GRP_V3 = {"none": "none", "conversion_to_agriculture": "agriculture", "plantation_rotation": "rotation",
          "cleared_use_unseen": "cleared", "unclear": "unclear", **{k: "other" for k in (
              "other_non_agricultural", "fire_without_conversion", "natural_disturbance", "selective_logging",
              "conversion_to_infrastructure", "agricultural_renovation")}}


def cause_form_v3(p: pd.DataFrame, t: pd.DataFrame) -> dict:
    m = t.merge(p, on="item_id")
    a = m[m.cause.notna() & m.change.notna()] if "change" in m else m.iloc[:0]
    a = a.assign(g_pred=a.change.map(GRP_V3), g_true=a.cause.map(GRP_V3))
    return {"n": len(a), "grouping": "form_v3", "acc": round(float((a.g_pred == a.g_true).mean()), 3) if len(a) else None,
            "confusion": pd.crosstab(a.g_true, a.g_pred).to_dict()}


def evaluate(model: str = MODEL, truth: str = "silver", truth_file: str | None = None, split: str | None = None):
    """Score every method against the silver labels, or against the human labels (truth="human": the table written by
    human_labels.py; report and predictions then get the suffix _human and the silver files stay untouched)."""
    t_all = silver_truth() if truth == "silver" else pd.read_csv(truth_file or OUT / "human_truth.csv")
    t = t_all[t_all.split == split].reset_index(drop=True) if split else t_all  # s1: TEST is scored, DEV only trains
    sfx = ("" if truth == "silver" else f"_{truth}") + (f"_{split}" if split else "")
    conf = t.silver_conf if "silver_conf" in t else t.conf
    vps, chks = [], []
    for mdl, tag in MODEL_TAG.items():  # every model with raw outputs; methods of other models get a suffix
        if not list(OUT.glob(f"raw_{mdl}_*.jsonl")):
            continue
        a, b = vlm_predictions(mdl)
        vps.append(a.assign(method=a.method + tag))
        chks.append(b.assign(method=b.method + tag) if len(b) else b)
    vp, chk = pd.concat(vps, ignore_index=True), pd.concat(chks, ignore_index=True)
    bp = pd.concat([baselines(), extra_baselines(t, t_all[t_all.split == "dev"] if split == "test" else None)], ignore_index=True)
    allp = pd.concat([vp, bp], ignore_index=True)
    rng = np.random.default_rng(42)
    res = {f"{truth}_truth": {"n_items": len(t), "deforestation_known": int(t.y2.notna().sum()), "deforestation_yes": int((t.y2 == 1).sum()),
                              "forest2020_known": int(t.y1.notna().sum())},
           "methods": {}, "by_kind": {}, f"medium_high_{truth}_only": {}}
    if "y2_strict" in t:  # human truth: the stricter reading of the forest thresholds as a sensitivity check
        res["strict_forest_definition"] = {}
    v3 = "cause" in t and t.cause.isin(["other_non_agricultural", "cleared_use_unseen"]).any()
    for m, p in allp.groupby("method"):
        res["methods"][m] = metrics(p, t, rng)
        if v3:
            res["methods"][m]["cause_alerts"] = cause_form_v3(p, t)
        for k in ("alert", "plot"):
            res["by_kind"].setdefault(m, {})[k] = metrics(p, t[t.kind == k], rng, B=200)["deforestation"]
        res[f"medium_high_{truth}_only"][m] = metrics(p, t[conf.isin(["medium", "high"])], rng, B=200)["deforestation"]
        if "y2_strict" in t:
            res["strict_forest_definition"][m] = metrics(p, t.assign(y1=t.y1_strict, y2=t.y2_strict), rng, B=200)["deforestation"]
    if len(chk):
        cols = [c for c in ("citation_valid", "logic_forest_vs_F4", "logic_defor_needs_forest", "logic_verdict", "logic_F1_area")
                if c in chk]  # the logic checks exist only for the v1 design (s1 runs v2 only)
        res["faithfulness"] = chk.groupby("method")[cols].mean().round(3).to_dict("index")
        res["valid_json"] = vp.groupby("method").valid.mean().round(3).to_dict()
    res["self_consistency"] = {}
    for meth in sorted(m for m in vp.method.unique() if m.split("@")[0] in ("vlm_sc_sym", "vlm_v2vi_sc")):  # every model
        sc = vp[vp.method == meth].merge(t, on="item_id")
        if not len(sc):
            continue
        curve = []
        for th in (0.4, 0.6, 0.8, 1.0):
            s = sc[sc.y2.notna()].copy()
            s["pred"] = np.where(s.vote >= th - 1e-9, s.deforestation.map({"yes": 1, "no": 0}), np.nan)
            c = s[s.pred.notna()]
            curve.append({"min_vote": th, "coverage": round(float(s.pred.notna().mean()), 3),
                          "acc_covered": round(float((c.pred == c.y2).mean()), 3) if len(c) else None,
                          "false_green": round(float((c[c.pred == 0].y2 == 1).mean()), 3) if (c.pred == 0).any() else None})
        res["self_consistency"][meth] = {"all5_agree": round(float((sc.vote >= 0.999).mean()), 3),
                                         "mean_vote": round(float(sc.vote.mean()), 3), "curve": curve}
    a, b = vp[vp.method == "vlm_full"].set_index("item_id"), vp[vp.method == "vlm_v1swap"].set_index("item_id")
    if len(a) and len(b):
        j = a.join(b, rsuffix="_v1", how="inner")
        res["gfc_version_sensitivity"] = {"verdict_changed": round(float((j.verdict != j.verdict_v1).mean()), 3),
                                          "forest2020_changed": round(float((j.forest_2020 != j.forest_2020_v1).mean()), 3),
                                          "plots_forest2020_changed": round(float((j[j.index.str.startswith("P")].forest_2020 !=
                                                                                   j[j.index.str.startswith("P")].forest_2020_v1).mean()), 3)}
    res["robustness"] = robustness(allp, t)
    RUNS.joinpath("dossier").mkdir(parents=True, exist_ok=True)
    (RUNS / "dossier" / f"legal_vlm_report{f'_{SET}' if SET else ''}{sfx}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    allp.to_csv(OUT / f"predictions_{model}{sfx}.csv", index=False)
    rows = [{"method": m, **{k: v for k, v in r["deforestation"].items() if k != "ci95"},
             "ci95_acc": r["deforestation"]["ci95"]["acc_covered"], "ci95_fg": r["deforestation"]["ci95"]["false_green"],
             "forest2020_acc": r["forest_2020"]["acc_covered"], "cause_acc": r.get("cause_alerts", {}).get("acc")}
            for m, r in res["methods"].items()]
    tab = pd.DataFrame(rows).sort_values("acc_all_uncertain_wrong", ascending=False)
    print(tab.to_string(index=False))
    print(json.dumps({k: res.get(k) for k in ("faithfulness", "valid_json", "self_consistency", "gfc_version_sensitivity")},
                     indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "cards", "run", "eval"])
    ap.add_argument("--variants", default="full,noimg,imgonly,nodef,v1swap,sc")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--truth", choices=["silver", "human", "q35"], default="silver",
                    help="eval: dot0 silver labels, human labels, or the s1 silver labels of Qwen3.5-35B-A3B (silver_set.py)")
    ap.add_argument("--split", default=None, help="eval: score only this split of the truth table (s1: test)")
    ap.add_argument("--chunk", type=int, default=0, help="run: items per GPU-server batch (s1)")
    ap.add_argument("--truth-file", default=None, help="eval --truth human: table from human_labels.py")
    a = ap.parse_args()
    if a.cmd == "features":
        features()
    elif a.cmd == "cards":
        cards()
    elif a.cmd == "run":
        run(a.variants.split(","), a.gpu, a.gpu_mem, a.model, a.tag, a.chunk)
    else:
        evaluate(a.model, a.truth, a.truth_file, a.split)


if __name__ == "__main__":
    main()
