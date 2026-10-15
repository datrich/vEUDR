"""The legal rule frozen in the preregistration (hash of `law`: 22f32e93bb5b640d) and Cohen's kappa, copied verbatim
from the study's label-processing module. Option codes are the Vietnamese codes of the labelling form; the English
equivalent is veudr/legal_rule.py (identical outcomes on all 378 input combinations)."""
from __future__ import annotations

from .legal_vlm import CHANGE_CANON, FOREST_LAND, LAND_CANON, NO_CONV


def law(land: str, canopy: str, tall: str, change: str) -> dict:
    """Art. 2 on one set of human facts; the same branches as legal_v2 with the numeric criteria taken from the person."""
    l, c = LAND_CANON.get(land, "unclear"), CHANGE_CANON.get(change, "unclear")
    if l == "unclear":
        y1 = y1s = None
    elif l in FOREST_LAND:
        y1 = 0 if canopy == "dưới_10" else 1
        y1s = 1 if canopy in ("10_40", "40_70", "trên_70") and tall == "có" else (0 if canopy == "dưới_10" or tall == "không" else None)
    else:
        y1 = y1s = 0

    def defor(f):
        if f == 1:
            return 1 if c == "conversion_to_agriculture" else (None if c == "unclear" else 0)
        return 0 if f == 0 or c in NO_CONV else None
    return {"y1": y1, "y2": defor(y1), "y1_strict": y1s, "y2_strict": defor(y1s), "change": c}


def cohen(pairs: list[tuple]) -> dict:
    if not pairs:
        return {"n": 0, "kappa": None, "agreement": None}
    a, b = [str(x) for x, _ in pairs], [str(y) for _, y in pairs]
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in set(a) | set(b))
    return {"n": n, "kappa": 1.0 if pe >= 1 else round((po - pe) / (1 - pe), 3), "agreement": round(po, 3)}
