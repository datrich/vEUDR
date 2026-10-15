"""Article 2 of Regulation (EU) 2023/1115, read with Section 11 of Commission guidance C(2026) 4972, applied by code to
facts about one item (an alert patch or a plot).

Annotators and models only state facts: the land use on 31 December 2020 (Q1), canopy cover and tree height in 2020,
and the kind of change after the cut-off (Q2). This function turns them into the forest-2020 status and the
deforestation conclusion. It has the same branches as the function frozen in the study's preregistration
(``law`` in the research code, hash 22f32e93bb5b640d), with English option codes.

    >>> apply_rule("natural_forest", "10_or_more", "yes", "conversion_to_agriculture")
    {'forest_2020': 1, 'deforestation': 1, 'forest_2020_strict': 1, 'deforestation_strict': 1, 'verdict': 'non_compliant'}
    >>> apply_rule("natural_forest", "10_or_more", "unclear", "cleared_no_use_yet")["verdict"]
    'undetermined'
"""
from __future__ import annotations

LAND_USE = ("natural_forest", "wood_plantation", "coffee_or_agroforestry", "other_perennial_crop",
            "annual_crop_or_bare", "built_up_or_water", "unclear")
CANOPY = ("under_10", "10_or_more", "unclear")
TALL = ("yes", "no", "unclear")
CHANGE = ("no_change", "conversion_to_agriculture", "plantation_harvested_and_replanted", "cleared_no_use_yet",
          "other_not_agricultural", "unclear")

FOREST_LAND = {"natural_forest", "wood_plantation"}
# changes that are not a conversion of forest to agricultural use (Art. 2(3); guidance s.11: harvest and replanting,
# disturbance with regrowth, selective logging and infrastructure are not deforestation)
NO_CONVERSION = {"no_change", "plantation_harvested_and_replanted", "other_not_agricultural"}


def apply_rule(land_use_2020: str, canopy_2020: str, trees_over_5m: str, change_after_2020: str) -> dict:
    """Forest on 31 Dec 2020 and deforestation after it, as 1 / 0 / None (None = undetermined).

    forest_2020         natural forest or wood plantation, canopy not below 10 %. Height is not required: Art. 2(4)
                        also counts trees able to reach 5 m in situ (young stands, temporarily unstocked land).
    forest_2020_strict  as above and trees taller than about 5 m seen (sensitivity analysis).
    deforestation       forest in 2020 and converted to agriculture with the use visible. Cleared forest without a
                        visible use is undetermined: temporarily unstocked land remains forest (guidance s.11).
                        Land that was not forest in 2020 cannot be deforested.
    """
    for v, allowed, name in ((land_use_2020, LAND_USE, "land_use_2020"), (canopy_2020, CANOPY, "canopy_2020"),
                             (trees_over_5m, TALL, "trees_over_5m"), (change_after_2020, CHANGE, "change_after_2020")):
        if v not in allowed:
            raise ValueError(f"{name}={v!r}; expected one of {allowed}")
    if land_use_2020 == "unclear":
        f = fs = None
    elif land_use_2020 in FOREST_LAND:
        f = 0 if canopy_2020 == "under_10" else 1
        fs = 1 if canopy_2020 == "10_or_more" and trees_over_5m == "yes" else (
            0 if canopy_2020 == "under_10" or trees_over_5m == "no" else None)
    else:
        f = fs = 0

    def defor(forest):
        if forest == 1:
            if change_after_2020 == "conversion_to_agriculture":
                return 1
            return None if change_after_2020 in ("cleared_no_use_yet", "unclear") else 0
        return 0 if forest == 0 or change_after_2020 in NO_CONVERSION else None

    d = defor(f)
    return {"forest_2020": f, "deforestation": d, "forest_2020_strict": fs, "deforestation_strict": defor(fs),
            "verdict": {1: "non_compliant", 0: "compliant"}.get(d, "undetermined")}
