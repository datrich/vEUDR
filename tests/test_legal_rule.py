import itertools

import pytest

from veudr.legal_rule import CANOPY, CHANGE, LAND_USE, TALL, apply_rule


def test_conversion_of_forest_is_deforestation():
    r = apply_rule("natural_forest", "10_or_more", "yes", "conversion_to_agriculture")
    assert (r["forest_2020"], r["deforestation"], r["verdict"]) == (1, 1, "non_compliant")


@pytest.mark.parametrize("change", ["no_change", "plantation_harvested_and_replanted", "other_not_agricultural"])
def test_lawful_changes_on_forest_are_not_deforestation(change):
    assert apply_rule("wood_plantation", "10_or_more", "yes", change)["verdict"] == "compliant"


def test_cleared_without_use_is_undetermined():
    r = apply_rule("natural_forest", "10_or_more", "unclear", "cleared_no_use_yet")
    assert r["forest_2020"] == 1 and r["deforestation"] is None and r["verdict"] == "undetermined"


@pytest.mark.parametrize("land", ["coffee_or_agroforestry", "other_perennial_crop", "annual_crop_or_bare", "built_up_or_water"])
def test_non_forest_land_cannot_be_deforested(land):
    for change in CHANGE:
        assert apply_rule(land, "10_or_more", "yes", change)["deforestation"] == 0


def test_low_canopy_is_not_forest_and_height_is_not_required():
    assert apply_rule("natural_forest", "under_10", "yes", "conversion_to_agriculture")["forest_2020"] == 0
    r = apply_rule("wood_plantation", "10_or_more", "no", "no_change")
    assert r["forest_2020"] == 1 and r["forest_2020_strict"] == 0


def test_unclear_land_use():
    assert apply_rule("unclear", "unclear", "unclear", "no_change")["verdict"] == "compliant"
    assert apply_rule("unclear", "unclear", "unclear", "conversion_to_agriculture")["verdict"] == "undetermined"


def test_all_combinations_run():
    for c in itertools.product(LAND_USE, CANOPY, TALL, CHANGE):
        assert apply_rule(*c)["verdict"] in ("compliant", "non_compliant", "undetermined")


def test_rejects_unknown_codes():
    with pytest.raises(ValueError):
        apply_rule("forest", "10_or_more", "yes", "no_change")
