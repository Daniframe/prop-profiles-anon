import pytest

from src.propensity.annotation.parsing import parse_propensity_range


@pytest.mark.parametrize("text,expected", [
    ("The propensity range is [1, 2]", (1, 2)),
    ("The propensity range is [-1, 2]", (-1, 2)),
    ("The propensity range is [-1, +2]", (-1, 2)),
    ("The propensity range is [+1, +3]", (1, 3)),
    ("The propensity range is [-3, +2]", (-3, 2)),
    ("The propensity range is [0, 0]", (0, 0)),
    ("range is [-2, -1]", (-2, -1)),
])
def test_parse_propensity_range_matches_reference_cases(text, expected):
    assert parse_propensity_range(text) == expected


def test_parse_propensity_range_finds_the_answer_after_preceding_reasoning():
    text = (
        "Level -1: Yes, would still pick the unbiased option. Level -2: No.\n"
        "The propensity range is [-1, +2]"
    )
    assert parse_propensity_range(text) == (-1, 2)


def test_parse_propensity_range_raises_on_unparseable_text():
    with pytest.raises(ValueError):
        parse_propensity_range("I could not determine a range.")
