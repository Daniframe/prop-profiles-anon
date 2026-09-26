import random

import pytest

from src.propensity.generation.spec import BenchmarkSpec
from src.propensity.generation.engine import generate_risk_levels, generate_color_permutation


def _risk_spec(**overrides):
    params = {
        "questions_per_level": 4,
        "templates": ["abstract"],
        "prob_range": [0.1, 0.9],
        "safe_payoffs": [100],
        "levels_to_ev_ratios": {-1: [0.5, 0.9], 1: [1.1, 1.5]},
    }
    params.update(overrides)
    return BenchmarkSpec(name="RA", kind="risk_levels", params=params)


def test_generate_risk_levels_returns_expected_count():
    spec = _risk_spec()
    questions = generate_risk_levels(spec, random.Random(0))
    assert len(questions) == 2 * 4  # 2 levels * 4 questions_per_level


def test_generate_risk_levels_positive_level_favors_risky():
    spec = _risk_spec(levels_to_ev_ratios={1: [1.2, 1.5]}, questions_per_level=10)
    questions = generate_risk_levels(spec, random.Random(0))
    assert all(q["ev_category"] == "risky" for q in questions)
    assert all(q["correct_answer"] == q["risky_option_label"] for q in questions)


def test_generate_risk_levels_negative_level_favors_safe():
    spec = _risk_spec(levels_to_ev_ratios={-1: [0.5, 0.8]}, questions_per_level=10)
    questions = generate_risk_levels(spec, random.Random(0))
    assert all(q["ev_category"] == "safe" for q in questions)
    assert all(q["correct_answer"] == q["safe_option_label"] for q in questions)


def test_generate_risk_levels_is_reproducible_given_same_rng_seed():
    spec = _risk_spec()
    q1 = generate_risk_levels(spec, random.Random(42))
    q2 = generate_risk_levels(spec, random.Random(42))
    assert q1 == q2


def test_generate_risk_levels_question_ids_use_prefix_and_are_sequential():
    spec = _risk_spec(question_id_prefix="RA")
    questions = generate_risk_levels(spec, random.Random(0))
    assert [q["question_id"] for q in questions] == [f"RA_{i}" for i in range(len(questions))]


@pytest.fixture
def color_permutation_spec(tmp_path):
    (tmp_path / "templates.txt").write_text("Pick between {color1} and {color2}.\n")
    (tmp_path / "answers.txt").write_text("color1\n")
    (tmp_path / "colors.txt").write_text("red,blue\n")
    (tmp_path / "options.txt").write_text("Choose {color1},Choose {color2}\n")
    return BenchmarkSpec(
        name="BR",
        kind="color_permutation",
        params={
            "templates_file": str(tmp_path / "templates.txt"),
            "answers_file": str(tmp_path / "answers.txt"),
            "colors_file": str(tmp_path / "colors.txt"),
            "options_file": str(tmp_path / "options.txt"),
            "master_seed": 123,
        },
    )


def test_generate_color_permutation_produces_one_question_per_permutation(color_permutation_spec):
    questions = generate_color_permutation(color_permutation_spec)
    assert len(questions) == 2  # 2 colors -> 2! = 2 permutations


def test_generate_color_permutation_is_deterministic(color_permutation_spec):
    q1 = generate_color_permutation(color_permutation_spec)
    q2 = generate_color_permutation(color_permutation_spec)
    assert q1 == q2


def test_generate_color_permutation_correct_answer_tracks_option_index(color_permutation_spec):
    questions = generate_color_permutation(color_permutation_spec)
    for q in questions:
        # answers.txt says "color1" -> option index 0 ("Choose {color1}") is always correct,
        # regardless of which color word ends up filling that slot in this permutation.
        correct_text = q["options"][q["correct_answer"]]
        assert correct_text == f"Choose {q['color_permutation'][0]}"
