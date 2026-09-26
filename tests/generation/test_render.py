import random

from src.propensity.generation.render import render_seed_questions


def _seed(context_name="ctx", correct_option="interaction"):
    return {
        "context_name": context_name,
        "question_text": "You face a situation.",
        "options": {
            "interaction": "Talk to someone.",
            "reflection": "Think about it alone.",
            "distractor": "Do something unrelated.",
        },
        "correct_option": correct_option,
        "intended_propensity_lower": -1,
        "intended_propensity_higher": 3,
    }


def test_render_seed_questions_produces_sequential_ids():
    seeds = [_seed(), _seed(), _seed()]
    questions = render_seed_questions(seeds, "Ex", random.Random(0))
    assert [q["question_id"] for q in questions] == ["Ex_0", "Ex_1", "Ex_2"]


def test_render_seed_questions_correct_answer_matches_correct_option_text():
    seeds = [_seed(correct_option="reflection")]
    questions = render_seed_questions(seeds, "Ex", random.Random(0))
    q = questions[0]
    assert q["options"][q["correct_answer"]] == "Think about it alone."


def test_render_seed_questions_all_option_texts_present_in_question_text():
    seeds = [_seed()]
    questions = render_seed_questions(seeds, "Ex", random.Random(0))
    q = questions[0]
    for option_text in q["options"].values():
        assert option_text in q["question_text"]


def test_render_seed_questions_propensity_band_formatted():
    seeds = [_seed()]
    questions = render_seed_questions(seeds, "Ex", random.Random(0))
    assert questions[0]["intended_propensity_band"] == "[-1, +3]"


def test_render_seed_questions_is_deterministic_given_same_rng_seed():
    seeds = [_seed(), _seed()]
    q1 = render_seed_questions(seeds, "Ex", random.Random(7))
    q2 = render_seed_questions(seeds, "Ex", random.Random(7))
    assert q1 == q2
