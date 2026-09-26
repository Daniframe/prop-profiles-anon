from src.propensity.annotation.prompts import build_annotation_prompt, ANNOTATION_SYSTEM_PROMPT


def test_build_annotation_prompt_fills_all_placeholders():
    prompt = build_annotation_prompt(
        propensity_name="risk aversion",
        rubric="<rubric text>",
        question_text="<question text>",
    )
    assert "risk aversion" in prompt
    assert "<rubric text>" in prompt
    assert "<question text>" in prompt


def test_build_annotation_prompt_asks_for_bracketed_range_format():
    prompt = build_annotation_prompt("RA", "rubric", "question")
    assert "propensity range is [LOWER BOUND, UPPER BOUND]" in prompt


def test_annotation_system_prompt_is_nonempty_string():
    assert isinstance(ANNOTATION_SYSTEM_PROMPT, str)
    assert len(ANNOTATION_SYSTEM_PROMPT) > 0
