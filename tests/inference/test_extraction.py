from src.propensity.inference.extraction import clean_text, strip_thinking, extract_final_answer


def test_clean_text_normalizes_whitespace():
    assert clean_text("  hello \n  world  ") == "hello world"


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_strip_thinking_removes_closed_think_block():
    text = "<think>internal reasoning</think>Final answer: A"
    assert strip_thinking(text) == "Final answer: A"


def test_strip_thinking_is_case_insensitive():
    text = "<THINK>internal</THINK>Final answer: A"
    assert strip_thinking(text) == "Final answer: A"


def test_strip_thinking_tolerates_malformed_closing_tag():
    text = "<think>internal<\\think>Final answer: A"
    assert strip_thinking(text) == "Final answer: A"


def test_strip_thinking_drops_everything_after_unclosed_tag():
    text = "<think>internal reasoning that never closes"
    assert strip_thinking(text) == ""


def test_strip_thinking_handles_no_think_block():
    assert strip_thinking("Final answer: A") == "Final answer: A"


def test_extract_final_answer_extracts_after_marker():
    assert extract_final_answer("blah blah Final answer: B - the second option") == "B - the second option"


def test_extract_final_answer_case_insensitive_marker():
    # the regex only varies the leading letter's case (matches original
    # r"[Ff]inal\s*[Aa]nswer:" exactly) -- "Final answer:" / "final Answer:" etc.
    assert extract_final_answer("final Answer:   C") == "C"


def test_extract_final_answer_returns_empty_when_missing():
    assert extract_final_answer("no marker here") == ""


def test_extract_final_answer_returns_empty_for_empty_input():
    assert extract_final_answer("") == ""
