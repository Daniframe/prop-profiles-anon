import pytest

from src.propensity.inference.runner import load_prompts, run_inference


class FakeBackend:
    """Returns canned responses in call order -- no real backend involved."""

    def __init__(self, responses_by_call):
        self._responses_by_call = list(responses_by_call)
        self.calls = []

    def generate_batch(self, chats):
        self.calls.append(chats)
        return self._responses_by_call.pop(0)


def _questions():
    return [
        {"question_id": "RA_0", "question_text": "Q0"},
        {"question_id": "RA_1", "question_text": "Q1"},
    ]


def test_load_prompts_reads_txt_and_md_files(tmp_path):
    (tmp_path / "0.txt").write_text("neutral prompt")
    (tmp_path / "baseline.md").write_text("baseline prompt")
    (tmp_path / "notes.json").write_text("{}")  # ignored, wrong extension

    prompts = load_prompts(tmp_path)

    assert prompts == {"0": "neutral prompt", "baseline": "baseline prompt"}


def test_load_prompts_raises_if_directory_has_no_prompt_files(tmp_path):
    with pytest.raises(ValueError):
        load_prompts(tmp_path)


def test_run_inference_builds_one_chat_per_question_per_prompt():
    backend = FakeBackend([["Final answer: A", "Final answer: B"]])
    run_inference(backend, _questions(), {"0": "neutral system prompt"})

    chats = backend.calls[0]
    assert len(chats) == 2
    assert chats[0][0] == {"role": "system", "content": "neutral system prompt\n\nYour final answer MUST include the full option text. Use this exact format:\n\"Final answer: [letter] - [full option text]\""}
    assert chats[0][1] == {"role": "user", "content": "Q0"}


def test_run_inference_extracts_final_answer_and_preserves_original_fields():
    backend = FakeBackend([["blah Final answer: A - safe option", "blah Final answer: B - risky option"]])
    result = run_inference(backend, _questions(), {"0": "system prompt"})

    responses = result["0"]["responses"]
    assert result["0"]["total"] == 2
    assert responses[0]["question_id"] == "RA_0"
    assert responses[0]["final_answer"] == "A - safe option"
    assert responses[0]["non_think"] == "NOT_VALID"


def test_run_inference_handles_thinking_models():
    raw = "<think>reasoning here</think>Final answer: A"
    backend = FakeBackend([[raw, raw]])
    result = run_inference(backend, _questions(), {"0": "system prompt"}, is_thinking_model=True)

    responses = result["0"]["responses"]
    assert responses[0]["non_think"] == "Final answer: A"
    assert responses[0]["final_answer"] == "A"

    # system prompt sent to the backend should end with "<think>"
    chats = backend.calls[0]
    assert chats[0][0]["content"].endswith("<think>")


def test_run_inference_runs_once_per_prompt():
    backend = FakeBackend([["r0a", "r0b"], ["r1a", "r1b"]])
    result = run_inference(backend, _questions(), {"promptA": "sysA", "promptB": "sysB"})

    assert set(result.keys()) == {"promptA", "promptB"}
    assert len(backend.calls) == 2
