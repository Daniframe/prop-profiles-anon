from types import SimpleNamespace

from src.propensity.annotation.sequential import annotate_question, annotate_sequential


class FakeChatClient:
    """Mocks client.chat.completions.create(...) -- no real network call."""

    def __init__(self, responses):
        self.responses = responses  # dict question_id -> response text, or Exception to raise
        self.calls = []

        outer = self

        class _Completions:
            @staticmethod
            def create(model, messages, temperature):
                outer.calls.append({"model": model, "messages": messages, "temperature": temperature})
                # question_id is not passed directly; recover it via call order
                question_id = list(outer.responses)[len(outer.calls) - 1]
                result = outer.responses[question_id]
                if isinstance(result, Exception):
                    raise result
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=result))])

        self.chat = SimpleNamespace(completions=_Completions())


def _question(qid="RA_0"):
    return {"question_id": qid, "question_text": "Pick an option.", "correct_answer": "Option A"}


def test_annotate_question_parses_range_and_merges_onto_question():
    client = FakeChatClient({"RA_0": "The propensity range is [-1, +2]"})
    result = annotate_question(client, _question(), rubric="<rubric>", propensity_name="RA", model="gpt-4.1")

    assert result["question_id"] == "RA_0"
    assert result["question_text"] == "Pick an option."
    assert result["propensity_lower"] == -1
    assert result["propensity_upper"] == 2
    assert result["raw_annotation_response"] == "The propensity range is [-1, +2]"


def test_annotate_question_sends_correct_model_and_temperature():
    client = FakeChatClient({"RA_0": "The propensity range is [0, 0]"})
    annotate_question(client, _question(), rubric="<rubric>", propensity_name="RA", model="gpt-4.1", temperature=0.3)

    assert client.calls[0]["model"] == "gpt-4.1"
    assert client.calls[0]["temperature"] == 0.3


def test_annotate_sequential_skips_and_continues_on_failure():
    client = FakeChatClient({
        "RA_0": "The propensity range is [-1, +1]",
        "RA_1": ValueError("simulated API error"),
        "RA_2": "The propensity range is [0, +2]",
    })
    questions = [_question("RA_0"), _question("RA_1"), _question("RA_2")]

    annotated = annotate_sequential(questions, rubric="<rubric>", propensity_name="RA", client=client, model="gpt-4.1")

    assert [a["question_id"] for a in annotated] == ["RA_0", "RA_2"]


def test_annotate_sequential_skips_unparseable_response():
    client = FakeChatClient({"RA_0": "I am not sure what the range is."})
    annotated = annotate_sequential([_question("RA_0")], rubric="<rubric>", propensity_name="RA", client=client, model="gpt-4.1")

    assert annotated == []
