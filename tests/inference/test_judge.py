import json
from types import SimpleNamespace

import pytest

from src.propensity.inference.judge import (
    build_judge_prompt,
    parse_judgment,
    judge_item,
    judge_sequential,
    build_judge_batch_requests,
    submit_judge_batch,
    check_judge_batch_status,
    process_judge_batch,
)


def test_build_judge_prompt_fills_placeholders():
    prompt = build_judge_prompt("What year?", "1990", "It was 1990")
    assert "What year?" in prompt
    assert "1990" in prompt
    assert "It was 1990" in prompt


@pytest.mark.parametrize("text,expected", [
    ("CORRECT", True),
    ("correct", True),
    ("Correct.", True),
    ("INCORRECT", False),
    ("incorrect", False),
    ("Incorrect, the model missed the date.", False),
])
def test_parse_judgment(text, expected):
    assert parse_judgment(text) == expected


def test_parse_judgment_raises_on_unparseable_text():
    with pytest.raises(ValueError):
        parse_judgment("I'm not sure.")


class FakeChatClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        outer = self

        class _Completions:
            @staticmethod
            def create(model, messages, temperature):
                outer.calls.append(messages)
                qid = list(outer.responses)[len(outer.calls) - 1]
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outer.responses[qid]))])

        self.chat = SimpleNamespace(completions=_Completions())


def _item(qid="TQ_0"):
    return {
        "question_id": qid,
        "question_text": "What year did X happen?",
        "correct_answer": "1990",
        "raw_response": "The answer is 1990.",
    }


def test_judge_item_merges_judgment_onto_item():
    client = FakeChatClient({"TQ_0": "CORRECT"})
    result = judge_item(client, _item(), model="gpt-4.1")

    assert result["question_id"] == "TQ_0"
    assert result["judgment"] is True
    assert result["raw_judgment_response"] == "CORRECT"


def test_judge_sequential_skips_unparseable_and_continues():
    client = FakeChatClient({"TQ_0": "CORRECT", "TQ_1": "not sure", "TQ_2": "INCORRECT"})
    items = [_item("TQ_0"), _item("TQ_1"), _item("TQ_2")]

    graded = judge_sequential(items, client, model="gpt-4.1")

    assert [g["question_id"] for g in graded] == ["TQ_0", "TQ_2"]
    assert graded[0]["judgment"] is True
    assert graded[1]["judgment"] is False


def test_build_judge_batch_requests_one_per_item():
    items = [_item("TQ_0"), _item("TQ_1")]
    requests = build_judge_batch_requests(items, model="gpt-4.1", temperature=0.2)

    assert [r["custom_id"] for r in requests] == ["TQ_0", "TQ_1"]
    assert all(r["url"] == "/v1/chat/completions" for r in requests)
    assert all(r["body"]["temperature"] == 0.2 for r in requests)


class FakeBatchClient:
    def __init__(self):
        self._files = {}
        self._batches = {}
        self._file_counter = 0
        self._batch_counter = 0
        outer = self

        class _Files:
            @staticmethod
            def create(file, purpose):
                outer._file_counter += 1
                file_id = f"file_{outer._file_counter}"
                outer._files[file_id] = file.read()
                return SimpleNamespace(id=file_id)

            @staticmethod
            def content(file_id):
                return SimpleNamespace(text=outer._files[file_id])

        class _Batches:
            @staticmethod
            def create(input_file_id, endpoint, completion_window):
                outer._batch_counter += 1
                batch_id = f"batch_{outer._batch_counter}"
                outer._batches[batch_id] = SimpleNamespace(
                    id=batch_id, status="in_progress", output_file_id=None,
                    created_at=0, request_counts=None,
                )
                return outer._batches[batch_id]

            @staticmethod
            def retrieve(batch_id):
                return outer._batches[batch_id]

        self.files = _Files()
        self.batches = _Batches()

    def upload_output(self, records):
        self._file_counter += 1
        file_id = f"file_{self._file_counter}"
        self._files[file_id] = "\n".join(json.dumps(r) for r in records)
        return file_id

    def mark_completed(self, batch_id, output_file_id):
        self._batches[batch_id].status = "completed"
        self._batches[batch_id].output_file_id = output_file_id


def _judge_completion_record(custom_id, content):
    return {"custom_id": custom_id, "response": {"body": {"choices": [{"message": {"content": content}}]}}}


def test_submit_and_check_status(tmp_path):
    client = FakeBatchClient()
    requests = build_judge_batch_requests([_item("TQ_0")], model="gpt-4.1")
    batch_id = submit_judge_batch(client, requests, tmp_path / "in.jsonl")

    status = check_judge_batch_status(client, batch_id)
    assert status["batch_id"] == batch_id
    assert status["status"] == "in_progress"


def test_process_judge_batch_merges_judgments(tmp_path):
    client = FakeBatchClient()
    items = [_item("TQ_0"), _item("TQ_1")]
    batch_id = submit_judge_batch(client, build_judge_batch_requests(items, "gpt-4.1"), tmp_path / "in.jsonl")
    output_file_id = client.upload_output([
        _judge_completion_record("TQ_0", "CORRECT"),
        _judge_completion_record("TQ_1", "INCORRECT"),
    ])
    client.mark_completed(batch_id, output_file_id)

    graded = process_judge_batch(client, batch_id, items)

    assert graded[0]["judgment"] is True
    assert graded[1]["judgment"] is False
    assert graded[0]["question_text"] == items[0]["question_text"]


def test_process_judge_batch_raises_if_not_completed(tmp_path):
    client = FakeBatchClient()
    batch_id = submit_judge_batch(client, [], tmp_path / "in.jsonl")

    with pytest.raises(ValueError):
        process_judge_batch(client, batch_id, [])
