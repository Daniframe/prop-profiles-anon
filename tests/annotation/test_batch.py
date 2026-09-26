import json
from types import SimpleNamespace

import pytest

from src.propensity.annotation.batch import (
    build_batch_requests,
    submit_batch,
    check_batch_status,
    process_batch,
)


class FakeBatchClient:
    """Mocks client.files.{create,content} and client.batches.{create,retrieve}
    -- no real network call."""

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
                    id=batch_id, status="in_progress", endpoint=endpoint,
                    input_file_id=input_file_id, output_file_id=None,
                    created_at=1234, request_counts=None,
                )
                return outer._batches[batch_id]

            @staticmethod
            def retrieve(batch_id):
                return outer._batches[batch_id]

        self.files = _Files()
        self.batches = _Batches()

    def upload_output(self, records: list[dict]) -> str:
        """Test helper: registers a fake completed-batch output file."""
        self._file_counter += 1
        file_id = f"file_{self._file_counter}"
        self._files[file_id] = "\n".join(json.dumps(r) for r in records)
        return file_id

    def mark_completed(self, batch_id: str, output_file_id: str):
        self._batches[batch_id].status = "completed"
        self._batches[batch_id].output_file_id = output_file_id


def _chat_completion_record(custom_id: str, content: str) -> dict:
    return {
        "custom_id": custom_id,
        "response": {"body": {"choices": [{"message": {"content": content}}]}},
    }


def test_build_batch_requests_one_per_question():
    questions = [{"question_id": "RA_0", "question_text": "Q0"}, {"question_id": "RA_1", "question_text": "Q1"}]
    requests = build_batch_requests(questions, rubric="<rubric>", propensity_name="RA", model="gpt-4.1", temperature=0.1)

    assert [r["custom_id"] for r in requests] == ["RA_0", "RA_1"]
    assert all(r["url"] == "/v1/chat/completions" for r in requests)
    assert all(r["body"]["model"] == "gpt-4.1" for r in requests)
    assert all(r["body"]["temperature"] == 0.1 for r in requests)


def test_submit_batch_uploads_requests_and_creates_batch(tmp_path):
    client = FakeBatchClient()
    requests = build_batch_requests(
        [{"question_id": "RA_0", "question_text": "Q0"}], rubric="<rubric>", propensity_name="RA", model="gpt-4.1"
    )
    batch_id = submit_batch(client, requests, tmp_path / "batch_input.jsonl")

    assert batch_id == "batch_1"
    assert (tmp_path / "batch_input.jsonl").exists()
    written = [json.loads(line) for line in (tmp_path / "batch_input.jsonl").read_text().splitlines()]
    assert written == requests


def test_check_batch_status_reports_current_state(tmp_path):
    client = FakeBatchClient()
    batch_id = submit_batch(client, [], tmp_path / "batch_input.jsonl")

    status = check_batch_status(client, batch_id)

    assert status["batch_id"] == batch_id
    assert status["status"] == "in_progress"


def test_process_batch_merges_results_onto_original_questions(tmp_path):
    client = FakeBatchClient()
    questions = [
        {"question_id": "RA_0", "question_text": "Q0"},
        {"question_id": "RA_1", "question_text": "Q1"},
    ]
    batch_id = submit_batch(client, build_batch_requests(questions, "<rubric>", "RA", "gpt-4.1"), tmp_path / "in.jsonl")
    output_file_id = client.upload_output([
        _chat_completion_record("RA_0", "The propensity range is [-1, +2]"),
        _chat_completion_record("RA_1", "The propensity range is [0, 0]"),
    ])
    client.mark_completed(batch_id, output_file_id)

    annotated = process_batch(client, batch_id, questions)

    assert len(annotated) == 2
    assert annotated[0]["propensity_lower"] == -1 and annotated[0]["propensity_upper"] == 2
    assert annotated[1]["propensity_lower"] == 0 and annotated[1]["propensity_upper"] == 0
    assert annotated[0]["question_text"] == "Q0"  # original fields preserved


def test_process_batch_skips_missing_and_unparseable_responses(tmp_path):
    client = FakeBatchClient()
    questions = [
        {"question_id": "RA_0", "question_text": "Q0"},
        {"question_id": "RA_1", "question_text": "Q1"},
        {"question_id": "RA_2", "question_text": "Q2"},
    ]
    batch_id = submit_batch(client, build_batch_requests(questions, "<rubric>", "RA", "gpt-4.1"), tmp_path / "in.jsonl")
    output_file_id = client.upload_output([
        _chat_completion_record("RA_0", "The propensity range is [-1, +2]"),
        _chat_completion_record("RA_1", "I could not determine a range."),
        # RA_2 has no response at all (e.g. dropped from the batch output)
    ])
    client.mark_completed(batch_id, output_file_id)

    annotated = process_batch(client, batch_id, questions)

    assert [a["question_id"] for a in annotated] == ["RA_0"]


def test_process_batch_raises_if_not_completed(tmp_path):
    client = FakeBatchClient()
    batch_id = submit_batch(client, [], tmp_path / "in.jsonl")

    with pytest.raises(ValueError):
        process_batch(client, batch_id, [])
