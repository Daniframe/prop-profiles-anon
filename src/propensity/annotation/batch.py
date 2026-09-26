"""Batch-mode LLM annotation via the OpenAI/Azure Batch API (Chat Completions
endpoint). Generalizes the original annotate_benchmark_azure.py's
submit/status/process flow to a duck-typed client instead of a hardcoded AzureOpenAI.
"""

import json
import logging
from pathlib import Path
from typing import Union

from .prompts import ANNOTATION_SYSTEM_PROMPT, build_annotation_prompt
from .parsing import parse_propensity_range


def build_batch_requests(questions: list[dict], rubric: str, propensity_name: str,
                          model: str, temperature: float = 0.0) -> list[dict]:
    """Builds one Batch API request per question, keyed by question_id."""
    requests = []
    for q in questions:
        prompt = build_annotation_prompt(propensity_name, rubric, q["question_text"])
        requests.append({
            "custom_id": q["question_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            },
        })
    return requests


def submit_batch(client, requests: list[dict], batch_input_path: Union[str, Path]) -> str:
    """Writes `requests` to batch_input_path, uploads it, and submits a Batch
    API job on /v1/chat/completions. Returns the batch id."""
    batch_input_path = Path(batch_input_path)
    batch_input_path.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_input_path, "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")

    with open(batch_input_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id


def check_batch_status(client, batch_id: str) -> dict:
    """Returns a small status summary dict for a submitted batch."""
    batch = client.batches.retrieve(batch_id)
    return {
        "batch_id": batch.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "request_counts": batch.request_counts.model_dump() if batch.request_counts else None,
        "output_file_id": getattr(batch, "output_file_id", None),
    }


def process_batch(client, batch_id: str, questions: list[dict]) -> list[dict]:
    """Retrieves a completed batch's results and merges them back onto the
    original questions (matched by question_id/custom_id). Skips (and logs)
    any question whose response is missing or failed to parse."""
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise ValueError(f"Batch {batch_id} is not completed. Status: {batch.status}")
    if not batch.output_file_id:
        raise ValueError(f"Batch {batch_id} has no output file")

    result_content = client.files.content(batch.output_file_id).text

    raw_responses = {}
    parsed_ranges = {}
    for line in result_content.strip().split("\n"):
        if not line:
            continue
        result = json.loads(line)
        question_id = result["custom_id"]
        response_text = result["response"]["body"]["choices"][0]["message"]["content"]
        raw_responses[question_id] = response_text
        try:
            parsed_ranges[question_id] = parse_propensity_range(response_text)
        except ValueError as ex:
            logging.warning(f"Failed to parse {question_id}: {ex}")

    annotated = []
    for q in questions:
        qid = q["question_id"]
        if qid not in raw_responses:
            logging.warning(f"No annotation found for {qid}")
            continue
        if qid not in parsed_ranges:
            continue  # parse failure already logged above
        lower, upper = parsed_ranges[qid]
        annotated.append({
            **q,
            "propensity_lower": lower,
            "propensity_upper": upper,
            "raw_annotation_response": raw_responses[qid],
        })
    return annotated
