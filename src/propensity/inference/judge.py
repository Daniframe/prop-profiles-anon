"""LLM-as-judge grading of open-ended model answers against a gold answer
(e.g. for TimeMenatQA, which has no multiple-choice options to string-match
against). Ported from the original parse_inference_llmjudge.py.

`client` is duck-typed throughout, same convention as generation/annotation --
nothing here is hardcoded to AzureOpenAI.
"""

import json
import logging
from pathlib import Path
from typing import Union

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a question-answering task.

You will be given:
- a question
- a gold (correct) answer
- a model's answer

Your task:
Decide whether the model's answer is semantically equivalent to the gold answer
based ONLY on the context.

Rules:
- The model answer must express the same fact as the gold answer.
- Extra words are allowed if the core answer is correct.
- If the model answer is incorrect, incomplete, or answers a different question,
  it is NOT correct.
- If the answer is not supported by the context, it is NOT correct.

Output format (follow exactly):
CORRECT or INCORRECT"""

JUDGE_PROMPT_TEMPLATE = """
Question: {question}

Gold answer: {gold_answer}

Model answer: {model_answer}
"""


def build_judge_prompt(question: str, gold_answer: str, model_answer: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(question=question, gold_answer=gold_answer, model_answer=model_answer)


def parse_judgment(response_text: str) -> bool:
    """Returns True for CORRECT, False for INCORRECT."""
    normalized = response_text.strip().upper()
    if normalized.startswith("INCORRECT"):
        return False
    if normalized.startswith("CORRECT"):
        return True
    raise ValueError(f"Could not parse judgment from: {response_text}")


# --- sequential -------------------------------------------------------------

def judge_item(client, item: dict, model: str, temperature: float = 0.0, *,
                question_field: str = "question_text", gold_field: str = "correct_answer",
                answer_field: str = "raw_response") -> dict:
    """Grades one item, returning it merged with judgment (bool) and
    raw_judgment_response."""
    prompt = build_judge_prompt(item[question_field], item[gold_field], item[answer_field])

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    response_text = response.choices[0].message.content
    if not response_text:
        raise ValueError(f"Empty judge response for item {item.get('question_id')}")

    return {
        **item,
        "judgment": parse_judgment(response_text),
        "raw_judgment_response": response_text,
    }


def judge_sequential(items: list[dict], client, model: str, temperature: float = 0.0, **field_kwargs) -> list[dict]:
    """Grades each item one at a time; skips (and logs) any that fail."""
    graded = []
    for item in items:
        try:
            graded.append(judge_item(client, item, model, temperature, **field_kwargs))
        except Exception as ex:
            logging.warning(f"Failed to judge {item.get('question_id')}: {ex}")
            continue
    return graded


# --- batch --------------------------------------------------------------

def build_judge_batch_requests(items: list[dict], model: str, temperature: float = 0.0, *,
                                question_field: str = "question_text", gold_field: str = "correct_answer",
                                answer_field: str = "raw_response") -> list[dict]:
    """Builds one Batch API request per item, keyed by question_id."""
    requests = []
    for item in items:
        prompt = build_judge_prompt(item[question_field], item[gold_field], item[answer_field])
        requests.append({
            "custom_id": item["question_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            },
        })
    return requests


def submit_judge_batch(client, requests: list[dict], batch_input_path: Union[str, Path]) -> str:
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


def check_judge_batch_status(client, batch_id: str) -> dict:
    """Returns a small status summary dict for a submitted judge batch."""
    batch = client.batches.retrieve(batch_id)
    return {
        "batch_id": batch.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "request_counts": batch.request_counts.model_dump() if batch.request_counts else None,
        "output_file_id": getattr(batch, "output_file_id", None),
    }


def process_judge_batch(client, batch_id: str, items: list[dict]) -> list[dict]:
    """Retrieves a completed judge batch's results and merges them back onto
    the original items (matched by question_id/custom_id). Skips (and logs)
    any item whose response is missing or failed to parse."""
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise ValueError(f"Batch {batch_id} is not completed. Status: {batch.status}")
    if not batch.output_file_id:
        raise ValueError(f"Batch {batch_id} has no output file")

    result_content = client.files.content(batch.output_file_id).text

    raw_responses = {}
    judgments = {}
    for line in result_content.strip().split("\n"):
        if not line:
            continue
        result = json.loads(line)
        question_id = result["custom_id"]
        response_text = result["response"]["body"]["choices"][0]["message"]["content"]
        raw_responses[question_id] = response_text
        try:
            judgments[question_id] = parse_judgment(response_text)
        except ValueError as ex:
            logging.warning(f"Failed to parse judgment for {question_id}: {ex}")

    graded = []
    for item in items:
        qid = item["question_id"]
        if qid not in raw_responses:
            logging.warning(f"No judgment found for {qid}")
            continue
        if qid not in judgments:
            continue  # parse failure already logged above
        graded.append({
            **item,
            "judgment": judgments[qid],
            "raw_judgment_response": raw_responses[qid],
        })
    return graded
