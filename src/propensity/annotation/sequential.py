"""One-by-one (non-batch) LLM annotation. `client` is duck-typed -- any object
exposing `chat.completions.create(...)`, e.g. common.llm_clients.get_azure_client()
or get_openai_client() -- there is no Azure-specific code here.
"""

import logging

from .prompts import ANNOTATION_SYSTEM_PROMPT, build_annotation_prompt
from .parsing import parse_propensity_range


def annotate_question(client, question: dict, rubric: str, propensity_name: str,
                       model: str, temperature: float = 0.0) -> dict:
    """Annotates a single question, returning it merged with
    propensity_lower/propensity_upper/raw_annotation_response -- ready for
    surfaces.data.load_demands_single_dim."""
    prompt = build_annotation_prompt(propensity_name, rubric, question["question_text"])

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    response_text = response.choices[0].message.content
    if not response_text:
        raise ValueError(f"Empty response for question {question['question_id']}")

    lower, upper = parse_propensity_range(response_text)

    return {
        **question,
        "propensity_lower": lower,
        "propensity_upper": upper,
        "raw_annotation_response": response_text,
    }


def annotate_sequential(questions: list[dict], rubric: str, propensity_name: str,
                         client, model: str, temperature: float = 0.0) -> list[dict]:
    """Annotates each question one at a time; skips (and logs) any that fail
    (empty response, unparseable range, API error)."""
    annotated = []
    for question in questions:
        try:
            annotated.append(
                annotate_question(client, question, rubric, propensity_name, model, temperature)
            )
        except Exception as ex:
            logging.warning(f"Failed to annotate {question.get('question_id')}: {ex}")
            continue
    return annotated
