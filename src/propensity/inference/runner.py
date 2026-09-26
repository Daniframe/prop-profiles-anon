"""The one inference loop shared by every backend -- factors out what was
~70% duplicated between the original infer-propensities-jsonl-robust.py
(vLLM) and openai-run-models.py (Azure): build chats for a prompt x question
batch, generate, extract the final answer, merge onto the original question,
aggregate per prompt.
"""

from pathlib import Path
from typing import Union

from .extraction import clean_text, strip_thinking, extract_final_answer, FINAL_ANSWER_SUFFIX


def load_prompts(prompts_dir: Union[str, Path]) -> dict[str, str]:
    """Loads every .txt/.md file in prompts_dir into {stem: text}, e.g.
    prompts/incitement/RA/ -> {"baseline": "...", "-3": "...", "0": "...", ...}."""
    prompts = {}
    for path in sorted(Path(prompts_dir).glob("*")):
        if path.suffix.lower() in (".txt", ".md"):
            prompts[path.stem] = path.read_text(encoding="utf-8").strip()
    if not prompts:
        raise ValueError(f"No prompt files found in {prompts_dir}")
    return prompts


def run_inference(
    backend,
    questions: list[dict],
    prompts: dict[str, str],
    question_field: str = "question_text",
    is_thinking_model: bool = False,
    final_answer_suffix: str = FINAL_ANSWER_SUFFIX,
) -> dict:
    """Runs every prompt x every question through `backend` (anything
    implementing InferenceBackend.generate_batch).

    prompts: {prompt_name: system_prompt_text}, e.g. loaded from
        prompts/incitement/{code}/*.txt.
    is_thinking_model: if True, the system prompt is forced to end with
        "<think>" and non_think is computed by stripping <think>...</think>
        traces from the raw response before extracting the final answer --
        this applies to any backend that might emit reasoning traces, not
        just vLLM-hosted DeepSeek-R1-style models.

    Returns {prompt_name: {"total": N, "responses": [...]}}, where each
    response is the original question dict plus idx/raw_response/non_think/
    final_answer -- the same shape both original scripts wrote to their
    results JSON.
    """
    per_prompt = {}

    for name, sys_prompt in prompts.items():
        system_prompt = (sys_prompt + "\n\n" + final_answer_suffix).strip()
        if is_thinking_model:
            system_prompt = system_prompt + "\n\n<think>"

        chats = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item[question_field]},
            ]
            for item in questions
        ]

        raw_responses = backend.generate_batch(chats)

        responses = []
        for i, raw_response in enumerate(raw_responses):
            raw_response = clean_text(raw_response)

            if is_thinking_model:
                non_think = strip_thinking(raw_response)
                final_answer = extract_final_answer(non_think)
            else:
                non_think = "NOT_VALID"
                final_answer = extract_final_answer(raw_response)

            responses.append({
                **questions[i],
                "idx": i,
                "raw_response": raw_response,
                "non_think": non_think,
                "final_answer": final_answer,
            })

        per_prompt[name] = {"total": len(questions), "responses": responses}

    return per_prompt
