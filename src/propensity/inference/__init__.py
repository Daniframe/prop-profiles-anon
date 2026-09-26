"""Sector 3: instance-level model outcomes across incitement levels
(paper stage 3).

- backends: InferenceBackend implementations (VLLMBackend, OpenAICompatibleBackend).
- models: vLLM model-shortcut registry.
- extraction: response post-processing (cleanup, <think> stripping, final-answer regex).
- runner: the shared prompt x question inference loop, backend-agnostic.
- judge: LLM-as-judge grading of open-ended answers.
- reunite: a small helper toolkit for turning results into outcome tables --
  not an opinionated pipeline, see reunite.py's module docstring.
"""

from .backends import InferenceBackend, VLLMBackend, OpenAICompatibleBackend
from .models import MODEL_SHORTCUTS, THINKING_MODELS, resolve_model
from .extraction import clean_text, strip_thinking, extract_final_answer, FINAL_ANSWER_SUFFIX
from .runner import load_prompts, run_inference
from .judge import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
    parse_judgment,
    judge_item,
    judge_sequential,
    build_judge_batch_requests,
    submit_judge_batch,
    check_judge_batch_status,
    process_judge_batch,
)
from .reunite import load_results, exact_match, judgment_field, score_responses, build_outcomes_table

__all__ = [
    "InferenceBackend",
    "VLLMBackend",
    "OpenAICompatibleBackend",
    "MODEL_SHORTCUTS",
    "THINKING_MODELS",
    "resolve_model",
    "clean_text",
    "strip_thinking",
    "extract_final_answer",
    "FINAL_ANSWER_SUFFIX",
    "load_prompts",
    "run_inference",
    "JUDGE_SYSTEM_PROMPT",
    "build_judge_prompt",
    "parse_judgment",
    "judge_item",
    "judge_sequential",
    "build_judge_batch_requests",
    "submit_judge_batch",
    "check_judge_batch_status",
    "process_judge_batch",
    "load_results",
    "exact_match",
    "judgment_field",
    "score_responses",
    "build_outcomes_table",
]
