"""Sector 2: rubric-based demand-interval annotation (paper stage 2).

- rubrics: canonical rubric path resolution + loading.
- prompts: the annotation prompt (system + rubric-following chain-of-thought template).
- parsing: extracts [lower, upper] from a free-text annotation response.
- sequential: one-by-one LLM annotation.
- batch: OpenAI/Azure Batch API submit/status/process annotation.

All functions take a duck-typed client (see src/propensity/common/llm_clients.py)
-- nothing here is hardcoded to AzureOpenAI.
"""

from .rubrics import rubric_path_for, load_rubric_text
from .prompts import ANNOTATION_SYSTEM_PROMPT, build_annotation_prompt
from .parsing import parse_propensity_range
from .sequential import annotate_question, annotate_sequential
from .batch import build_batch_requests, submit_batch, check_batch_status, process_batch

__all__ = [
    "rubric_path_for",
    "load_rubric_text",
    "ANNOTATION_SYSTEM_PROMPT",
    "build_annotation_prompt",
    "parse_propensity_range",
    "annotate_question",
    "annotate_sequential",
    "build_batch_requests",
    "submit_batch",
    "check_batch_status",
    "process_batch",
]
