"""Backend-agnostic response post-processing: whitespace cleanup, reasoning-
trace stripping, and final-answer extraction. Ported from
the original infer-propensities-jsonl-robust.py.
"""

import re

FINAL_ANSWER_SUFFIX = """
Your final answer MUST include the full option text. Use this exact format:
"Final answer: [letter] - [full option text]"
""".strip()

FINAL_ANSWER_REGEX = r"[Ff]inal\s*[Aa]nswer:\s*(.+)"

_FINAL_ANSWER_PATTERN = re.compile(FINAL_ANSWER_REGEX)
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?(</think>|<\\think>)", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED_PATTERN = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalizes whitespace only. Does NOT remove any content."""
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def strip_thinking(text: str) -> str:
    """Extracts non-thinking text by removing <think>...</think> blocks.
    Also tolerates a malformed closing tag <\\think>, case-insensitive. If an
    opening <think> exists without a close, everything after it is dropped."""
    if not text:
        return ""
    text = _THINK_BLOCK_PATTERN.sub("", text)
    text = _THINK_UNCLOSED_PATTERN.sub("", text)
    return text.strip()


def extract_final_answer(text: str) -> str:
    """Extracts the final answer string via FINAL_ANSWER_REGEX. Returns '' if missing."""
    if not text:
        return ""
    match = _FINAL_ANSWER_PATTERN.search(text)
    return match.group(1).strip() if match else ""
