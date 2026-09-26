"""Extracts the annotated [lower, upper] propensity range from a free-text
annotation response. Ported verbatim from the original
annotate_benchmark_azure.py.
"""

import re

_RANGE_PATTERN = re.compile(r"\[([+-]?\d+),\s*([+-]?\d+)\]")


def parse_propensity_range(response_text: str) -> tuple[int, int]:
    """Extracts the propensity range from an annotation response.

    Expected format: 'The propensity range is [LB, UB]'
    Handles optional +/- signs: [-1, +2] or [+1, +3]
    """
    match = _RANGE_PATTERN.search(response_text)
    if not match:
        raise ValueError(f"Could not parse propensity range from: {response_text}")
    return int(match.group(1)), int(match.group(2))
