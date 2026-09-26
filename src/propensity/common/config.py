"""Per-sector YAML config loading.

Replaces the old, unused src/config_use.py demo pattern (which hardcoded a
single config/default.yaml path and never actually resolved ${VAR} placeholders
in the YAML). load_config(sector) reads config/{sector}.yaml, loads .env, and
substitutes any string value of the exact form "${VAR_NAME}" with os.environ["VAR_NAME"].
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_VAR_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

_dotenv_loaded = False


def _resolve_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    if isinstance(value, str):
        match = _VAR_PATTERN.match(value)
        if match:
            return os.environ.get(match.group(1))
    return value


def load_config(sector: str, config_dir: str = "config") -> dict:
    """Loads config/{sector}.yaml, resolving ${VAR_NAME} placeholders against the
    environment (after loading .env, once per process).

    Args:
        sector: one of "generation", "annotation", "inference", "surfaces",
            "predictability" (i.e. the file stem under config_dir).
        config_dir: directory containing the per-sector YAML files.

    Returns:
        The parsed config dict, with ${VAR_NAME} strings substituted.
    """
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv()
        _dotenv_loaded = True

    path = Path(config_dir) / f"{sector}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _resolve_env_vars(raw)
