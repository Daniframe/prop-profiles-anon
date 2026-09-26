"""Cross-sector utilities shared by generation/annotation/inference/surfaces/predictability.

- ids: reproducible YYYY-MM-DD-XXXX ID/seed scheme.
- config: per-sector YAML config loading with .env / ${VAR} resolution.
- io: JSONL/CSV read-write helpers shared across sectors.
- llm_clients: OpenAI/Azure OpenAI client factory.
"""

from .ids import generate_id, seed_from_id, hex_seed_from_id
from .config import load_config
from .io import read_table, read_jsonl, write_jsonl

__all__ = [
    "generate_id",
    "seed_from_id",
    "hex_seed_from_id",
    "load_config",
    "read_table",
    "read_jsonl",
    "write_jsonl",
]
