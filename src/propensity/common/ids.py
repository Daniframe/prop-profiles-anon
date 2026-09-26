"""Reproducible IDs with embedded seeds. Ported verbatim from the original id_utils.py."""

import secrets
from datetime import date


def generate_id(hex_seed: str | None = None) -> str:
    """Generate a unique ID: YYYY-MM-DD-XXXX (4 hex chars representing the seed).

    The hex suffix IS the random seed, allowing reproduction of results.

    Args:
        hex_seed: Optional seed to use in hex. If None, generates a random 16-bit seed.

    Returns:
        ID in format YYYY-MM-DD-XXXX
    """
    today = date.today().isoformat()
    if hex_seed is None:
        seed = secrets.randbelow(65536)  # 16-bit: 0-65535
        hex_seed = f"{seed:04x}"
    return f"{today}-{hex_seed}"


def seed_from_id(id_string: str) -> int:
    """Extract the random seed from an ID.

    Args:
        id_string: ID in format YYYY-MM-DD-XXXX

    Returns:
        The seed as an integer
    """
    return int(hex_seed_from_id(id_string), 16)


def hex_seed_from_id(id_string: str) -> str:
    """Extract the random hex seed from an ID.

    Args:
        id_string: ID in format YYYY-MM-DD-XXXX

    Returns:
        The XXXX hex seed
    """
    hex_suffix = id_string.split('-')[-1]
    return hex_suffix
