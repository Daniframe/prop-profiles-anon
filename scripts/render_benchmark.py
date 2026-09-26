"""Standalone CLI: renders an existing context-grounded seed file (JSON/JSONL,
as produced by generate_context_seeds or checked in under data/benchmarks/raw/)
into a ready/*.jsonl multiple choice question set, without calling Azure.

Useful to re-render with a different --hex-seed (different option shuffle) or
to render a pre-generated seed file directly (e.g. Ul, see benchmarks/ul.yaml).
generate_benchmark.py already does this as its last step for context_grounded
specs -- use this script when you only have a seed file and no spec.
"""

import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.propensity.common import generate_id, seed_from_id, hex_seed_from_id, write_jsonl, read_jsonl
from src.propensity.generation import render_seed_questions


def load_seeds(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        return list(read_jsonl(path))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Render a seed file into a ready multiple choice question set")
    parser.add_argument("--seed-file", type=str, required=True)
    parser.add_argument("--benchmark-code", type=str, required=True, help="e.g. Ex, Ul")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--hex-seed", type=str, default=None)
    args = parser.parse_args()

    generation_id = generate_id(hex_seed=args.hex_seed)
    rng = random.Random(seed_from_id(generation_id))
    print(f"Generation ID: {generation_id} (hex seed: {hex_seed_from_id(generation_id)})")

    seeds = load_seeds(args.seed_file)
    print(f"Loaded {len(seeds)} seed questions from {args.seed_file}")

    questions = render_seed_questions(seeds, args.benchmark_code, rng)
    for q in questions:
        q["generation_id"] = generation_id
    write_jsonl(questions, args.out)
    print(f"Saved {len(questions)} questions to {args.out}")


if __name__ == "__main__":
    main()
