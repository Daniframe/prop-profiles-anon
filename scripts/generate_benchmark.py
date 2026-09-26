"""CLI entry point for Sector 1: generates a benchmark's final rendered
question set (data/benchmarks/ready/{code}/{code}.jsonl) from its declarative
spec (benchmarks/*.yaml), dispatching on spec.kind:

- risk_levels / color_permutation: generated directly (no separate seed stage).
- context_grounded: generates (or loads a pre-generated) seed file into
  data/benchmarks/raw/{code}/, then renders it -- same as running
  generate_benchmark.py followed by render_benchmark.py in one step.
- external: adapts an existing real benchmark file, no generation/rendering.
"""

import os
import sys
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.propensity.common import generate_id, seed_from_id, hex_seed_from_id, write_jsonl, read_jsonl, load_config
from src.propensity.common.llm_clients import get_azure_client
from src.propensity.generation import (
    load_spec,
    generate_risk_levels,
    generate_color_permutation,
    generate_context_seeds,
    render_seed_questions,
    load_external_benchmark,
)


def main():
    parser = argparse.ArgumentParser(description="Generate a benchmark's ready question set from its spec")
    parser.add_argument("--spec", type=str, required=True, help="Path to a benchmarks/*.yaml spec file")
    parser.add_argument("--out", type=str, required=True, help="Output path for the rendered ready/*.jsonl file")
    parser.add_argument("--hex-seed", type=str, default=None, help="Hex seed for reproducible generation")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    generation_id = generate_id(hex_seed=args.hex_seed)
    rng = random.Random(seed_from_id(generation_id))
    print(f"Generation ID: {generation_id} (hex seed: {hex_seed_from_id(generation_id)})")

    if spec.kind == "risk_levels":
        questions = generate_risk_levels(spec, rng)
        for q in questions:
            q["generation_id"] = generation_id
        write_jsonl(questions, args.out)

    elif spec.kind == "color_permutation":
        questions = generate_color_permutation(spec)
        for q in questions:
            q["generation_id"] = generation_id
        write_jsonl(questions, args.out)

    elif spec.kind == "context_grounded":
        seed_file = spec.params.get("seed_file")
        if seed_file:
            print(f"Using pre-generated seed file: {seed_file}")
            seeds = list(read_jsonl(seed_file)) if str(seed_file).endswith(".jsonl") else __import__("json").load(open(seed_file, encoding="utf-8"))
        else:
            config = load_config("generation")
            client = get_azure_client()
            model = spec.params.get("azure_model", config.get("azure", {}).get("model", "gpt-4o"))
            temperature = spec.params.get("temperature", config.get("azure", {}).get("temperature", 0.5))
            seeds = generate_context_seeds(spec, client, model=model, temperature=temperature)
            seed_file = f"data/benchmarks/raw/{spec.name}/{spec.name}_generated_context_levels.json"
            os.makedirs(os.path.dirname(seed_file), exist_ok=True)
            import json
            with open(seed_file, "w", encoding="utf-8") as f:
                json.dump(seeds, f, indent=2)
            print(f"Saved {len(seeds)} generated seeds to {seed_file}")

        questions = render_seed_questions(seeds, spec.name, rng)
        for q in questions:
            q["generation_id"] = generation_id
        write_jsonl(questions, args.out)

    elif spec.kind == "external":
        questions = load_external_benchmark(spec)
        write_jsonl(questions, args.out)

    else:
        raise ValueError(f"Unhandled benchmark kind: {spec.kind}")

    print(f"Saved {len(questions)} questions to {args.out}")


if __name__ == "__main__":
    main()
