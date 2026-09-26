"""CLI entry point for Sector 2: annotate a benchmark's propensity demand
intervals.

Same submit/status/process/sequential mode contract as the original
annotate_benchmark_azure.py script (same
flags, same four modes), now built on top of src/propensity/annotation/*
with a provider-agnostic client (--provider azure|openai) instead of a
hardcoded AzureOpenAI.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.propensity.common import load_config, read_jsonl, write_jsonl
from src.propensity.common.llm_clients import get_azure_client, get_openai_client
from src.propensity.annotation import (
    load_rubric_text,
    annotate_sequential,
    build_batch_requests,
    submit_batch,
    check_batch_status,
    process_batch,
)


def _get_client(provider: str):
    return get_azure_client() if provider == "azure" else get_openai_client()


def main():
    parser = argparse.ArgumentParser(description="Annotate a benchmark's propensity demand intervals")
    parser.add_argument("mode", choices=["submit", "status", "process", "sequential"])
    parser.add_argument("--dataset_path", type=str, help="Path to the ready benchmark JSONL to annotate")
    parser.add_argument("--rubric_path", type=str, help="Path to the rubric text file")
    parser.add_argument("--prop_name", type=str, help="Human-readable name of the propensity being measured")
    parser.add_argument("--batch_input_path", type=str, help="Path to write the batch input JSONL file")
    parser.add_argument("--batch_id", type=str, help="Batch ID (required for status and process modes)")
    parser.add_argument("--annotation_output_path", type=str, help="Path to write the annotated output JSONL file")
    parser.add_argument("--provider", choices=["azure", "openai"], default="azure")
    parser.add_argument("--model", type=str, default=None, help="Overrides config/annotation.yaml's annotator_model")
    parser.add_argument("--temperature", type=float, default=None, help="Overrides config/annotation.yaml's annotator_temperature")
    args = parser.parse_args()

    config = load_config("annotation")
    model = args.model or config.get("annotator_model", "gpt-4.1")
    temperature = args.temperature if args.temperature is not None else config.get("annotator_temperature", 0.0)

    if args.mode == "submit":
        assert args.dataset_path and args.rubric_path and args.prop_name and args.batch_input_path, \
            "submit requires --dataset_path --rubric_path --prop_name --batch_input_path"
        rubric = load_rubric_text(args.rubric_path)
        questions = list(read_jsonl(args.dataset_path))
        print(f"Loaded {len(questions)} questions")

        requests = build_batch_requests(questions, rubric, args.prop_name, model, temperature)
        client = _get_client(args.provider)
        batch_id = submit_batch(client, requests, args.batch_input_path)
        print(f"Batch successfully submitted with id {batch_id}")
        print(f"\nTo check status, run:\n  python {__file__} status --batch_id {batch_id} --provider {args.provider}")
        print(f"\nTo process results once complete, run:\n  python {__file__} process --batch_id {batch_id} --dataset_path {args.dataset_path} --annotation_output_path <output> --provider {args.provider}")

    elif args.mode == "status":
        assert args.batch_id, "status requires --batch_id"
        client = _get_client(args.provider)
        status = check_batch_status(client, args.batch_id)
        for k, v in status.items():
            print(f"{k}: {v}")

    elif args.mode == "process":
        assert args.batch_id and args.dataset_path and args.annotation_output_path, \
            "process requires --batch_id --dataset_path --annotation_output_path"
        questions = list(read_jsonl(args.dataset_path))
        client = _get_client(args.provider)
        annotated = process_batch(client, args.batch_id, questions)
        write_jsonl(annotated, args.annotation_output_path)
        print(f"Saved {len(annotated)}/{len(questions)} annotated questions to {args.annotation_output_path}")

    elif args.mode == "sequential":
        assert args.dataset_path and args.rubric_path and args.prop_name and args.annotation_output_path, \
            "sequential requires --dataset_path --rubric_path --prop_name --annotation_output_path"
        rubric = load_rubric_text(args.rubric_path)
        questions = list(read_jsonl(args.dataset_path))
        print(f"Loaded {len(questions)} questions")

        client = _get_client(args.provider)
        annotated = annotate_sequential(questions, rubric, args.prop_name, client, model, temperature)
        write_jsonl(annotated, args.annotation_output_path)
        print(f"Saved {len(annotated)}/{len(questions)} annotated questions to {args.annotation_output_path}")


if __name__ == "__main__":
    main()
