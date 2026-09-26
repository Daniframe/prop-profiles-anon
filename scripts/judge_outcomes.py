"""CLI entry point for grading open-ended responses via LLM-as-judge (e.g.
for TimeMenatQA, which has no options to string-match against).

Same submit/status/process/sequential mode contract as
scripts/annotate_benchmark.py, operating on one --persona (prompt name, e.g.
"0", "-1", "baseline") from a results JSON written by scripts/run_inference.py.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.propensity.common import load_config, write_jsonl
from src.propensity.common.llm_clients import get_azure_client, get_openai_client
from src.propensity.inference.reunite import load_results
from src.propensity.inference.judge import (
    judge_sequential,
    build_judge_batch_requests,
    submit_judge_batch,
    check_judge_batch_status,
    process_judge_batch,
)


def _get_client(provider: str):
    return get_azure_client() if provider == "azure" else get_openai_client()


def main():
    parser = argparse.ArgumentParser(description="Grade open-ended responses via LLM-as-judge")
    parser.add_argument("mode", choices=["submit", "status", "process", "sequential"])
    parser.add_argument("--results_filename", type=str, help="Path to a results JSON from scripts/run_inference.py")
    parser.add_argument("--persona", type=str, help="Prompt name within the results file (e.g. '0', '-1', 'baseline')")
    parser.add_argument("--batch_input_filename", type=str, help="Path to write the batch input JSONL file")
    parser.add_argument("--batch_id", type=str, help="Batch ID (required for status and process modes)")
    parser.add_argument("--judge_output_path", type=str, help="Path to write the graded output JSONL file")
    parser.add_argument("--provider", choices=["azure", "openai"], default="azure")
    parser.add_argument("--model", type=str, default=None, help="Overrides config/inference.yaml's judge.model")
    parser.add_argument("--temperature", type=float, default=None, help="Overrides config/inference.yaml's judge.temperature")
    args = parser.parse_args()

    config = load_config("inference")
    judge_cfg = config.get("judge", {})
    model = args.model or judge_cfg.get("model", "gpt-4.1")
    temperature = args.temperature if args.temperature is not None else judge_cfg.get("temperature", 0.0)

    if args.mode == "submit":
        assert args.results_filename and args.persona and args.batch_input_filename, \
            "submit requires --results_filename --persona --batch_input_filename"
        items = load_results(args.results_filename)["results"][args.persona]["responses"]
        print(f"Loaded {len(items)} responses for persona '{args.persona}'")

        requests = build_judge_batch_requests(items, model, temperature)
        client = _get_client(args.provider)
        batch_id = submit_judge_batch(client, requests, args.batch_input_filename)
        print(f"Batch successfully submitted with id {batch_id}")
        print(f"\nTo check status, run:\n  python {__file__} status --batch_id {batch_id} --provider {args.provider}")
        print(f"\nTo process results once complete, run:\n  python {__file__} process --batch_id {batch_id} --results_filename {args.results_filename} --persona {args.persona} --judge_output_path <output> --provider {args.provider}")

    elif args.mode == "status":
        assert args.batch_id, "status requires --batch_id"
        client = _get_client(args.provider)
        status = check_judge_batch_status(client, args.batch_id)
        for k, v in status.items():
            print(f"{k}: {v}")

    elif args.mode == "process":
        assert args.batch_id and args.results_filename and args.persona and args.judge_output_path, \
            "process requires --batch_id --results_filename --persona --judge_output_path"
        items = load_results(args.results_filename)["results"][args.persona]["responses"]
        client = _get_client(args.provider)
        graded = process_judge_batch(client, args.batch_id, items)
        write_jsonl(graded, args.judge_output_path)
        print(f"Saved {len(graded)}/{len(items)} graded responses to {args.judge_output_path}")

    elif args.mode == "sequential":
        assert args.results_filename and args.persona and args.judge_output_path, \
            "sequential requires --results_filename --persona --judge_output_path"
        items = load_results(args.results_filename)["results"][args.persona]["responses"]
        print(f"Loaded {len(items)} responses for persona '{args.persona}'")

        client = _get_client(args.provider)
        graded = judge_sequential(items, client, model, temperature)
        write_jsonl(graded, args.judge_output_path)
        print(f"Saved {len(graded)}/{len(items)} graded responses to {args.judge_output_path}")


if __name__ == "__main__":
    main()
