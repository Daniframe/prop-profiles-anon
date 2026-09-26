"""CLI entry point for Sector 3: run every incitement-level system prompt
against every question in a ready benchmark, for one model.

Local GPU inference:
  python scripts/run_inference.py --backend vllm --model llama33 \\
    --input data/benchmarks/ready/RA/RA.jsonl --prompts-dir prompts/incitement/RA

Any OpenAI-Chat-Completions-compatible HTTP endpoint (Azure, OpenAI, or a
self-hosted server via --provider openai --base-url ...):
  python scripts/run_inference.py --backend openai_compatible --model gpt-4o \\
    --provider azure --input data/benchmarks/ready/RA/RA.jsonl --prompts-dir prompts/incitement/RA
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.propensity.common import load_config, read_jsonl
from src.propensity.common.llm_clients import get_azure_client, get_openai_client
from src.propensity.inference import load_prompts, run_inference, resolve_model
from src.propensity.inference.backends import VLLMBackend, OpenAICompatibleBackend


def build_backend(args, config):
    """Returns (backend, is_thinking_model)."""
    if args.backend == "vllm":
        resolved_model, is_thinking = resolve_model(args.model, config.get("models"))
        vllm_cfg = config.get("vllm", {})
        backend = VLLMBackend(
            resolved_model,
            max_model_len=vllm_cfg.get("max_model_len", 4096),
            max_tokens=vllm_cfg.get("max_tokens", 2048),
            temperature=vllm_cfg.get("temperature", 0.0),
            gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.9),
            tensor_parallel_size=args.gpus or vllm_cfg.get("tensor_parallel_size", 1),
        )
        return backend, is_thinking

    client = get_azure_client() if args.provider == "azure" else get_openai_client(base_url=args.base_url)
    oc_cfg = config.get("openai_compatible", {})
    backend = OpenAICompatibleBackend(
        client,
        args.model,
        max_tokens=oc_cfg.get("max_tokens", 2048),
        temperature=oc_cfg.get("temperature", 0.0),
        concurrency=oc_cfg.get("concurrency", 10),
        retries=oc_cfg.get("retries", 3),
        reasoning_model=args.reasoning_model,
    )
    return backend, args.strip_thinking


def main():
    parser = argparse.ArgumentParser(description="Run model inference across incitement levels")
    parser.add_argument("--backend", choices=["vllm", "openai_compatible"], required=True)
    parser.add_argument("--model", type=str, required=True,
                         help="Model shortcut/HF repo id (vllm) or provider model/deployment name (openai_compatible)")
    parser.add_argument("--input", type=str, required=True, help="Path to a ready benchmark JSONL")
    parser.add_argument("--prompts-dir", type=str, required=True, help="Directory of incitement system prompts (.txt/.md)")
    parser.add_argument("--question-field", type=str, default="question_text")
    parser.add_argument("--provider", choices=["azure", "openai"], default="azure", help="openai_compatible only")
    parser.add_argument("--base-url", type=str, default=None,
                         help="openai_compatible + --provider openai only: point at a self-hosted/alternate endpoint")
    parser.add_argument("--reasoning-model", action="store_true",
                         help="openai_compatible only: o1-style call params (max_completion_tokens, no system role)")
    parser.add_argument("--strip-thinking", action="store_true",
                         help="openai_compatible only: strip <think>...</think> traces before extracting the final answer "
                              "(vllm backends decide this automatically from the model registry)")
    parser.add_argument("--gpus", type=int, default=None, help="vllm only: tensor parallel size override")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    config = load_config("inference")
    results_dir = Path(args.results_dir or config.get("results_dir", "data/inference/raw"))
    results_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts_dir)
    questions = list(read_jsonl(args.input))
    print(f"Loaded {len(questions)} questions, {len(prompts)} prompts: {list(prompts)}")

    backend, is_thinking_model = build_backend(args, config)

    per_prompt = run_inference(
        backend, questions, prompts,
        question_field=args.question_field,
        is_thinking_model=is_thinking_model,
    )

    model_basename = args.model.split("/")[-1]
    input_stem = Path(args.input).stem
    run_stem = args.run_name or f"{model_basename}__{input_stem}"

    results_path = results_dir / f"{run_stem}.json"
    summary_path = results_dir / f"{run_stem}_summary.csv"

    payload = {
        "meta": {
            "backend": args.backend,
            "model": args.model,
            "is_thinking_model": is_thinking_model,
            "num_questions": len(questions),
            "prompts": list(prompts),
        },
        "results": per_prompt,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    summary = pd.DataFrame([{"prompt": k, "total": v["total"]} for k, v in per_prompt.items()])
    summary.to_csv(summary_path, index=False)

    print(f"Wrote:\n- {results_path}\n- {summary_path}")


if __name__ == "__main__":
    main()
