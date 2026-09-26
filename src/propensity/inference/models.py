"""vLLM model-shortcut registry (short name -> HF repo id) and the set of
shortcuts that are reasoning/"thinking" models. Ported from
the original infer-propensities-jsonl-robust.py.

Only relevant to VLLMBackend -- OpenAICompatibleBackend takes --model as the
literal provider/deployment name, since API model names are already short.
"""

MODEL_SHORTCUTS = {
    "llama33": "meta-llama/Llama-3.3-70B-Instruct",
    "llama32": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma3": "google/gemma-3-27b-it",

    "ministral": "mistralai/Ministral-3-14B-Instruct-2512",
    "ministral-r": "mistralai/Ministral-3-14B-Reasoning-2512",

    "qwen3": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3-t": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "qwen3-4b-t": "Qwen/Qwen3-4B-Thinking-2507",
    "qwen3-4b-i": "Qwen/Qwen3-4B-Instruct-2507",

    "Nemo": "nvidia/Nemotron-Cascade-14B-Thinking",

    "ds-r1-llama70": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "ds-r1-qwen32": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "ds-r1-llama8": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}

THINKING_MODELS = {"qwen3-t", "ministral-r", "ds-r1-llama70", "ds-r1-qwen32", "ds-r1-llama8"}


def resolve_model(model_key: str, extra_shortcuts: dict[str, str] | None = None) -> tuple[str, bool]:
    """Resolves a model shortcut to (hf_repo_id, is_thinking_model). Unknown
    keys pass through unchanged (treated as an already-full HF repo id).

    extra_shortcuts (e.g. from config/inference.yaml's `models:` block) are
    checked first, letting a repo's config extend/override the built-in
    registry without touching this module.
    """
    shortcuts = {**MODEL_SHORTCUTS, **(extra_shortcuts or {})}
    resolved = shortcuts.get(model_key, model_key)
    is_thinking = model_key in THINKING_MODELS
    return resolved, is_thinking
