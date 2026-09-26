"""Local/cluster GPU inference via vLLM. Ported from
the original infer-propensities-jsonl-robust.py.

`vllm` (and its `torch`/CUDA dependencies) is not a hard requirement of this
package -- it's imported lazily inside VLLMBackend.__init__, so the rest of
src/propensity/inference stays importable (and testable) on machines without
a GPU or vllm installed. Install it yourself in a GPU environment:
`pip install vllm torch` before constructing a VLLMBackend.
"""

import gc


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "out of memory" in msg:
        return True
    try:
        import torch
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type and isinstance(exc, oom_type):
            return True
    except ImportError:
        pass
    return False


def _robust_chat_generate_ordered(llm, chats: list[list[dict]], sampling):
    """Runs llm.chat() over all chats; on an out-of-memory error, clears the
    CUDA cache and recursively retries on halved batches until each half
    succeeds (or a single-chat batch still OOMs, which re-raises)."""
    if not chats:
        return []
    try:
        return llm.chat(messages=chats, sampling_params=sampling, use_tqdm=True)
    except Exception as exc:
        if not _is_oom(exc):
            raise

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        if len(chats) == 1:
            raise RuntimeError("OOM even for single request") from exc

        mid = len(chats) // 2
        return (
            _robust_chat_generate_ordered(llm, chats[:mid], sampling)
            + _robust_chat_generate_ordered(llm, chats[mid:], sampling)
        )


class VLLMBackend:
    def __init__(
        self,
        model: str,
        *,
        max_model_len: int = 4096,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
    ):
        from vllm import LLM, SamplingParams

        common = dict(
            model=model,
            max_model_len=max_model_len,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        if "mistral" in model.lower():
            self._llm = LLM(**common, tokenizer_mode="mistral", load_format="mistral", config_format="mistral")
        else:
            self._llm = LLM(**common)

        self._sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def generate_batch(self, chats: list[list[dict]]) -> list[str]:
        outputs = _robust_chat_generate_ordered(self._llm, chats, self._sampling)
        texts = []
        for out in outputs:
            text = ""
            for gen in getattr(out, "outputs", []) or []:
                text = str(getattr(gen, "text", ""))
                break
            texts.append(text)
        return texts
