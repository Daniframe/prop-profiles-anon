"""Inference backends, all implementing InferenceBackend.generate_batch.

- base: the InferenceBackend protocol.
- vllm_backend: local/cluster GPU inference (lazy `import vllm`).
- openai_compatible: any OpenAI-Chat-Completions-compatible HTTP endpoint.
"""

from .base import InferenceBackend
from .vllm_backend import VLLMBackend
from .openai_compatible import OpenAICompatibleBackend

__all__ = ["InferenceBackend", "VLLMBackend", "OpenAICompatibleBackend"]
