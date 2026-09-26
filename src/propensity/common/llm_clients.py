"""OpenAI / Azure OpenAI client factory shared by generation (context-grounded
benchmarks), annotation, and inference (Azure backend + LLM-judge). Holds only
env-var-driven client construction -- no prompt content or business logic.
"""

import os

from openai import AzureOpenAI, OpenAI


def get_openai_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    """Returns an OpenAI client, defaulting to OPENAI_API_KEY from the environment.

    base_url lets this point at any OpenAI-Chat-Completions-compatible HTTP
    endpoint instead of api.openai.com -- a self-hosted vLLM/Ollama/LM Studio
    server, OpenRouter, Together, Groq, etc. -- which is what makes
    inference.backends.OpenAICompatibleBackend provider-agnostic rather than
    tied to OpenAI specifically.
    """
    return OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)


def get_azure_client(
    api_key: str | None = None,
    endpoint: str | None = None,
    api_version: str = "2024-10-21",
) -> AzureOpenAI:
    """Returns an AzureOpenAI client, defaulting to AZURE_OPENAI_API_KEY /
    AZURE_OPENAI_ENDPOINT from the environment."""
    return AzureOpenAI(
        api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version=api_version,
    )
