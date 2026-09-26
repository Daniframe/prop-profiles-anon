"""Any OpenAI-Chat-Completions-compatible HTTP backend: OpenAI, Azure OpenAI,
or a self-hosted OpenAI-compatible server (vLLM's own server mode, Ollama, LM
Studio, OpenRouter, Together, Groq, ...). Generalizes the
original openai-run-models.py, which hardcoded
AzureOpenAI, to a duck-typed `client` -- see
common.llm_clients.get_azure_client()/get_openai_client() (the latter takes
an optional base_url for self-hosted/alternate endpoints).
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class OpenAICompatibleBackend:
    def __init__(
        self,
        client,
        model: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        concurrency: int = 10,
        retries: int = 3,
        retry_backoff_s: float = 2.0,
        reasoning_model: bool = False,
    ):
        """reasoning_model=True switches to o1-style call parameters:
        max_completion_tokens instead of max_tokens, and no system role
        (system content is folded into the first user turn instead)."""
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s
        self.reasoning_model = reasoning_model

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        if not self.reasoning_model:
            return messages

        system_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        if system_text and rest:
            rest[0] = {**rest[0], "content": f"{system_text}\n\n{rest[0]['content']}"}
        return rest

    def _call_one(self, messages: list[dict]) -> str:
        kwargs = {
            "model": self.model,
            "messages": self._prepare_messages(messages),
            "temperature": self.temperature,
        }
        kwargs["max_completion_tokens" if self.reasoning_model else "max_tokens"] = self.max_tokens

        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as exc:
                if attempt < self.retries - 1:
                    time.sleep(self.retry_backoff_s * (attempt + 1))
                else:
                    return f"ERROR: {exc}"

    def generate_batch(self, chats: list[list[dict]]) -> list[str]:
        results: list[str] = [None] * len(chats)
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_idx = {executor.submit(self._call_one, chat): i for i, chat in enumerate(chats)}
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()
        return results
