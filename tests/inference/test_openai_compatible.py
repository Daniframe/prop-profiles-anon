import threading
from types import SimpleNamespace

from src.propensity.inference.backends.openai_compatible import OpenAICompatibleBackend


class FakeChatClient:
    """Mocks client.chat.completions.create(...) -- no real network call.
    Responds based on the last user message content so results can be
    checked against the originating chat regardless of completion order."""

    def __init__(self, fail_first_n_calls=0):
        self._fail_first_n_calls = fail_first_n_calls
        self._call_count = 0
        self._lock = threading.Lock()
        self.received_kwargs = []

        outer = self

        class _Completions:
            @staticmethod
            def create(**kwargs):
                with outer._lock:
                    outer._call_count += 1
                    call_num = outer._call_count
                    outer.received_kwargs.append(kwargs)
                if call_num <= outer._fail_first_n_calls:
                    raise RuntimeError("simulated transient error")
                user_content = kwargs["messages"][-1]["content"]
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"echo: {user_content}"))])

        self.chat = SimpleNamespace(completions=_Completions())


def _chat(user_text, with_system=True):
    messages = [{"role": "user", "content": user_text}]
    if with_system:
        messages.insert(0, {"role": "system", "content": "You are helpful."})
    return messages


def test_generate_batch_preserves_order_regardless_of_completion_order():
    client = FakeChatClient()
    backend = OpenAICompatibleBackend(client, model="gpt-4o", concurrency=5)

    chats = [_chat(f"question {i}") for i in range(10)]
    results = backend.generate_batch(chats)

    assert results == [f"echo: question {i}" for i in range(10)]


def test_generate_batch_sends_model_and_temperature():
    client = FakeChatClient()
    backend = OpenAICompatibleBackend(client, model="gpt-4o", temperature=0.7, max_tokens=500)

    backend.generate_batch([_chat("hi")])

    kwargs = client.received_kwargs[0]
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 500
    assert "max_completion_tokens" not in kwargs


def test_reasoning_model_uses_max_completion_tokens_and_drops_system_role():
    client = FakeChatClient()
    backend = OpenAICompatibleBackend(client, model="o1", reasoning_model=True, max_tokens=300)

    backend.generate_batch([_chat("hi")])

    kwargs = client.received_kwargs[0]
    assert kwargs["max_completion_tokens"] == 300
    assert "max_tokens" not in kwargs
    assert all(m["role"] != "system" for m in kwargs["messages"])
    assert "You are helpful." in kwargs["messages"][0]["content"]
    assert "hi" in kwargs["messages"][0]["content"]


def test_generate_batch_retries_on_failure_then_succeeds():
    client = FakeChatClient(fail_first_n_calls=1)
    backend = OpenAICompatibleBackend(client, model="gpt-4o", retries=3, concurrency=1, retry_backoff_s=0)

    results = backend.generate_batch([_chat("only one")])

    assert results == ["echo: only one"]
    assert client._call_count == 2  # one failure, then a successful retry


def test_generate_batch_returns_error_string_after_exhausting_retries():
    client = FakeChatClient(fail_first_n_calls=99)
    backend = OpenAICompatibleBackend(client, model="gpt-4o", retries=2, concurrency=1, retry_backoff_s=0)

    results = backend.generate_batch([_chat("hi")])

    assert results[0].startswith("ERROR:")
