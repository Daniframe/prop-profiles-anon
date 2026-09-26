"""Tests the OOM-detection and batch-halving retry logic without importing
the real `vllm` package at all (it's not installed in this environment) --
VLLMBackend.__init__ is the only place that imports vllm, and it's lazy, so
we can exercise everything else against a fake `llm` object."""

from types import SimpleNamespace

from src.propensity.inference.backends.vllm_backend import _is_oom, _robust_chat_generate_ordered


class SimulatedOOMError(Exception):
    pass


def test_is_oom_detects_out_of_memory_message():
    assert _is_oom(RuntimeError("CUDA out of memory. Tried to allocate...")) is True


def test_is_oom_rejects_unrelated_errors():
    assert _is_oom(ValueError("bad input")) is False


class FakeLLM:
    """Fails with a simulated OOM on any batch of >= oom_threshold chats,
    succeeds otherwise -- lets us verify the halving retry converges."""

    def __init__(self, oom_threshold):
        self.oom_threshold = oom_threshold
        self.calls = []

    def chat(self, messages, sampling_params, use_tqdm):
        self.calls.append(len(messages))
        if len(messages) >= self.oom_threshold:
            raise SimulatedOOMError("out of memory")
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=f"response to {chat[0]['content']}")])
            for chat in messages
        ]


def _chats(n):
    return [[{"role": "user", "content": f"q{i}"}] for i in range(n)]


def test_robust_chat_generate_ordered_returns_immediately_when_no_oom():
    llm = FakeLLM(oom_threshold=100)
    result = _robust_chat_generate_ordered(llm, _chats(4), sampling=None)
    assert len(result) == 4
    assert llm.calls == [4]


def test_robust_chat_generate_ordered_halves_batch_on_oom_until_it_fits():
    llm = FakeLLM(oom_threshold=3)
    result = _robust_chat_generate_ordered(llm, _chats(4), sampling=None)

    assert len(result) == 4
    # first call OOMs at size 4, then splits into halves of 2, both succeed
    assert llm.calls == [4, 2, 2]


def test_robust_chat_generate_ordered_preserves_order_across_halves():
    llm = FakeLLM(oom_threshold=3)
    result = _robust_chat_generate_ordered(llm, _chats(4), sampling=None)
    assert [r.outputs[0].text for r in result] == [
        "response to q0", "response to q1", "response to q2", "response to q3"
    ]


def test_robust_chat_generate_ordered_raises_if_single_request_still_ooms():
    llm = FakeLLM(oom_threshold=1)
    try:
        _robust_chat_generate_ordered(llm, _chats(2), sampling=None)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "OOM even for single request" in str(exc)


def test_robust_chat_generate_ordered_returns_empty_list_for_no_chats():
    llm = FakeLLM(oom_threshold=1)
    assert _robust_chat_generate_ordered(llm, [], sampling=None) == []
    assert llm.calls == []
