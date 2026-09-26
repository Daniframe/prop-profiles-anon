"""The one interface every inference backend implements. Deliberately a
Protocol (structural typing), not an ABC -- matches the duck-typed `client`
convention already used in generation/context_generation.py and
annotation/*: a backend just needs to expose generate_batch, nothing here
cares whether it's local GPU inference or an HTTP API call underneath.
"""

from typing import Protocol


class InferenceBackend(Protocol):
    def generate_batch(self, chats: list[list[dict]]) -> list[str]:
        """Generates one completion per chat, in the same order as `chats`.

        chats: a list of message lists, each in the standard
            [{"role": "system"|"user", "content": str}, ...] shape.

        Implementations decide how to parallelize/batch internally (a single
        batched call for local inference, concurrent HTTP requests for an
        API backend, etc.) -- callers only see one text response per chat.
        """
        ...
