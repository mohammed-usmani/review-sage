from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    family: str

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    family: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
