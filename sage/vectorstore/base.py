from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass
class Filter:
    category: str | None = None
    price_max: float | None = None

    def matches(self, payload: dict) -> bool:
        if self.category is not None and payload.get("category") != self.category:
            return False
        if self.price_max is not None and payload.get("price", float("inf")) > self.price_max:
            return False
        return True


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict


@runtime_checkable
class VectorStore(Protocol):
    dim: int

    def upsert(
        self, ids: Sequence[str], vectors: Sequence[Sequence[float]], payloads: Sequence[dict]
    ) -> None: ...

    def search(
        self, vector: Sequence[float], top_k: int, flt: Filter | None = None
    ) -> list[SearchHit]: ...
