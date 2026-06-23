from __future__ import annotations

from typing import Sequence

import numpy as np

from sage.vectorstore.base import Filter, SearchHit


class FaissStore:
    """In-process cosine store. Over-fetches then applies payload filter in Python."""

    def __init__(self, dim: int) -> None:
        import faiss

        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._payloads: list[dict] = []

    @staticmethod
    def _normed(vectors: Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def upsert(self, ids, vectors, payloads) -> None:
        self._index.add(self._normed(vectors))
        self._ids.extend(ids)
        self._payloads.extend(payloads)

    def search(self, vector, top_k, flt: Filter | None = None) -> list[SearchHit]:
        if not self._ids:
            return []
        over = min(len(self._ids), max(top_k * 5, top_k))
        q = self._normed([vector])
        scores, idxs = self._index.search(q, over)
        out: list[SearchHit] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            payload = self._payloads[idx]
            if flt is not None and not flt.matches(payload):
                continue
            out.append(SearchHit(id=self._ids[idx], score=float(score), payload=payload))
            if len(out) >= top_k:
                break
        return out
