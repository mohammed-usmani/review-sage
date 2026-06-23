from __future__ import annotations

from sage.models import Candidate, ReviewChunk
from sage.providers.base import EmbeddingProvider
from sage.vectorstore.base import Filter, VectorStore


def retrieve_candidates(
    text: str,
    *,
    embedder: EmbeddingProvider,
    store: VectorStore,
    flt: Filter | None,
    top_k: int,
    max_candidates: int,
    max_snippets: int,
) -> list[Candidate]:
    vector = embedder.embed([text])[0]
    hits = store.search(vector, top_k=top_k, flt=flt)

    grouped: dict[str, list[ReviewChunk]] = {}
    order: list[str] = []
    for hit in hits:
        p = hit.payload
        pid = p["product_id"]
        if pid not in grouped:
            grouped[pid] = []
            order.append(pid)
        if len(grouped[pid]) >= max_snippets:
            continue
        grouped[pid].append(
            ReviewChunk(
                review_id=p["review_id"],
                product_id=pid,
                brand=p["brand"],
                category=p["category"],
                price=p["price"],
                avg_rating=p["avg_rating"],
                review_count=p["review_count"],
                text=p["text"],
            )
        )

    candidates = [Candidate.from_chunks(pid, grouped[pid]) for pid in order]
    return candidates[:max_candidates]
