from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sage.config import Settings
from sage.hyde import embedding_text
from sage.providers.base import EmbeddingProvider, LLMProvider
from sage.retrieve import retrieve_candidates
from sage.vectorstore.base import Filter, VectorStore


@dataclass
class AblationRow:
    query: str
    naive_products: list[str]
    hyde_products: list[str]
    jaccard: float


def _product_ids(
    text: str,
    embedder: EmbeddingProvider,
    store: VectorStore,
    flt: Filter | None,
    settings: Settings,
) -> list[str]:
    cands = retrieve_candidates(
        text, embedder=embedder, store=store, flt=flt, top_k=settings.top_k,
        max_candidates=settings.max_candidates, max_snippets=settings.max_snippets,
    )
    return [c.product_id for c in cands]


def ablate(
    *,
    queries: list[str],
    flt: Filter | None,
    embedder: EmbeddingProvider,
    store: VectorStore,
    make_hyde_llm: Callable[[], LLMProvider],
    settings: Settings,
) -> list[AblationRow]:
    rows: list[AblationRow] = []
    for q in queries:
        naive = _product_ids(q, embedder, store, flt, settings)
        hyde_text = embedding_text(q, mode="hyde", generator=make_hyde_llm())
        hyde = _product_ids(hyde_text, embedder, store, flt, settings)
        ns, hs = set(naive), set(hyde)
        union = ns | hs
        jaccard = len(ns & hs) / len(union) if union else 1.0
        rows.append(AblationRow(query=q, naive_products=naive, hyde_products=hyde, jaccard=jaccard))
    return rows
