from __future__ import annotations

from typing import Callable

from sage.config import Settings
from sage.hyde import embedding_text
from sage.judge.judge import judge_rank
from sage.models import RecommendResult
from sage.providers.base import EmbeddingProvider, LLMProvider
from sage.retrieve import retrieve_candidates
from sage.vectorstore.base import Filter, VectorStore


def recommend(
    query: str,
    *,
    mode: str,
    bias_on: bool,
    flt: Filter | None,
    embedder: EmbeddingProvider,
    store: VectorStore,
    hyde_llm: LLMProvider,
    judge_llm: LLMProvider,
    settings: Settings,
) -> RecommendResult:
    text = embedding_text(query, mode=mode, generator=hyde_llm)
    candidates = retrieve_candidates(
        text,
        embedder=embedder,
        store=store,
        flt=flt,
        top_k=settings.top_k,
        max_candidates=settings.max_candidates,
        max_snippets=settings.max_snippets,
    )
    ranked = judge_rank(
        query,
        candidates,
        llm=judge_llm,
        bias_on=bias_on,
        position_orderings=settings.position_orderings,
        seed=settings.seed,
        max_snippets=settings.max_snippets,
        max_snippet_chars=settings.max_snippet_chars,
    )
    return RecommendResult(query=query, mode=mode, bias_on=bias_on, ranked=ranked)


def compare(
    query: str,
    *,
    flt: Filter | None,
    embedder: EmbeddingProvider,
    store: VectorStore,
    make_hyde_llm: Callable[[], LLMProvider],
    make_judge_llm: Callable[[], LLMProvider],
    settings: Settings,
) -> dict[str, RecommendResult]:
    """Run all four combinations for the UI compare view. Fresh LLM instances per run
    so scripted-fake state and call logs do not bleed across combinations."""
    out: dict[str, RecommendResult] = {}
    for mode in ("naive", "hyde"):
        for bias_on in (False, True):
            key = f"{mode}_bias_{'on' if bias_on else 'off'}"
            out[key] = recommend(
                query, mode=mode, bias_on=bias_on, flt=flt, embedder=embedder, store=store,
                hyde_llm=make_hyde_llm(), judge_llm=make_judge_llm(), settings=settings,
            )
    return out
