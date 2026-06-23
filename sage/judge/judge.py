from __future__ import annotations

import json

from sage.models import Candidate, RankedProduct
from sage.providers.base import LLMProvider
from sage.judge.bias_controls import blind_brands, cap_evidence, normalized_review_count, orderings

JUDGE_INSTRUCTIONS = """You are an impartial shopping judge. Rank the products below from
best to worst for the shopper's need. Judge only on the evidence shown. Return STRICT JSON:
{{"ranking": [{{"product_id": "...", "reason": "..."}}, ...]}} with every product included once.

Shopper need: {query}

Products:
{products}

JSON:"""


def build_judge_prompt(query: str, candidates: list[Candidate]) -> str:
    blocks = []
    for c in candidates:
        vol = normalized_review_count(c.review_count)
        snippets = " | ".join(c.snippets)
        blocks.append(
            f"PRODUCT {c.product_id}: brand={c.brand} price=${c.price:.0f} "
            f"rating={c.avg_rating} review_volume={vol:.2f}\n  reviews: {snippets}"
        )
    return JUDGE_INSTRUCTIONS.format(query=query, products="\n".join(blocks))


def _rank_once(query: str, candidates: list[Candidate], llm: LLMProvider) -> list[str]:
    prompt = build_judge_prompt(query, candidates)
    raw = llm.generate(prompt, temperature=0.0)
    data = json.loads(raw)
    return [item["product_id"] for item in data["ranking"]]


def _reasons_from(query: str, candidates: list[Candidate], llm: LLMProvider) -> dict[str, str]:
    prompt = build_judge_prompt(query, candidates)
    data = json.loads(llm.generate(prompt, temperature=0.0))
    return {item["product_id"]: item.get("reason", "") for item in data["ranking"]}


def judge_rank(
    query: str,
    candidates: list[Candidate],
    *,
    llm: LLMProvider,
    bias_on: bool,
    position_orderings: int,
    seed: int,
    max_snippets: int,
    max_snippet_chars: int,
) -> list[RankedProduct]:
    if not candidates:
        return []

    # original (un-blinded) candidates, keyed for the final result
    original = {c.product_id: c for c in candidates}

    if bias_on:
        prepared = [
            cap_evidence(c, max_snippets=max_snippets, max_chars=max_snippet_chars)
            for c in candidates
        ]
        prepared, _mapping = blind_brands(prepared)
        variants = orderings(prepared, n=position_orderings, seed=seed)
    else:
        prepared = candidates
        variants = [prepared]

    # aggregate mean rank position across orderings (lower = better)
    rank_sums: dict[str, float] = {c.product_id: 0.0 for c in prepared}
    counts: dict[str, int] = {c.product_id: 0 for c in prepared}
    for variant in variants:
        order = _rank_once(query, variant, llm)
        for position, pid in enumerate(order):
            rank_sums[pid] += position
            counts[pid] += 1

    mean_rank = {pid: rank_sums[pid] / max(counts[pid], 1) for pid in rank_sums}
    final_order = sorted(mean_rank, key=lambda pid: (mean_rank[pid], pid))

    reasons = _reasons_from(query, variants[0], llm)

    ranked: list[RankedProduct] = []
    for i, pid in enumerate(final_order, start=1):
        cand = original[pid]
        ranked.append(
            RankedProduct(
                candidate=cand,
                rank=i,
                reasoning=reasons.get(pid, ""),
                cited_snippet_ids=cand.chunk_ids,
            )
        )
    return ranked
