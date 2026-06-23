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
        snippets = " | ".join(c.snippets) if c.snippets else "(no review text)"
        # price rounded to whole dollars on purpose: removes cent-level anchoring.
        blocks.append(
            f"PRODUCT {c.product_id}: brand={c.brand} price=${c.price:.0f} "
            f"rating={c.avg_rating} review_volume={vol:.2f}\n  reviews: {snippets}"
        )
    return JUDGE_INSTRUCTIONS.format(query=query, products="\n".join(blocks))


def _loads_json(raw: str) -> dict:
    """Parse model JSON, tolerating ```json fences and surrounding prose that
    real LLMs (esp. Gemini/OpenAI without JSON mode) tend to add."""
    text = raw.strip()
    if text.startswith("```"):
        text = text[3:]
        if text[:4].lower() == "json":
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # last resort: extract the outermost {...}
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _rank_once(query: str, candidates: list[Candidate], llm: LLMProvider) -> list[dict]:
    """Return the LLM's ranking as a list of {product_id, reason} items."""
    data = _loads_json(llm.generate(build_judge_prompt(query, candidates), temperature=0.0))
    return data["ranking"]


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

    valid_ids = {c.product_id for c in prepared}
    n_products = len(prepared)

    # Aggregate mean rank position across orderings (lower = better). Robustness:
    #  - unknown/hallucinated ids returned by the LLM are ignored
    #  - a product the LLM OMITS in a variant gets the worst position for that
    #    variant (never accidentally promoted to rank 0)
    #  - duplicate ids within one ranking count only on first appearance
    rank_sums: dict[str, float] = {pid: 0.0 for pid in valid_ids}
    first_reasons: dict[str, str] = {}
    for v_idx, variant in enumerate(variants):
        ranking = _rank_once(query, variant, llm)
        positions: dict[str, int] = {}
        next_pos = 0
        for item in ranking:
            pid = item.get("product_id")
            if pid in valid_ids and pid not in positions:
                positions[pid] = next_pos
                next_pos += 1
                if v_idx == 0:
                    first_reasons[pid] = item.get("reason", "")
        for pid in rank_sums:
            rank_sums[pid] += positions.get(pid, n_products)

    mean_rank = {pid: rank_sums[pid] / len(variants) for pid in rank_sums}
    final_order = sorted(mean_rank, key=lambda pid: (mean_rank[pid], pid))

    ranked: list[RankedProduct] = []
    for i, pid in enumerate(final_order, start=1):
        cand = original[pid]
        ranked.append(
            RankedProduct(
                candidate=cand,
                rank=i,
                reasoning=first_reasons.get(pid, ""),
                cited_snippet_ids=cand.chunk_ids,
            )
        )
    return ranked
