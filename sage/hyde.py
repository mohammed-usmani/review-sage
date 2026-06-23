from __future__ import annotations

from sage.providers.base import LLMProvider

HYDE_PROMPT = """You are writing a single, concise hypothetical product review that would
perfectly answer this shopper's need. Write as if a real owner wrote it. Mention the
concrete features and use-cases implied by the request. Do not name brands.

Shopper need: {query}

Hypothetical ideal review:"""


def embedding_text(query: str, *, mode: str, generator: LLMProvider) -> str:
    """Return the text to embed for retrieval.

    naive -> the raw query; hyde -> an LLM-written hypothetical ideal review.
    """
    if mode == "naive":
        return query
    if mode == "hyde":
        return generator.generate(HYDE_PROMPT.format(query=query), temperature=0.0)
    raise ValueError(f"unknown mode: {mode}")
