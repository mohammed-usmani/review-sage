from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReviewChunk:
    review_id: str
    product_id: str
    brand: str
    category: str
    price: float
    avg_rating: float
    review_count: int
    text: str


@dataclass
class Candidate:
    product_id: str
    brand: str
    category: str
    price: float
    avg_rating: float
    review_count: int
    snippets: list[str]
    chunk_ids: list[str]

    @classmethod
    def from_chunks(cls, product_id: str, chunks: list[ReviewChunk]) -> "Candidate":
        first = chunks[0]
        return cls(
            product_id=product_id,
            brand=first.brand,
            category=first.category,
            price=first.price,
            avg_rating=first.avg_rating,
            review_count=first.review_count,
            snippets=[c.text for c in chunks],
            chunk_ids=[c.review_id for c in chunks],
        )


@dataclass
class RankedProduct:
    candidate: Candidate
    rank: int
    reasoning: str
    cited_snippet_ids: list[str] = field(default_factory=list)


@dataclass
class RecommendResult:
    query: str
    mode: str  # "naive" | "hyde"
    bias_on: bool
    ranked: list[RankedProduct]
