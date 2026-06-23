import pytest

from sage.models import ReviewChunk, Candidate, RankedProduct, RecommendResult


def test_candidate_from_chunks_aggregates():
    chunks = [
        ReviewChunk(review_id="r1", product_id="p1", brand="Acme", category="earbuds",
                    price=80.0, avg_rating=4.5, review_count=120, text="great bass"),
        ReviewChunk(review_id="r2", product_id="p1", brand="Acme", category="earbuds",
                    price=80.0, avg_rating=4.5, review_count=120, text="stays in at gym"),
    ]
    cand = Candidate.from_chunks("p1", chunks)
    assert cand.product_id == "p1"
    assert cand.brand == "Acme"
    assert cand.price == 80.0
    assert cand.snippets == ["great bass", "stays in at gym"]
    assert cand.chunk_ids == ["r1", "r2"]


def test_recommend_result_holds_ranking():
    cand = Candidate(product_id="p1", brand="Acme", category="earbuds", price=80.0,
                     avg_rating=4.5, review_count=120, snippets=["x"], chunk_ids=["r1"])
    rp = RankedProduct(candidate=cand, rank=1, reasoning="best fit", cited_snippet_ids=["r1"])
    res = RecommendResult(query="q", mode="hyde", bias_on=True, ranked=[rp])
    assert res.ranked[0].rank == 1
    assert res.mode == "hyde"
    assert res.bias_on is True
    assert len(res.ranked) == 1


def test_ranked_product_cited_snippets_default_empty():
    cand = Candidate(product_id="p1", brand="Acme", category="earbuds", price=80.0,
                     avg_rating=4.5, review_count=120, snippets=["x"], chunk_ids=["r1"])
    rp = RankedProduct(candidate=cand, rank=1, reasoning="best")
    assert rp.cited_snippet_ids == []


def test_from_chunks_rejects_empty():
    with pytest.raises(ValueError):
        Candidate.from_chunks("p1", [])
