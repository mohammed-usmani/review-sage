import pytest

from sage.models import Candidate
from sage.judge.bias_controls import (
    blind_brands,
    cap_evidence,
    normalized_review_count,
    orderings,
)


def _cand(pid, brand, snippets, count=100):
    return Candidate(product_id=pid, brand=brand, category="earbuds", price=80.0,
                     avg_rating=4.4, review_count=count, snippets=snippets, chunk_ids=["x"])


def test_blind_brands_strips_names_from_text_and_field():
    c = _cand("p1", "Acme", ["Acme buds are great", "love my acme pair"])
    blinded, mapping = blind_brands([c])
    assert blinded[0].brand == "Brand-A"
    assert "Acme" not in " ".join(blinded[0].snippets)
    assert "acme" not in " ".join(blinded[0].snippets).lower()
    assert mapping["Brand-A"] == "Acme"


def test_blind_brands_does_not_mutate_input():
    c = _cand("p1", "Acme", ["Acme buds"])
    blind_brands([c])
    assert c.brand == "Acme"  # original untouched
    assert c.snippets == ["Acme buds"]


def test_blind_brands_respects_word_boundaries():
    # brand "On" must NOT corrupt unrelated words like "Only" / "comfortable"
    c = _cand("p1", "On", ["On earbuds. Only comfortable for long calls."])
    blinded, _ = blind_brands([c])
    text = blinded[0].snippets[0]
    assert text.startswith("Brand-A earbuds")
    assert "Only" in text  # not turned into "Brand-Aly"
    assert "comfortable" in text


def test_blind_brands_distinct_labels_per_brand():
    cands = [_cand("p1", "Acme", ["a"]), _cand("p2", "Bolt", ["b"]), _cand("p3", "Acme", ["c"])]
    blinded, mapping = blind_brands(cands)
    assert [b.brand for b in blinded] == ["Brand-A", "Brand-B", "Brand-A"]
    assert mapping == {"Brand-A": "Acme", "Brand-B": "Bolt"}


def test_blind_brands_raises_when_label_pool_exhausted():
    cands = [_cand(f"p{i}", f"Brand{i}", ["s"]) for i in range(17)]
    with pytest.raises(ValueError):
        blind_brands(cands)


def test_normalized_review_count_bounds():
    assert normalized_review_count(0) == 0.0
    assert normalized_review_count(1000) == 1.0
    assert normalized_review_count(5000) == 1.0  # capped
    assert normalized_review_count(500) == 0.5
    assert normalized_review_count(-10) == 0.0  # clamped, never negative
    with pytest.raises(ValueError):
        normalized_review_count(100, cap=0)


def test_cap_evidence_limits_snippets_and_chars():
    c = _cand("p1", "Acme", ["x" * 500, "y" * 500, "z" * 500])
    capped = cap_evidence(c, max_snippets=2, max_chars=10)
    assert len(capped.snippets) == 2
    assert all(len(s) <= 10 for s in capped.snippets)


def test_orderings_are_deterministic_and_cover_candidates():
    cands = [_cand(f"p{i}", "B", ["s"]) for i in range(3)]
    a = orderings(cands, n=3, seed=7)
    b = orderings(cands, n=3, seed=7)
    assert [[c.product_id for c in o] for o in a] == [[c.product_id for c in o] for o in b]
    assert len(a) == 3
    for o in a:
        assert {c.product_id for c in o} == {"p0", "p1", "p2"}
