import pytest

from sage.vectorstore.base import Filter, VectorStore
from sage.vectorstore.faiss_store import FaissStore


def _vec(seed: float, dim: int = 8) -> list[float]:
    return [seed] * dim


def test_search_returns_nearest_first():
    store = FaissStore(dim=8)
    store.upsert(
        ids=["a", "b"],
        vectors=[_vec(0.1), _vec(0.9)],
        payloads=[{"product_id": "pa", "price": 50}, {"product_id": "pb", "price": 50}],
    )
    hits = store.search(_vec(0.9), top_k=2)
    assert hits[0].id == "b"
    assert hits[0].payload["product_id"] == "pb"


def test_search_applies_price_filter():
    store = FaissStore(dim=8)
    store.upsert(
        ids=["a", "b"],
        vectors=[_vec(0.5), _vec(0.5)],
        payloads=[{"category": "earbuds", "price": 200}, {"category": "earbuds", "price": 80}],
    )
    flt = Filter(category="earbuds", price_max=100)
    hits = store.search(_vec(0.5), top_k=5, flt=flt)
    assert [h.id for h in hits] == ["b"]


def test_empty_store_returns_empty():
    store = FaissStore(dim=8)
    assert store.search(_vec(0.5), top_k=3) == []


def test_filter_removing_all_returns_empty():
    store = FaissStore(dim=8)
    store.upsert(ids=["a"], vectors=[_vec(0.5)], payloads=[{"category": "earbuds", "price": 200}])
    assert store.search(_vec(0.5), top_k=5, flt=Filter(price_max=100)) == []


def test_top_k_exceeds_store_size():
    store = FaissStore(dim=8)
    store.upsert(ids=["a"], vectors=[_vec(0.5)], payloads=[{"product_id": "p"}])
    hits = store.search(_vec(0.5), top_k=10)
    assert [h.id for h in hits] == ["a"]


def test_selective_filter_finds_low_similarity_match():
    # A rare item that ranks below the over-fetch window must still be found.
    store = FaissStore(dim=4)
    common = [[1.0, 0.0, 0.0, 0.0]] * 30
    store.upsert(
        ids=[f"c{i}" for i in range(30)],
        vectors=common,
        payloads=[{"category": "common", "price": 50} for _ in range(30)],
    )
    store.upsert(ids=["rare"], vectors=[[0.0, 1.0, 0.0, 0.0]], payloads=[{"category": "rare", "price": 50}])
    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=1, flt=Filter(category="rare"))
    assert [h.id for h in hits] == ["rare"]


def test_upsert_rejects_length_mismatch():
    store = FaissStore(dim=8)
    with pytest.raises(ValueError):
        store.upsert(ids=["a"], vectors=[_vec(0.1), _vec(0.2)], payloads=[{}, {}])


def test_faiss_store_satisfies_protocol():
    assert isinstance(FaissStore(dim=8), VectorStore)
