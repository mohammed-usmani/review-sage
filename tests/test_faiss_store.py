from sage.vectorstore.base import Filter
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
