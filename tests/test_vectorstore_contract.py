import os

import pytest

from sage.vectorstore.base import Filter
from sage.vectorstore.faiss_store import FaissStore


def _onehot(i: int, dim: int = 8) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


def _exercise(store):
    # Non-collinear vectors so cosine similarity actually distinguishes them:
    # "b" aligns with the query, "a" is orthogonal. (Collinear vectors normalize
    # to the same unit vector and tie at score 1.0, making ordering backend-defined.)
    store.upsert(
        ids=["a", "b"],
        vectors=[_onehot(1), _onehot(0)],
        payloads=[{"category": "earbuds", "price": 200}, {"category": "earbuds", "price": 80}],
    )
    top = store.search(_onehot(0), top_k=1)
    assert top[0].id == "b"
    filtered = store.search(_onehot(0), top_k=5, flt=Filter(price_max=100))
    assert [h.id for h in filtered] == ["b"]


def test_faiss_contract():
    _exercise(FaissStore(dim=8))


@pytest.mark.skipif(os.environ.get("SAGE_RUN_QDRANT") != "1", reason="needs Qdrant")
def test_qdrant_contract():
    from sage.vectorstore.qdrant_store import QdrantStore

    store = QdrantStore(dim=8, url="http://localhost:6333", collection="contract_test")
    store.reset()
    _exercise(store)
