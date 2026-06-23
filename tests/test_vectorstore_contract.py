import os

import pytest

from sage.vectorstore.base import Filter
from sage.vectorstore.faiss_store import FaissStore


def _vec(seed: float, dim: int = 8) -> list[float]:
    return [seed] * dim


def _exercise(store):
    store.upsert(
        ids=["a", "b"],
        vectors=[_vec(0.2), _vec(0.9)],
        payloads=[{"category": "earbuds", "price": 200}, {"category": "earbuds", "price": 80}],
    )
    top = store.search(_vec(0.9), top_k=1)
    assert top[0].id == "b"
    filtered = store.search(_vec(0.5), top_k=5, flt=Filter(price_max=100))
    assert [h.id for h in filtered] == ["b"]


def test_faiss_contract():
    _exercise(FaissStore(dim=8))


@pytest.mark.skipif(os.environ.get("SAGE_RUN_QDRANT") != "1", reason="needs Qdrant")
def test_qdrant_contract():
    from sage.vectorstore.qdrant_store import QdrantStore

    store = QdrantStore(dim=8, url="http://localhost:6333", collection="contract_test")
    store.reset()
    _exercise(store)
