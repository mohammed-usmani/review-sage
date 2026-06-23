from sage.config import Settings
from sage.runtime import build_store


def test_build_store_faiss():
    s = Settings(vector_backend="faiss", embedding_provider="fake")
    store = build_store(s, dim=16)
    assert store.dim == 16
    # smoke: upsert + search round-trips
    store.upsert(["a"], [[0.5] * 16], [{"product_id": "p", "category": "earbuds", "price": 10,
                                        "review_id": "a", "brand": "X", "avg_rating": 4,
                                        "review_count": 1, "text": "t"}])
    assert store.search([0.5] * 16, top_k=1)[0].id == "a"
