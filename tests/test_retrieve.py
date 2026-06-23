from sage.providers.fakes import FakeEmbeddingProvider
from sage.retrieve import retrieve_candidates
from sage.vectorstore.base import Filter
from sage.vectorstore.faiss_store import FaissStore


def _seed_store(emb: FakeEmbeddingProvider) -> FaissStore:
    store = FaissStore(dim=emb.dim)
    rows = [
        ("r1", "p1", "Acme", 80, "stays in at the gym, sweatproof"),
        ("r2", "p1", "Acme", 80, "great bass for running"),
        ("r3", "p2", "Bolt", 90, "comfortable for office calls"),
        ("r4", "p3", "Cory", 250, "premium audiophile sound"),
    ]
    ids, vecs, payloads = [], [], []
    for rid, pid, brand, price, text in rows:
        ids.append(rid)
        vecs.append(emb.embed([text])[0])
        payloads.append({
            "review_id": rid, "product_id": pid, "brand": brand, "category": "earbuds",
            "price": price, "avg_rating": 4.4, "review_count": 100, "text": text,
        })
    store.upsert(ids, vecs, payloads)
    return store


def test_retrieve_groups_chunks_by_product():
    emb = FakeEmbeddingProvider(dim=32)
    store = _seed_store(emb)
    cands = retrieve_candidates(
        "stays in at the gym", embedder=emb, store=store,
        flt=Filter(category="earbuds", price_max=100), top_k=10, max_candidates=5, max_snippets=4,
    )
    by_id = {c.product_id: c for c in cands}
    assert "p3" not in by_id  # filtered out by price_max=100
    assert set(by_id["p1"].chunk_ids) <= {"r1", "r2"}
    assert by_id["p1"].brand == "Acme"


def test_retrieve_respects_max_candidates():
    emb = FakeEmbeddingProvider(dim=32)
    store = _seed_store(emb)
    cands = retrieve_candidates(
        "earbuds", embedder=emb, store=store, flt=Filter(category="earbuds"),
        top_k=10, max_candidates=1, max_snippets=4,
    )
    assert len(cands) == 1


def test_retrieve_caps_snippets_but_keeps_product():
    emb = FakeEmbeddingProvider(dim=32)
    store = FaissStore(dim=emb.dim)
    # one product with 3 reviews
    ids, vecs, payloads = [], [], []
    for rid, text in [("r1", "gym fit"), ("r2", "sweatproof"), ("r3", "great bass")]:
        ids.append(rid)
        vecs.append(emb.embed([text])[0])
        payloads.append({
            "review_id": rid, "product_id": "p1", "brand": "Acme", "category": "earbuds",
            "price": 80, "avg_rating": 4.4, "review_count": 100, "text": text,
        })
    store.upsert(ids, vecs, payloads)

    cands = retrieve_candidates(
        "gym", embedder=emb, store=store, flt=Filter(category="earbuds"),
        top_k=10, max_candidates=5, max_snippets=2,
    )
    assert len(cands) == 1  # product still present
    assert len(cands[0].snippets) == 2  # but snippets capped
    assert len(cands[0].chunk_ids) == 2
