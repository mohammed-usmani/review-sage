from sage.config import Settings
from sage.eval.run_ablation import ablate, AblationRow
from sage.providers.fakes import FakeEmbeddingProvider, FakeLLMProvider
from sage.vectorstore.base import Filter
from sage.vectorstore.faiss_store import FaissStore


def _store(emb):
    store = FaissStore(dim=emb.dim)
    rows = [("r1", "p1", "Acme", 80, "stays in at the gym sweatproof"),
            ("r2", "p2", "Bolt", 90, "audiophile studio sound")]
    ids, vecs, pl = [], [], []
    for rid, pid, brand, price, text in rows:
        ids.append(rid); vecs.append(emb.embed([text])[0])
        pl.append({"review_id": rid, "product_id": pid, "brand": brand, "category": "earbuds",
                   "price": price, "avg_rating": 4.4, "review_count": 100, "text": text})
    store.upsert(ids, vecs, pl)
    return store


def test_ablate_reports_per_query_candidate_sets():
    emb = FakeEmbeddingProvider(dim=32)
    store = _store(emb)
    rows = ablate(
        queries=["earbuds for the gym"], flt=Filter(category="earbuds"), embedder=emb,
        store=store, make_hyde_llm=lambda: FakeLLMProvider(responses=["stays in at the gym"]),
        settings=Settings(),
    )
    assert len(rows) == 1
    r = rows[0]
    assert isinstance(r, AblationRow)
    assert r.query == "earbuds for the gym"
    assert isinstance(r.naive_products, list) and isinstance(r.hyde_products, list)
    assert 0.0 <= r.jaccard <= 1.0


def test_ablate_jaccard_one_when_identical_results():
    """jaccard == 1.0 when naive and hyde retrieve identical product sets."""
    emb = FakeEmbeddingProvider(dim=32)
    store = _store(emb)
    # hyde generator echoes the exact query text → same embedding → same results
    rows = ablate(
        queries=["earbuds for the gym"], flt=Filter(category="earbuds"), embedder=emb,
        store=store,
        make_hyde_llm=lambda: FakeLLMProvider(responses=["earbuds for the gym"]),
        settings=Settings(),
    )
    assert len(rows) == 1
    r = rows[0]
    assert r.jaccard == 1.0


def test_ablate_jaccard_one_for_empty_union():
    """jaccard == 1.0 when both modes retrieve nothing (empty-union edge case)."""
    emb = FakeEmbeddingProvider(dim=32)
    store = FaissStore(dim=emb.dim)  # empty store → no hits
    rows = ablate(
        queries=["something nobody wrote about"], flt=None, embedder=emb,
        store=store, make_hyde_llm=lambda: FakeLLMProvider(responses=["also nothing"]),
        settings=Settings(),
    )
    assert len(rows) == 1
    assert rows[0].jaccard == 1.0
