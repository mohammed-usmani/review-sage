from pathlib import Path

import pytest

from sage.ingest import load_records, ingest
from sage.providers.fakes import FakeEmbeddingProvider
from sage.vectorstore.faiss_store import FaissStore

FIXTURE = Path(__file__).parent / "fixtures" / "reviews_sample.jsonl"


def test_load_records_parses_jsonl():
    recs = load_records(FIXTURE)
    assert len(recs) == 2
    assert recs[0]["product_id"] == "p1"


def test_ingest_upserts_searchable_chunks():
    emb = FakeEmbeddingProvider(dim=32)
    store = FaissStore(dim=emb.dim)
    n = ingest(FIXTURE, embedder=emb, store=store, batch_size=10)
    assert n == 2
    hits = store.search(emb.embed(["gym sweat"])[0], top_k=2)
    assert {h.payload["product_id"] for h in hits} == {"p1", "p2"}
    assert "text" in hits[0].payload


def test_ingest_multiple_batches():
    # batch_size < record count exercises the multi-batch loop path.
    emb = FakeEmbeddingProvider(dim=32)
    store = FaissStore(dim=emb.dim)
    n = ingest(FIXTURE, embedder=emb, store=store, batch_size=1)
    assert n == 2
    hits = store.search(emb.embed(["gym sweat"])[0], top_k=2)
    assert {h.payload["product_id"] for h in hits} == {"p1", "p2"}


def test_load_records_raises_on_missing_field(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"review_id": "r99", "product_id": "p99"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_records(bad)
