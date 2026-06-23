from __future__ import annotations

import json
from pathlib import Path

from sage.providers.base import EmbeddingProvider
from sage.vectorstore.base import VectorStore

_REQUIRED = ("review_id", "product_id", "brand", "category", "price", "avg_rating",
             "review_count", "text")


def load_records(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = [k for k in _REQUIRED if k not in rec]
            if missing:
                raise ValueError(f"record {rec.get('review_id')} missing fields: {missing}")
            records.append(rec)
    return records


def ingest(
    path: str | Path,
    *,
    embedder: EmbeddingProvider,
    store: VectorStore,
    batch_size: int = 128,
) -> int:
    records = load_records(path)
    total = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        vectors = embedder.embed([r["text"] for r in batch])
        ids = [r["review_id"] for r in batch]
        payloads = [{k: r[k] for k in _REQUIRED} for r in batch]
        store.upsert(ids, vectors, payloads)
        total += len(batch)
    return total
