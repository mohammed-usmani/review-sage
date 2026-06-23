from __future__ import annotations

from typing import Sequence

from sage.vectorstore.base import Filter, SearchHit


class QdrantStore:
    def __init__(self, dim: int, url: str, collection: str) -> None:
        from qdrant_client import QdrantClient

        self.dim = dim
        self.collection = collection
        self._client = QdrantClient(url=url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def reset(self) -> None:
        self._client.delete_collection(self.collection)
        self._ensure_collection()

    def upsert(self, ids, vectors, payloads) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=i, vector=list(v), payload={**p, "_ext_id": ext})
            for i, (ext, v, p) in enumerate(zip(ids, vectors, payloads))
        ]
        self._client.upsert(self.collection, points=points)

    def _build_filter(self, flt: Filter | None):
        from qdrant_client.models import FieldCondition, Filter as QFilter, MatchValue, Range

        if flt is None:
            return None
        must = []
        if flt.category is not None:
            must.append(FieldCondition(key="category", match=MatchValue(value=flt.category)))
        if flt.price_max is not None:
            must.append(FieldCondition(key="price", range=Range(lte=flt.price_max)))
        return QFilter(must=must) if must else None

    def search(self, vector, top_k, flt: Filter | None = None) -> list[SearchHit]:
        res = self._client.search(
            self.collection,
            query_vector=list(vector),
            limit=top_k,
            query_filter=self._build_filter(flt),
            with_payload=True,
        )
        out = []
        for p in res:
            payload = dict(p.payload or {})
            ext = payload.pop("_ext_id", str(p.id))
            out.append(SearchHit(id=ext, score=float(p.score), payload=payload))
        return out
