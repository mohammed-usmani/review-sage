from __future__ import annotations

import uuid

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

    @staticmethod
    def _point_id(ext_id: str) -> str:
        # Deterministic UUID per external id: collision-safe and idempotent on
        # re-upsert (the same review id always maps to the same point).
        return str(uuid.uuid5(uuid.NAMESPACE_URL, ext_id))

    def upsert(self, ids, vectors, payloads) -> None:
        from qdrant_client.models import PointStruct

        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError(
                f"upsert length mismatch: {len(ids)} ids, {len(vectors)} vectors, "
                f"{len(payloads)} payloads"
            )
        points = [
            PointStruct(id=self._point_id(ext), vector=list(v), payload={**p, "_ext_id": ext})
            for ext, v, p in zip(ids, vectors, payloads)
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
        res = self._client.query_points(
            self.collection,
            query=list(vector),
            limit=top_k,
            query_filter=self._build_filter(flt),
            with_payload=True,
        )
        out = []
        for p in res.points:
            payload = dict(p.payload or {})
            ext = payload.pop("_ext_id", str(p.id))
            out.append(SearchHit(id=ext, score=float(p.score), payload=payload))
        return out
