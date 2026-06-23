from __future__ import annotations

from sage.config import Settings
from sage.vectorstore.base import VectorStore


def build_store(settings: Settings, dim: int) -> VectorStore:
    if settings.vector_backend == "faiss":
        from sage.vectorstore.faiss_store import FaissStore

        return FaissStore(dim=dim)
    if settings.vector_backend == "qdrant":
        from sage.vectorstore.qdrant_store import QdrantStore

        return QdrantStore(dim=dim, url=settings.qdrant_url, collection=settings.collection)
    raise ValueError(f"unknown vector backend: {settings.vector_backend}")
