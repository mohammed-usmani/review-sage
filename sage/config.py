from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SAGE_", env_file=".env", extra="ignore")

    # embeddings
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-001"

    # hyde generator
    hyde_provider: str = "gemini"
    hyde_model: str = "gemini-2.0-flash"

    # judge
    judge_provider: str = "anthropic"
    judge_model: str = "claude-haiku-4-5-20251001"
    allow_same_family_judge: bool = False  # self-preference guard

    # vector store
    vector_backend: str = "qdrant"  # "qdrant" | "faiss"
    qdrant_url: str = "http://localhost:6333"
    collection: str = "reviews"

    # retrieval / judging
    top_k: int = 40
    max_candidates: int = 6
    max_snippets: int = 4
    max_snippet_chars: int = 280
    position_orderings: int = 3
    seed: int = 7
