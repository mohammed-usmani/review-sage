from sage.config import Settings


def test_defaults_match_spec():
    s = Settings()
    assert s.embedding_provider == "gemini"
    assert s.embedding_model == "gemini-embedding-001"
    assert s.hyde_provider == "gemini"
    assert s.judge_provider == "anthropic"
    assert s.vector_backend == "qdrant"
    assert s.top_k == 40
    assert s.position_orderings == 3
    assert s.seed == 7


def test_env_override(monkeypatch):
    monkeypatch.setenv("SAGE_JUDGE_PROVIDER", "openai")
    s = Settings()
    assert s.judge_provider == "openai"
