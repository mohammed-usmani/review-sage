from sage.providers.fakes import FakeLLMProvider, FakeEmbeddingProvider


def test_fake_embedding_is_deterministic():
    emb = FakeEmbeddingProvider(dim=16)
    a = emb.embed(["hello world"])
    b = emb.embed(["hello world"])
    assert a == b
    assert len(a) == 1 and len(a[0]) == 16


def test_fake_embedding_distinguishes_text():
    emb = FakeEmbeddingProvider(dim=16)
    assert emb.embed(["gym earbuds"])[0] != emb.embed(["office headphones"])[0]


def test_fake_llm_returns_scripted_then_default():
    llm = FakeLLMProvider(responses=["first"], default="fallback")
    assert llm.generate("p1") == "first"
    assert llm.generate("p2") == "fallback"
    assert llm.family == "fake"
    assert llm.calls == ["p1", "p2"]
