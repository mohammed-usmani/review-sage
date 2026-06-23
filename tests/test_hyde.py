from sage.hyde import embedding_text, HYDE_PROMPT
from sage.providers.fakes import FakeLLMProvider


def test_naive_mode_returns_raw_query():
    gen = FakeLLMProvider(responses=["SHOULD NOT BE USED"])
    assert embedding_text("good gym earbuds", mode="naive", generator=gen) == "good gym earbuds"
    assert gen.calls == []  # generator untouched in naive mode


def test_hyde_mode_generates_hypothetical_doc():
    gen = FakeLLMProvider(responses=["These earbuds stay locked in during burpees..."])
    out = embedding_text("good gym earbuds", mode="hyde", generator=gen)
    assert out == "These earbuds stay locked in during burpees..."
    assert "good gym earbuds" in gen.calls[0]
    assert HYDE_PROMPT  # prompt constant exists


def test_unknown_mode_raises_value_error():
    gen = FakeLLMProvider()
    try:
        embedding_text("query", mode="bogus", generator=gen)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "bogus" in str(e)
