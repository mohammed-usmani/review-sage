import sys
import types

from sage.providers.anthropic import AnthropicLLM
from sage.providers.gemini import GeminiEmbedding, GeminiLLM
from sage.providers.openai import OpenAIEmbedding, OpenAILLM
from sage.providers.vertex import VertexLLM


def test_anthropic_generate_calls_sdk(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="VERDICT")])

    class FakeClient:
        def __init__(self, *a, **k):
            self.messages = FakeMessages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = FakeClient
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    llm = AnthropicLLM("claude-haiku-4-5-20251001")
    out = llm.generate("rank these", temperature=0.0)
    assert out == "VERDICT"
    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["max_tokens"] == 1024
    assert captured["temperature"] == 0.0
    assert captured["messages"] == [{"role": "user", "content": "rank these"}]
    assert llm.family == "anthropic"


def _install_openai(monkeypatch, captured):
    class FakeChat:
        def __init__(self):
            self.completions = self

        def create(self, **kwargs):
            captured["chat"] = kwargs
            msg = types.SimpleNamespace(content="OPENAI_OUT")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class FakeEmbeddings:
        def create(self, **kwargs):
            captured["embed"] = kwargs
            return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[0.1, 0.2])])

    class FakeClient:
        def __init__(self, *a, **k):
            self.chat = FakeChat()
            self.embeddings = FakeEmbeddings()

    fake_mod = types.ModuleType("openai")
    fake_mod.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_mod)


def test_openai_llm_and_embedding(monkeypatch):
    captured = {}
    _install_openai(monkeypatch, captured)

    llm = OpenAILLM("gpt-x")
    assert llm.generate("hi", temperature=0.0) == "OPENAI_OUT"
    assert captured["chat"]["model"] == "gpt-x"
    assert llm.family == "openai"

    emb = OpenAIEmbedding("text-embed")
    vecs = emb.embed(["a", "b"])
    assert vecs == [[0.1, 0.2]]
    assert captured["embed"]["input"] == ["a", "b"]
    assert emb.dim == 1536
    assert emb.family == "openai"


def _install_genai(monkeypatch, captured, *, expect_vertex=False):
    class FakeModels:
        def generate_content(self, **kwargs):
            captured["gen"] = kwargs
            return types.SimpleNamespace(text="GEMINI_OUT")

        def embed_content(self, **kwargs):
            captured["embed"] = kwargs
            return types.SimpleNamespace(embeddings=[types.SimpleNamespace(values=[0.3, 0.4])])

    class FakeClient:
        def __init__(self, *a, **k):
            captured["client_kwargs"] = k
            self.models = FakeModels()

    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = FakeClient
    google_mod = types.ModuleType("google")
    google_mod.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)


def test_gemini_llm_and_embedding(monkeypatch):
    captured = {}
    _install_genai(monkeypatch, captured)

    llm = GeminiLLM("gemini-2.0-flash")
    assert llm.generate("hi", temperature=0.0) == "GEMINI_OUT"
    assert captured["gen"]["model"] == "gemini-2.0-flash"
    assert captured["gen"]["config"] == {"temperature": 0.0}
    assert llm.family == "gemini"
    assert captured["client_kwargs"] == {}  # plain (non-vertex) client

    emb = GeminiEmbedding("gemini-embedding-001")
    assert emb.embed(["x"]) == [[0.3, 0.4]]
    assert emb.dim == 768
    assert emb.family == "gemini"
    # must request truncation to the index dimension, else real vectors are 3072-d
    assert captured["embed"]["config"] == {"output_dimensionality": 768}


def test_vertex_uses_vertexai_client(monkeypatch):
    captured = {}
    _install_genai(monkeypatch, captured)

    llm = VertexLLM("gemini-2.0-flash")
    assert llm.generate("hi") == "GEMINI_OUT"
    assert captured["client_kwargs"] == {"vertexai": True}
    assert llm.family == "vertex"


def test_vertex_embedding_uses_vertexai_client(monkeypatch):
    from sage.providers.gemini import VertexEmbedding

    captured = {}
    _install_genai(monkeypatch, captured)
    emb = VertexEmbedding("gemini-embedding-001")
    assert emb.embed(["x"]) == [[0.3, 0.4]]
    assert captured["client_kwargs"] == {"vertexai": True}
    assert emb.family == "vertex"
