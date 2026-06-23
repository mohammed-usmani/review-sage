import sys
import types

from sage.providers.anthropic import AnthropicLLM


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
    assert llm.family == "anthropic"
