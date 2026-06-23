from __future__ import annotations

import warnings

from sage.config import Settings
from sage.providers.base import EmbeddingProvider, LLMProvider
from sage.providers.fakes import FakeEmbeddingProvider, FakeJudgeLLM, FakeLLMProvider


class SelfPreferenceError(RuntimeError):
    """Raised when judge and generator share a model family and that is not allowed."""


def assert_distinct_families(*, hyde_family: str, judge_family: str, allow_same: bool) -> None:
    if hyde_family == judge_family:
        msg = (
            f"Self-preference risk: judge and HyDE generator share family "
            f"'{judge_family}'. Set allow_same_family_judge=True to override."
        )
        if not allow_same:
            raise SelfPreferenceError(msg)
        warnings.warn(msg, stacklevel=2)


def build_llm(role: str, settings: Settings) -> LLMProvider:
    provider = getattr(settings, f"{role}_provider")
    model = getattr(settings, f"{role}_model")
    if provider == "fake":
        # The judge needs to emit valid ranking JSON; a generic stub string would
        # crash json parsing. This makes the all-fake, zero-key path runnable.
        if role == "judge":
            return FakeJudgeLLM()
        return FakeLLMProvider(default=f"fake-{role}")
    if provider == "gemini":
        from sage.providers.gemini import GeminiLLM

        return GeminiLLM(model)
    if provider == "vertex":
        from sage.providers.vertex import VertexLLM

        return VertexLLM(model)
    if provider == "openai":
        from sage.providers.openai import OpenAILLM

        return OpenAILLM(model)
    if provider == "anthropic":
        from sage.providers.anthropic import AnthropicLLM

        return AnthropicLLM(model)
    raise ValueError(f"unknown LLM provider: {provider}")


def build_embedder(settings: Settings) -> EmbeddingProvider:
    provider = settings.embedding_provider
    if provider == "fake":
        return FakeEmbeddingProvider()
    if provider == "gemini":
        from sage.providers.gemini import GeminiEmbedding

        return GeminiEmbedding(settings.embedding_model)
    if provider == "vertex":
        from sage.providers.gemini import VertexEmbedding

        return VertexEmbedding(settings.embedding_model)
    if provider == "openai":
        from sage.providers.openai import OpenAIEmbedding

        return OpenAIEmbedding(settings.embedding_model)
    raise ValueError(f"unknown embedding provider: {provider}")


def build_role_llms(settings: Settings) -> tuple[LLMProvider, LLMProvider]:
    """Build the (hyde_generator, judge) LLM pair and enforce the self-preference
    guard. This is the composition entry point callers should use so the judge is
    never the same family as the generator unless explicitly allowed."""
    hyde_llm = build_llm("hyde", settings)
    judge_llm = build_llm("judge", settings)
    assert_distinct_families(
        hyde_family=hyde_llm.family,
        judge_family=judge_llm.family,
        allow_same=settings.allow_same_family_judge,
    )
    return hyde_llm, judge_llm
