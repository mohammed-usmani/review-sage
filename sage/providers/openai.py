from __future__ import annotations

from typing import Sequence


class OpenAILLM:
    family = "openai"

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _ensure(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI()
        return self._client

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        client = self._ensure()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI returned no content (empty or refused completion)")
        return content


class OpenAIEmbedding:
    family = "openai"
    dim = 1536

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _ensure(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI()
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        client = self._ensure()
        resp = client.embeddings.create(model=self.model, input=list(texts))
        return [d.embedding for d in resp.data]
