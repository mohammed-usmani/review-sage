from __future__ import annotations

from typing import Sequence


class GeminiLLM:
    family = "gemini"

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _ensure(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client()
        return self._client

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        client = self._ensure()
        resp = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": temperature},
        )
        return resp.text


class GeminiEmbedding:
    family = "gemini"
    dim = 768

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _ensure(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client()
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        client = self._ensure()
        resp = client.models.embed_content(model=self.model, contents=list(texts))
        return [e.values for e in resp.embeddings]
