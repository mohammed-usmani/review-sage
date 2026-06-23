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
        text = resp.text
        if text is None:
            raise RuntimeError("Gemini returned no text (empty or blocked completion)")
        return text


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
        # Request output truncated to self.dim. gemini-embedding-001 defaults to
        # 3072 dims; without this the returned vectors would not match the index
        # dimension we build the vector store with.
        resp = client.models.embed_content(
            model=self.model,
            contents=list(texts),
            config={"output_dimensionality": self.dim},
        )
        return [e.values for e in resp.embeddings]


class VertexEmbedding(GeminiEmbedding):
    """Gemini embeddings on Vertex AI (uses google-genai with the Vertex backend)."""

    family = "vertex"

    def _ensure(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(vertexai=True)
        return self._client
