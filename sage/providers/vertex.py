from __future__ import annotations

from sage.providers.gemini import GeminiLLM


class VertexLLM(GeminiLLM):
    """Gemini on Vertex AI. Uses google-genai with Vertex backend via env
    (GOOGLE_GENAI_USE_VERTEXAI=true, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION)."""

    family = "vertex"

    def _ensure(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(vertexai=True)
        return self._client
