from __future__ import annotations


class AnthropicLLM:
    family = "anthropic"

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _ensure(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        client = self._ensure()
        msg = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
