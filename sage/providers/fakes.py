from __future__ import annotations

import hashlib
import json
import math
from typing import Sequence


class FakeLLMProvider:
    """Returns scripted responses in order, then a default. Records calls."""

    family = "fake"

    def __init__(self, responses: list[str] | None = None, default: str = "") -> None:
        self._responses = list(responses or [])
        self._default = default
        self.calls: list[str] = []

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        if self._responses:
            return self._responses.pop(0)
        return self._default


class FakeJudgeLLM:
    """A fake judge that returns valid ranking JSON by reading the PRODUCT lines
    in the prompt and keeping their given order. Lets the whole pipeline run with
    zero API keys (e.g. SAGE_JUDGE_PROVIDER=fake) for local smoke runs."""

    family = "fake"

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        ids: list[str] = []
        for line in prompt.splitlines():
            if line.startswith("PRODUCT "):
                ids.append(line.split()[1].strip(":"))
        return json.dumps({"ranking": [{"product_id": p, "reason": "fake judge"} for p in ids]})


class FakeEmbeddingProvider:
    """Deterministic hash-based embedding — same text always maps to same vector."""

    family = "fake"

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        out: list[float] = []
        for i in range(self.dim):
            h = hashlib.sha256(f"{i}:{text}".encode()).digest()
            out.append(int.from_bytes(h[:4], "big") / 2**32)
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        return [x / norm for x in out]
