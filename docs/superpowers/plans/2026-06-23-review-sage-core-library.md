# Review-Sage Core Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure, testable core library for Review-Sage — a headphones purchase advisor that retrieves real reviews via HyDE and ranks candidate products with a bias-controlled LLM-as-judge, exposed as a CLI and an offline ablation script.

**Architecture:** A layered library (`sage/`) with swappable provider and vector-store interfaces. Provider selection is per-role (HyDE generator / judge / embeddings) so the judge can be a different model family than the generator (self-preference control). One `pipeline.recommend()` orchestrator is the single entry point; the CLI and eval script are thin callers. All logic is tested with deterministic fake providers and an in-process FAISS store — zero API calls in CI.

**Tech Stack:** Python 3.11+, pydantic-settings, numpy, faiss-cpu, qdrant-client, google-genai / openai / anthropic (lazy-imported), pytest, ruff. Web/API layers are a separate plan.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Package metadata, deps, ruff/pytest config |
| `sage/models.py` | Dataclasses: `ReviewChunk`, `Candidate`, `RankedProduct`, `RecommendResult` |
| `sage/config.py` | `Settings` (pydantic-settings): per-role models, vector backend, top-k, seed |
| `sage/providers/base.py` | `LLMProvider` / `EmbeddingProvider` protocols |
| `sage/providers/fakes.py` | Deterministic `FakeLLMProvider` / `FakeEmbeddingProvider` |
| `sage/providers/factory.py` | Per-role provider resolution + self-preference assertion |
| `sage/providers/gemini.py` etc. | Real adapters (lazy SDK import) |
| `sage/vectorstore/base.py` | `SearchHit`, `VectorStore` protocol, `Filter` |
| `sage/vectorstore/faiss_store.py` | In-process backend (over-fetch + payload filter) |
| `sage/vectorstore/qdrant_store.py` | Qdrant backend (native metadata filter) |
| `sage/ingest.py` | Amazon Reviews subset → chunk → embed → upsert |
| `sage/hyde.py` | query → hypothetical review text (or naive passthrough) |
| `sage/retrieve.py` | embed + search + group hits into `Candidate`s |
| `sage/judge/bias_controls.py` | position / verbosity / brand-blind / self-preference helpers |
| `sage/judge/judge.py` | Listwise ranking under active bias controls |
| `sage/pipeline.py` | `recommend()` + `compare()` orchestrators |
| `sage/cli.py` | `seed` / `recommend` / `eval` commands |
| `sage/eval/run_ablation.py` | naive vs HyDE on seed queries |

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `sage/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

`tests/test_smoke.py`:
```python
import sage


def test_package_imports():
    assert sage.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sage'`.

- [ ] **Step 3: Create the package and config**

`pyproject.toml`:
```toml
[project]
name = "review-sage"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "numpy>=1.26",
    "faiss-cpu>=1.8",
    "qdrant-client>=1.9",
    "typer>=0.12",
]

[project.optional-dependencies]
gemini = ["google-genai>=0.3"]
openai = ["openai>=1.30"]
anthropic = ["anthropic>=0.30"]
dev = ["pytest>=8.0", "ruff>=0.5"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["sage*"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
```

`sage/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/__init__.py`: empty file.

- [ ] **Step 4: Install and run test to verify it passes**

Run: `pip install -e ".[dev]" && python -m pytest tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml sage/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold review-sage package"
```

---

## Task 2: Data models

**Files:**
- Create: `sage/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
from sage.models import ReviewChunk, Candidate, RankedProduct, RecommendResult


def test_candidate_from_chunks_aggregates():
    chunks = [
        ReviewChunk(review_id="r1", product_id="p1", brand="Acme", category="earbuds",
                    price=80.0, avg_rating=4.5, review_count=120, text="great bass"),
        ReviewChunk(review_id="r2", product_id="p1", brand="Acme", category="earbuds",
                    price=80.0, avg_rating=4.5, review_count=120, text="stays in at gym"),
    ]
    cand = Candidate.from_chunks("p1", chunks)
    assert cand.product_id == "p1"
    assert cand.brand == "Acme"
    assert cand.price == 80.0
    assert cand.snippets == ["great bass", "stays in at gym"]
    assert cand.chunk_ids == ["r1", "r2"]


def test_recommend_result_holds_ranking():
    cand = Candidate(product_id="p1", brand="Acme", category="earbuds", price=80.0,
                     avg_rating=4.5, review_count=120, snippets=["x"], chunk_ids=["r1"])
    rp = RankedProduct(candidate=cand, rank=1, reasoning="best fit", cited_snippet_ids=["r1"])
    res = RecommendResult(query="q", mode="hyde", bias_on=True, ranked=[rp])
    assert res.ranked[0].rank == 1
    assert res.mode == "hyde"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sage.models'`.

- [ ] **Step 3: Write the implementation**

`sage/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReviewChunk:
    review_id: str
    product_id: str
    brand: str
    category: str
    price: float
    avg_rating: float
    review_count: int
    text: str


@dataclass
class Candidate:
    product_id: str
    brand: str
    category: str
    price: float
    avg_rating: float
    review_count: int
    snippets: list[str]
    chunk_ids: list[str]

    @classmethod
    def from_chunks(cls, product_id: str, chunks: list[ReviewChunk]) -> "Candidate":
        first = chunks[0]
        return cls(
            product_id=product_id,
            brand=first.brand,
            category=first.category,
            price=first.price,
            avg_rating=first.avg_rating,
            review_count=first.review_count,
            snippets=[c.text for c in chunks],
            chunk_ids=[c.review_id for c in chunks],
        )


@dataclass
class RankedProduct:
    candidate: Candidate
    rank: int
    reasoning: str
    cited_snippet_ids: list[str] = field(default_factory=list)


@dataclass
class RecommendResult:
    query: str
    mode: str  # "naive" | "hyde"
    bias_on: bool
    ranked: list[RankedProduct]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/models.py tests/test_models.py
git commit -m "feat: add core data models"
```

---

## Task 3: Provider protocols + deterministic fakes

**Files:**
- Create: `sage/providers/__init__.py` (empty)
- Create: `sage/providers/base.py`
- Create: `sage/providers/fakes.py`
- Test: `tests/test_fakes.py`

- [ ] **Step 1: Write the failing test**

`tests/test_fakes.py`:
```python
from sage.providers.fakes import FakeLLMProvider, FakeEmbeddingProvider


def test_fake_embedding_is_deterministic():
    emb = FakeEmbeddingProvider(dim=16)
    a = emb.embed(["hello world"])
    b = emb.embed(["hello world"])
    assert a == b
    assert len(a) == 1 and len(a[0]) == 16


def test_fake_embedding_distinguishes_text():
    emb = FakeEmbeddingProvider(dim=16)
    assert emb.embed(["gym earbuds"])[0] != emb.embed(["office headphones"])[0]


def test_fake_llm_returns_scripted_then_default():
    llm = FakeLLMProvider(responses=["first"], default="fallback")
    assert llm.generate("p1") == "first"
    assert llm.generate("p2") == "fallback"
    assert llm.family == "fake"
    assert llm.calls == ["p1", "p2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fakes.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`sage/providers/base.py`:
```python
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    family: str

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    family: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
```

`sage/providers/fakes.py`:
```python
from __future__ import annotations

import hashlib
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fakes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/providers/
git add tests/test_fakes.py
git commit -m "feat: add provider protocols and deterministic fakes"
```

---

## Task 4: Settings

**Files:**
- Create: `sage/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from sage.config import Settings


def test_defaults_match_spec():
    s = Settings()
    assert s.embedding_provider == "gemini"
    assert s.embedding_model == "gemini-embedding-001"
    assert s.hyde_provider == "gemini"
    assert s.judge_provider == "anthropic"
    assert s.vector_backend == "qdrant"
    assert s.top_k == 40
    assert s.position_orderings == 3
    assert s.seed == 7


def test_env_override(monkeypatch):
    monkeypatch.setenv("SAGE_JUDGE_PROVIDER", "openai")
    s = Settings()
    assert s.judge_provider == "openai"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`sage/config.py`:
```python
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SAGE_", env_file=".env", extra="ignore")

    # embeddings
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-001"

    # hyde generator
    hyde_provider: str = "gemini"
    hyde_model: str = "gemini-2.0-flash"

    # judge
    judge_provider: str = "anthropic"
    judge_model: str = "claude-haiku-4-5-20251001"
    allow_same_family_judge: bool = False  # self-preference guard

    # vector store
    vector_backend: str = "qdrant"  # "qdrant" | "faiss"
    qdrant_url: str = "http://localhost:6333"
    collection: str = "reviews"

    # retrieval / judging
    top_k: int = 40
    max_candidates: int = 6
    max_snippets: int = 4
    max_snippet_chars: int = 280
    position_orderings: int = 3
    seed: int = 7
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/config.py tests/test_config.py
git commit -m "feat: add settings with per-role model config"
```

---

## Task 5: Provider factory with self-preference assertion

**Files:**
- Create: `sage/providers/factory.py`
- Test: `tests/test_factory.py`

- [ ] **Step 1: Write the failing test**

`tests/test_factory.py`:
```python
import pytest

from sage.config import Settings
from sage.providers.factory import build_llm, SelfPreferenceError, assert_distinct_families


def test_assert_distinct_families_raises_on_same():
    with pytest.raises(SelfPreferenceError):
        assert_distinct_families(hyde_family="gemini", judge_family="gemini", allow_same=False)


def test_assert_distinct_families_allows_when_opted_in():
    # should not raise
    assert_distinct_families(hyde_family="gemini", judge_family="gemini", allow_same=True)


def test_build_llm_uses_fake_when_requested():
    s = Settings(hyde_provider="fake")
    llm = build_llm("hyde", s)
    assert llm.family == "fake"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_factory.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`sage/providers/factory.py`:
```python
from __future__ import annotations

import warnings

from sage.config import Settings
from sage.providers.base import EmbeddingProvider, LLMProvider
from sage.providers.fakes import FakeEmbeddingProvider, FakeLLMProvider


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
    if provider in ("gemini", "vertex"):
        from sage.providers.gemini import GeminiEmbedding

        return GeminiEmbedding(settings.embedding_model)
    if provider == "openai":
        from sage.providers.openai import OpenAIEmbedding

        return OpenAIEmbedding(settings.embedding_model)
    raise ValueError(f"unknown embedding provider: {provider}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_factory.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/providers/factory.py tests/test_factory.py
git commit -m "feat: add provider factory with self-preference guard"
```

---

## Task 6: Real provider adapters (lazy SDK import)

These are thin wrappers; they are exercised by unit tests that monkeypatch the SDK client so no network/keys are needed. Each adapter exposes `family` and the protocol method.

**Files:**
- Create: `sage/providers/gemini.py`
- Create: `sage/providers/openai.py`
- Create: `sage/providers/anthropic.py`
- Create: `sage/providers/vertex.py`
- Test: `tests/test_adapters.py`

- [ ] **Step 1: Write the failing test**

`tests/test_adapters.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_adapters.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementations**

`sage/providers/anthropic.py`:
```python
from __future__ import annotations

from typing import Sequence


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
```

`sage/providers/openai.py`:
```python
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
        return resp.choices[0].message.content


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
```

`sage/providers/gemini.py`:
```python
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
```

`sage/providers/vertex.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_adapters.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/providers/gemini.py sage/providers/openai.py sage/providers/anthropic.py sage/providers/vertex.py tests/test_adapters.py
git commit -m "feat: add real provider adapters with lazy SDK import"
```

---

## Task 7: VectorStore interface + FAISS backend

**Files:**
- Create: `sage/vectorstore/__init__.py` (empty)
- Create: `sage/vectorstore/base.py`
- Create: `sage/vectorstore/faiss_store.py`
- Test: `tests/test_faiss_store.py`

- [ ] **Step 1: Write the failing test**

`tests/test_faiss_store.py`:
```python
from sage.vectorstore.base import Filter
from sage.vectorstore.faiss_store import FaissStore


def _vec(seed: float, dim: int = 8) -> list[float]:
    return [seed] * dim


def test_search_returns_nearest_first():
    store = FaissStore(dim=8)
    store.upsert(
        ids=["a", "b"],
        vectors=[_vec(0.1), _vec(0.9)],
        payloads=[{"product_id": "pa", "price": 50}, {"product_id": "pb", "price": 50}],
    )
    hits = store.search(_vec(0.9), top_k=2)
    assert hits[0].id == "b"
    assert hits[0].payload["product_id"] == "pb"


def test_search_applies_price_filter():
    store = FaissStore(dim=8)
    store.upsert(
        ids=["a", "b"],
        vectors=[_vec(0.5), _vec(0.5)],
        payloads=[{"category": "earbuds", "price": 200}, {"category": "earbuds", "price": 80}],
    )
    flt = Filter(category="earbuds", price_max=100)
    hits = store.search(_vec(0.5), top_k=5, flt=flt)
    assert [h.id for h in hits] == ["b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_faiss_store.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`sage/vectorstore/base.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class Filter:
    category: str | None = None
    price_max: float | None = None

    def matches(self, payload: dict) -> bool:
        if self.category is not None and payload.get("category") != self.category:
            return False
        if self.price_max is not None and payload.get("price", float("inf")) > self.price_max:
            return False
        return True


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict


class VectorStore(Protocol):
    dim: int

    def upsert(
        self, ids: Sequence[str], vectors: Sequence[Sequence[float]], payloads: Sequence[dict]
    ) -> None: ...

    def search(
        self, vector: Sequence[float], top_k: int, flt: Filter | None = None
    ) -> list[SearchHit]: ...
```

`sage/vectorstore/faiss_store.py`:
```python
from __future__ import annotations

from typing import Sequence

import numpy as np

from sage.vectorstore.base import Filter, SearchHit


class FaissStore:
    """In-process cosine store. Over-fetches then applies payload filter in Python."""

    def __init__(self, dim: int) -> None:
        import faiss

        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._payloads: list[dict] = []

    @staticmethod
    def _normed(vectors: Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def upsert(self, ids, vectors, payloads) -> None:
        self._index.add(self._normed(vectors))
        self._ids.extend(ids)
        self._payloads.extend(payloads)

    def search(self, vector, top_k, flt: Filter | None = None) -> list[SearchHit]:
        if not self._ids:
            return []
        over = min(len(self._ids), max(top_k * 5, top_k))
        q = self._normed([vector])
        scores, idxs = self._index.search(q, over)
        out: list[SearchHit] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            payload = self._payloads[idx]
            if flt is not None and not flt.matches(payload):
                continue
            out.append(SearchHit(id=self._ids[idx], score=float(score), payload=payload))
            if len(out) >= top_k:
                break
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_faiss_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/vectorstore/__init__.py sage/vectorstore/base.py sage/vectorstore/faiss_store.py tests/test_faiss_store.py
git commit -m "feat: add VectorStore interface and FAISS backend"
```

---

## Task 8: Qdrant backend + shared contract test

**Files:**
- Create: `sage/vectorstore/qdrant_store.py`
- Create: `tests/test_vectorstore_contract.py`

The contract test runs the same assertions against any backend. FAISS runs in CI; Qdrant runs only when `SAGE_RUN_QDRANT=1` (skipped otherwise) so CI needs no Docker.

- [ ] **Step 1: Write the failing test**

`tests/test_vectorstore_contract.py`:
```python
import os

import pytest

from sage.vectorstore.base import Filter
from sage.vectorstore.faiss_store import FaissStore


def _vec(seed: float, dim: int = 8) -> list[float]:
    return [seed] * dim


def _exercise(store):
    store.upsert(
        ids=["a", "b"],
        vectors=[_vec(0.2), _vec(0.9)],
        payloads=[{"category": "earbuds", "price": 200}, {"category": "earbuds", "price": 80}],
    )
    top = store.search(_vec(0.9), top_k=1)
    assert top[0].id == "b"
    filtered = store.search(_vec(0.5), top_k=5, flt=Filter(price_max=100))
    assert [h.id for h in filtered] == ["b"]


def test_faiss_contract():
    _exercise(FaissStore(dim=8))


@pytest.mark.skipif(os.environ.get("SAGE_RUN_QDRANT") != "1", reason="needs Qdrant")
def test_qdrant_contract():
    from sage.vectorstore.qdrant_store import QdrantStore

    store = QdrantStore(dim=8, url="http://localhost:6333", collection="contract_test")
    store.reset()
    _exercise(store)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vectorstore_contract.py -v`
Expected: `test_faiss_contract` PASS, `test_qdrant_contract` SKIP — but file import fails first because `qdrant_store` is imported only inside the test, so initial run FAILs only if FAISS contract regresses. Confirm both behaviors after writing the backend.

- [ ] **Step 3: Write the implementation**

`sage/vectorstore/qdrant_store.py`:
```python
from __future__ import annotations

from typing import Sequence

from sage.vectorstore.base import Filter, SearchHit


class QdrantStore:
    def __init__(self, dim: int, url: str, collection: str) -> None:
        from qdrant_client import QdrantClient

        self.dim = dim
        self.collection = collection
        self._client = QdrantClient(url=url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def reset(self) -> None:
        self._client.delete_collection(self.collection)
        self._ensure_collection()

    def upsert(self, ids, vectors, payloads) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=i, vector=list(v), payload={**p, "_ext_id": ext})
            for i, (ext, v, p) in enumerate(zip(ids, vectors, payloads))
        ]
        self._client.upsert(self.collection, points=points)

    def _build_filter(self, flt: Filter | None):
        from qdrant_client.models import FieldCondition, Filter as QFilter, MatchValue, Range

        if flt is None:
            return None
        must = []
        if flt.category is not None:
            must.append(FieldCondition(key="category", match=MatchValue(value=flt.category)))
        if flt.price_max is not None:
            must.append(FieldCondition(key="price", range=Range(lte=flt.price_max)))
        return QFilter(must=must) if must else None

    def search(self, vector, top_k, flt: Filter | None = None) -> list[SearchHit]:
        res = self._client.search(
            self.collection,
            query_vector=list(vector),
            limit=top_k,
            query_filter=self._build_filter(flt),
            with_payload=True,
        )
        out = []
        for p in res:
            payload = dict(p.payload or {})
            ext = payload.pop("_ext_id", str(p.id))
            out.append(SearchHit(id=ext, score=float(p.score), payload=payload))
        return out
```

- [ ] **Step 4: Run tests to verify**

Run: `python -m pytest tests/test_vectorstore_contract.py -v`
Expected: `test_faiss_contract` PASS, `test_qdrant_contract` SKIP.
Optional (with Docker): `docker run -p 6333:6333 qdrant/qdrant` then `SAGE_RUN_QDRANT=1 python -m pytest tests/test_vectorstore_contract.py::test_qdrant_contract -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/vectorstore/qdrant_store.py tests/test_vectorstore_contract.py
git commit -m "feat: add Qdrant backend and shared vectorstore contract test"
```

---

## Task 9: HyDE module

**Files:**
- Create: `sage/hyde.py`
- Test: `tests/test_hyde.py`

- [ ] **Step 1: Write the failing test**

`tests/test_hyde.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hyde.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`sage/hyde.py`:
```python
from __future__ import annotations

from sage.providers.base import LLMProvider

HYDE_PROMPT = """You are writing a single, concise hypothetical product review that would
perfectly answer this shopper's need. Write as if a real owner wrote it. Mention the
concrete features and use-cases implied by the request. Do not name brands.

Shopper need: {query}

Hypothetical ideal review:"""


def embedding_text(query: str, *, mode: str, generator: LLMProvider) -> str:
    """Return the text to embed for retrieval.

    naive -> the raw query; hyde -> an LLM-written hypothetical ideal review.
    """
    if mode == "naive":
        return query
    if mode == "hyde":
        return generator.generate(HYDE_PROMPT.format(query=query), temperature=0.0)
    raise ValueError(f"unknown mode: {mode}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hyde.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/hyde.py tests/test_hyde.py
git commit -m "feat: add HyDE module with naive passthrough"
```

---

## Task 10: Retrieve — group hits into candidates

**Files:**
- Create: `sage/retrieve.py`
- Test: `tests/test_retrieve.py`

- [ ] **Step 1: Write the failing test**

`tests/test_retrieve.py`:
```python
from sage.providers.fakes import FakeEmbeddingProvider
from sage.retrieve import retrieve_candidates
from sage.vectorstore.base import Filter
from sage.vectorstore.faiss_store import FaissStore


def _seed_store(emb: FakeEmbeddingProvider) -> FaissStore:
    store = FaissStore(dim=emb.dim)
    rows = [
        ("r1", "p1", "Acme", 80, "stays in at the gym, sweatproof"),
        ("r2", "p1", "Acme", 80, "great bass for running"),
        ("r3", "p2", "Bolt", 90, "comfortable for office calls"),
        ("r4", "p3", "Cory", 250, "premium audiophile sound"),
    ]
    ids, vecs, payloads = [], [], []
    for rid, pid, brand, price, text in rows:
        ids.append(rid)
        vecs.append(emb.embed([text])[0])
        payloads.append({
            "review_id": rid, "product_id": pid, "brand": brand, "category": "earbuds",
            "price": price, "avg_rating": 4.4, "review_count": 100, "text": text,
        })
    store.upsert(ids, vecs, payloads)
    return store


def test_retrieve_groups_chunks_by_product():
    emb = FakeEmbeddingProvider(dim=32)
    store = _seed_store(emb)
    cands = retrieve_candidates(
        "stays in at the gym", embedder=emb, store=store,
        flt=Filter(category="earbuds", price_max=100), top_k=10, max_candidates=5, max_snippets=4,
    )
    by_id = {c.product_id: c for c in cands}
    assert "p3" not in by_id  # filtered out by price_max=100
    assert set(by_id["p1"].chunk_ids) <= {"r1", "r2"}
    assert by_id["p1"].brand == "Acme"


def test_retrieve_respects_max_candidates():
    emb = FakeEmbeddingProvider(dim=32)
    store = _seed_store(emb)
    cands = retrieve_candidates(
        "earbuds", embedder=emb, store=store, flt=Filter(category="earbuds"),
        top_k=10, max_candidates=1, max_snippets=4,
    )
    assert len(cands) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retrieve.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`sage/retrieve.py`:
```python
from __future__ import annotations

from sage.models import Candidate, ReviewChunk
from sage.providers.base import EmbeddingProvider
from sage.vectorstore.base import Filter, VectorStore


def retrieve_candidates(
    text: str,
    *,
    embedder: EmbeddingProvider,
    store: VectorStore,
    flt: Filter | None,
    top_k: int,
    max_candidates: int,
    max_snippets: int,
) -> list[Candidate]:
    vector = embedder.embed([text])[0]
    hits = store.search(vector, top_k=top_k, flt=flt)

    grouped: dict[str, list[ReviewChunk]] = {}
    order: list[str] = []
    for hit in hits:
        p = hit.payload
        pid = p["product_id"]
        if pid not in grouped:
            grouped[pid] = []
            order.append(pid)
        if len(grouped[pid]) >= max_snippets:
            continue
        grouped[pid].append(
            ReviewChunk(
                review_id=p["review_id"],
                product_id=pid,
                brand=p["brand"],
                category=p["category"],
                price=p["price"],
                avg_rating=p["avg_rating"],
                review_count=p["review_count"],
                text=p["text"],
            )
        )

    candidates = [Candidate.from_chunks(pid, grouped[pid]) for pid in order]
    return candidates[:max_candidates]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retrieve.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/retrieve.py tests/test_retrieve.py
git commit -m "feat: add retrieve step grouping hits into candidates"
```

---

## Task 11: Bias controls

**Files:**
- Create: `sage/judge/__init__.py` (empty)
- Create: `sage/judge/bias_controls.py`
- Test: `tests/test_bias_controls.py`

- [ ] **Step 1: Write the failing test**

`tests/test_bias_controls.py`:
```python
from sage.models import Candidate
from sage.judge.bias_controls import (
    blind_brands,
    cap_evidence,
    orderings,
)


def _cand(pid, brand, snippets, count=100):
    return Candidate(product_id=pid, brand=brand, category="earbuds", price=80.0,
                     avg_rating=4.4, review_count=count, snippets=snippets, chunk_ids=["x"])


def test_blind_brands_strips_names_from_text_and_field():
    c = _cand("p1", "Acme", ["Acme buds are great", "love my acme pair"])
    blinded, mapping = blind_brands([c])
    assert blinded[0].brand == "Brand-A"
    assert "Acme" not in " ".join(blinded[0].snippets)
    assert "acme" not in " ".join(blinded[0].snippets).lower()
    assert mapping["Brand-A"] == "Acme"


def test_cap_evidence_limits_snippets_and_chars():
    c = _cand("p1", "Acme", ["x" * 500, "y" * 500, "z" * 500])
    capped = cap_evidence(c, max_snippets=2, max_chars=10)
    assert len(capped.snippets) == 2
    assert all(len(s) <= 10 for s in capped.snippets)


def test_orderings_are_deterministic_and_cover_candidates():
    cands = [_cand(f"p{i}", "B", ["s"]) for i in range(3)]
    a = orderings(cands, n=3, seed=7)
    b = orderings(cands, n=3, seed=7)
    assert [[c.product_id for c in o] for o in a] == [[c.product_id for c in o] for o in b]
    assert len(a) == 3
    for o in a:
        assert {c.product_id for c in o} == {"p0", "p1", "p2"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bias_controls.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`sage/judge/bias_controls.py`:
```python
from __future__ import annotations

import random
import re
from dataclasses import replace

from sage.models import Candidate

_LABELS = [f"Brand-{c}" for c in "ABCDEFGHIJKLMNOP"]


def blind_brands(candidates: list[Candidate]) -> tuple[list[Candidate], dict[str, str]]:
    """Replace each distinct brand with a neutral label in both the brand field and
    snippet text. Returns blinded candidates and a label->real-brand mapping for the UI."""
    mapping: dict[str, str] = {}
    label_for: dict[str, str] = {}
    out: list[Candidate] = []
    for cand in candidates:
        if cand.brand not in label_for:
            label = _LABELS[len(label_for)]
            label_for[cand.brand] = label
            mapping[label] = cand.brand
        label = label_for[cand.brand]
        pattern = re.compile(re.escape(cand.brand), re.IGNORECASE)
        snippets = [pattern.sub(label, s) for s in cand.snippets]
        out.append(replace(cand, brand=label, snippets=snippets))
    return out, mapping


def cap_evidence(cand: Candidate, *, max_snippets: int, max_chars: int) -> Candidate:
    """Verbosity control: equal evidence budget per candidate."""
    snippets = [s[:max_chars] for s in cand.snippets[:max_snippets]]
    return replace(cand, snippets=snippets)


def normalized_review_count(count: int, *, cap: int = 1000) -> float:
    """Squash raw review counts into [0,1] so volume cannot dominate quality."""
    return min(count, cap) / cap


def orderings(candidates: list[Candidate], *, n: int, seed: int) -> list[list[Candidate]]:
    """Position control: produce n deterministic shuffles of the candidate list."""
    rng = random.Random(seed)
    result: list[list[Candidate]] = []
    for _ in range(n):
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        result.append(shuffled)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bias_controls.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/judge/__init__.py sage/judge/bias_controls.py tests/test_bias_controls.py
git commit -m "feat: add bias control helpers"
```

---

## Task 12: Judge — listwise ranking under bias controls

The judge prompts the LLM to return a JSON ranking. With bias controls ON it caps evidence, blinds brands, and aggregates over `position_orderings` shuffles by mean rank. With controls OFF it does a single pass in the given order with raw evidence.

**Files:**
- Create: `sage/judge/judge.py`
- Test: `tests/test_judge.py`

- [ ] **Step 1: Write the failing test**

`tests/test_judge.py`:
```python
import json

from sage.models import Candidate
from sage.judge.judge import judge_rank, build_judge_prompt
from sage.providers.fakes import FakeLLMProvider


def _cand(pid, brand, snippets, count=100):
    return Candidate(product_id=pid, brand=brand, category="earbuds", price=80.0,
                     avg_rating=4.4, review_count=count, snippets=snippets, chunk_ids=[pid + "c"])


class RankByProductLLM:
    """Fake judge: always ranks p_good first regardless of position, returning the
    product_ids it actually sees (so we can detect brand-blinding leaks)."""

    family = "fake"

    def __init__(self):
        self.seen_prompts = []

    def generate(self, prompt, *, temperature=0.0):
        self.seen_prompts.append(prompt)
        # parse the labels the prompt exposed, then rank "good" product first
        order = []
        for line in prompt.splitlines():
            if line.startswith("PRODUCT "):
                order.append(line.split()[1].strip(":"))
        ranked = sorted(order, key=lambda pid: 0 if pid == "p_good" else 1)
        return json.dumps({"ranking": [{"product_id": p, "reason": "r"} for p in ranked]})


def test_judge_ranks_and_returns_ranked_products():
    cands = [_cand("p_bad", "Acme", ["ok"]), _cand("p_good", "Bolt", ["amazing"])]
    out = judge_rank("best earbuds", cands, llm=RankByProductLLM(), bias_on=False,
                     position_orderings=1, seed=7, max_snippets=4, max_snippet_chars=280)
    assert out[0].candidate.product_id == "p_good"
    assert out[0].rank == 1
    assert out[1].rank == 2


def test_position_aggregation_is_order_independent():
    cands = [_cand("p_bad", "Acme", ["ok"]), _cand("p_good", "Bolt", ["amazing"])]
    forward = judge_rank("q", cands, llm=RankByProductLLM(), bias_on=True,
                         position_orderings=3, seed=7, max_snippets=4, max_snippet_chars=280)
    reversed_in = judge_rank("q", list(reversed(cands)), llm=RankByProductLLM(), bias_on=True,
                             position_orderings=3, seed=7, max_snippets=4, max_snippet_chars=280)
    assert [r.candidate.product_id for r in forward] == [r.candidate.product_id for r in reversed_in]


def test_bias_on_blinds_brands_in_prompt():
    llm = RankByProductLLM()
    cands = [_cand("p_good", "Acme", ["Acme is loud"])]
    judge_rank("q", cands, llm=llm, bias_on=True, position_orderings=1, seed=7,
               max_snippets=4, max_snippet_chars=280)
    assert "Acme" not in llm.seen_prompts[0]
    assert "Brand-A" in llm.seen_prompts[0]


def test_build_prompt_lists_products_with_label_lines():
    cands = [_cand("p1", "Brand-A", ["nice"])]
    prompt = build_judge_prompt("q", cands)
    assert "PRODUCT p1:" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_judge.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`sage/judge/judge.py`:
```python
from __future__ import annotations

import json

from sage.models import Candidate, RankedProduct
from sage.providers.base import LLMProvider
from sage.judge.bias_controls import blind_brands, cap_evidence, normalized_review_count, orderings

JUDGE_INSTRUCTIONS = """You are an impartial shopping judge. Rank the products below from
best to worst for the shopper's need. Judge only on the evidence shown. Return STRICT JSON:
{{"ranking": [{{"product_id": "...", "reason": "..."}}, ...]}} with every product included once.

Shopper need: {query}

Products:
{products}

JSON:"""


def build_judge_prompt(query: str, candidates: list[Candidate]) -> str:
    blocks = []
    for c in candidates:
        vol = normalized_review_count(c.review_count)
        snippets = " | ".join(c.snippets)
        blocks.append(
            f"PRODUCT {c.product_id}: brand={c.brand} price=${c.price:.0f} "
            f"rating={c.avg_rating} review_volume={vol:.2f}\n  reviews: {snippets}"
        )
    return JUDGE_INSTRUCTIONS.format(query=query, products="\n".join(blocks))


def _rank_once(query: str, candidates: list[Candidate], llm: LLMProvider) -> list[str]:
    prompt = build_judge_prompt(query, candidates)
    raw = llm.generate(prompt, temperature=0.0)
    data = json.loads(raw)
    return [item["product_id"] for item in data["ranking"]]


def _reasons_from(query: str, candidates: list[Candidate], llm: LLMProvider) -> dict[str, str]:
    prompt = build_judge_prompt(query, candidates)
    data = json.loads(llm.generate(prompt, temperature=0.0))
    return {item["product_id"]: item.get("reason", "") for item in data["ranking"]}


def judge_rank(
    query: str,
    candidates: list[Candidate],
    *,
    llm: LLMProvider,
    bias_on: bool,
    position_orderings: int,
    seed: int,
    max_snippets: int,
    max_snippet_chars: int,
) -> list[RankedProduct]:
    if not candidates:
        return []

    # original (un-blinded) candidates, keyed for the final result
    original = {c.product_id: c for c in candidates}

    if bias_on:
        prepared = [
            cap_evidence(c, max_snippets=max_snippets, max_chars=max_snippet_chars)
            for c in candidates
        ]
        prepared, _mapping = blind_brands(prepared)
        variants = orderings(prepared, n=position_orderings, seed=seed)
    else:
        prepared = candidates
        variants = [prepared]

    # aggregate mean rank position across orderings (lower = better)
    rank_sums: dict[str, float] = {c.product_id: 0.0 for c in prepared}
    counts: dict[str, int] = {c.product_id: 0 for c in prepared}
    for variant in variants:
        order = _rank_once(query, variant, llm)
        for position, pid in enumerate(order):
            rank_sums[pid] += position
            counts[pid] += 1

    mean_rank = {pid: rank_sums[pid] / max(counts[pid], 1) for pid in rank_sums}
    final_order = sorted(mean_rank, key=lambda pid: (mean_rank[pid], pid))

    reasons = _reasons_from(query, variants[0], llm)

    ranked: list[RankedProduct] = []
    for i, pid in enumerate(final_order, start=1):
        cand = original[pid]
        ranked.append(
            RankedProduct(
                candidate=cand,
                rank=i,
                reasoning=reasons.get(pid, ""),
                cited_snippet_ids=cand.chunk_ids,
            )
        )
    return ranked
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_judge.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/judge/judge.py tests/test_judge.py
git commit -m "feat: add bias-controlled listwise judge"
```

---

## Task 13: Pipeline orchestrator

**Files:**
- Create: `sage/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
from sage.config import Settings
from sage.models import RecommendResult
from sage.pipeline import recommend, compare
from sage.providers.fakes import FakeEmbeddingProvider
from sage.vectorstore.base import Filter
from sage.vectorstore.faiss_store import FaissStore
from tests.test_judge import RankByProductLLM  # reuse the deterministic judge fake


def _build():
    emb = FakeEmbeddingProvider(dim=32)
    store = FaissStore(dim=emb.dim)
    rows = [
        ("r1", "p_good", "Bolt", 80, "amazing gym fit, sweatproof"),
        ("r2", "p_bad", "Acme", 90, "fell out during run"),
    ]
    ids, vecs, payloads = [], [], []
    for rid, pid, brand, price, text in rows:
        ids.append(rid)
        vecs.append(emb.embed([text])[0])
        payloads.append({"review_id": rid, "product_id": pid, "brand": brand,
                         "category": "earbuds", "price": price, "avg_rating": 4.4,
                         "review_count": 100, "text": text})
    store.upsert(ids, vecs, payloads)
    return emb, store


def test_recommend_returns_ranked_result():
    emb, store = _build()
    res = recommend(
        "best gym earbuds", mode="hyde", bias_on=True, flt=Filter(category="earbuds"),
        embedder=emb, store=store,
        hyde_llm=__import__("sage.providers.fakes", fromlist=["FakeLLMProvider"]).FakeLLMProvider(
            responses=["earbuds that stay in at the gym, sweatproof"]),
        judge_llm=RankByProductLLM(), settings=Settings(),
    )
    assert isinstance(res, RecommendResult)
    assert res.mode == "hyde" and res.bias_on is True
    assert res.ranked[0].candidate.product_id == "p_good"


def test_compare_returns_all_four_combinations():
    emb, store = _build()
    from sage.providers.fakes import FakeLLMProvider

    out = compare(
        "best gym earbuds", flt=Filter(category="earbuds"), embedder=emb, store=store,
        make_hyde_llm=lambda: FakeLLMProvider(responses=["stays in at gym"]),
        make_judge_llm=lambda: RankByProductLLM(), settings=Settings(),
    )
    assert set(out.keys()) == {"naive_bias_off", "naive_bias_on", "hyde_bias_off", "hyde_bias_on"}
    assert all(isinstance(v, RecommendResult) for v in out.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`sage/pipeline.py`:
```python
from __future__ import annotations

from typing import Callable

from sage.config import Settings
from sage.hyde import embedding_text
from sage.judge.judge import judge_rank
from sage.models import RecommendResult
from sage.providers.base import EmbeddingProvider, LLMProvider
from sage.retrieve import retrieve_candidates
from sage.vectorstore.base import Filter, VectorStore


def recommend(
    query: str,
    *,
    mode: str,
    bias_on: bool,
    flt: Filter | None,
    embedder: EmbeddingProvider,
    store: VectorStore,
    hyde_llm: LLMProvider,
    judge_llm: LLMProvider,
    settings: Settings,
) -> RecommendResult:
    text = embedding_text(query, mode=mode, generator=hyde_llm)
    candidates = retrieve_candidates(
        text,
        embedder=embedder,
        store=store,
        flt=flt,
        top_k=settings.top_k,
        max_candidates=settings.max_candidates,
        max_snippets=settings.max_snippets,
    )
    ranked = judge_rank(
        query,
        candidates,
        llm=judge_llm,
        bias_on=bias_on,
        position_orderings=settings.position_orderings,
        seed=settings.seed,
        max_snippets=settings.max_snippets,
        max_snippet_chars=settings.max_snippet_chars,
    )
    return RecommendResult(query=query, mode=mode, bias_on=bias_on, ranked=ranked)


def compare(
    query: str,
    *,
    flt: Filter | None,
    embedder: EmbeddingProvider,
    store: VectorStore,
    make_hyde_llm: Callable[[], LLMProvider],
    make_judge_llm: Callable[[], LLMProvider],
    settings: Settings,
) -> dict[str, RecommendResult]:
    """Run all four combinations for the UI compare view. Fresh LLM instances per run
    so scripted-fake state and call logs do not bleed across combinations."""
    out: dict[str, RecommendResult] = {}
    for mode in ("naive", "hyde"):
        for bias_on in (False, True):
            key = f"{mode}_bias_{'on' if bias_on else 'off'}"
            out[key] = recommend(
                query, mode=mode, bias_on=bias_on, flt=flt, embedder=embedder, store=store,
                hyde_llm=make_hyde_llm(), judge_llm=make_judge_llm(), settings=settings,
            )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline recommend and compare orchestrators"
```

---

## Task 14: Ingestion

Reads a JSONL subset of the Amazon Reviews headphones data (one record per review with
`review_id, product_id, brand, category, price, avg_rating, review_count, text`), chunks
long review text, embeds, and upserts. A tiny fixture drives the test; the real dataset
path is passed at runtime.

**Files:**
- Create: `sage/ingest.py`
- Create: `tests/fixtures/reviews_sample.jsonl`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

`tests/fixtures/reviews_sample.jsonl`:
```
{"review_id": "r1", "product_id": "p1", "brand": "Acme", "category": "earbuds", "price": 80.0, "avg_rating": 4.5, "review_count": 120, "text": "These stay in at the gym and survive sweat."}
{"review_id": "r2", "product_id": "p2", "brand": "Bolt", "category": "earbuds", "price": 95.0, "avg_rating": 4.2, "review_count": 60, "text": "Great for office calls, clear mic."}
```

`tests/test_ingest.py`:
```python
from pathlib import Path

from sage.ingest import load_records, ingest
from sage.providers.fakes import FakeEmbeddingProvider
from sage.vectorstore.faiss_store import FaissStore

FIXTURE = Path(__file__).parent / "fixtures" / "reviews_sample.jsonl"


def test_load_records_parses_jsonl():
    recs = load_records(FIXTURE)
    assert len(recs) == 2
    assert recs[0]["product_id"] == "p1"


def test_ingest_upserts_searchable_chunks():
    emb = FakeEmbeddingProvider(dim=32)
    store = FaissStore(dim=emb.dim)
    n = ingest(FIXTURE, embedder=emb, store=store, batch_size=10)
    assert n == 2
    hits = store.search(emb.embed(["gym sweat"])[0], top_k=2)
    assert {h.payload["product_id"] for h in hits} == {"p1", "p2"}
    assert "text" in hits[0].payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`sage/ingest.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

from sage.providers.base import EmbeddingProvider
from sage.vectorstore.base import VectorStore

_REQUIRED = ("review_id", "product_id", "brand", "category", "price", "avg_rating",
             "review_count", "text")


def load_records(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = [k for k in _REQUIRED if k not in rec]
            if missing:
                raise ValueError(f"record {rec.get('review_id')} missing fields: {missing}")
            records.append(rec)
    return records


def ingest(
    path: str | Path,
    *,
    embedder: EmbeddingProvider,
    store: VectorStore,
    batch_size: int = 128,
) -> int:
    records = load_records(path)
    total = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        vectors = embedder.embed([r["text"] for r in batch])
        ids = [r["review_id"] for r in batch]
        payloads = [{k: r[k] for k in _REQUIRED} for r in batch]
        store.upsert(ids, vectors, payloads)
        total += len(batch)
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/ingest.py tests/fixtures/reviews_sample.jsonl tests/test_ingest.py
git commit -m "feat: add JSONL ingestion"
```

---

## Task 15: Eval / ablation script

Computes a simple retrieval-overlap metric showing HyDE pulls different (and, on a curated
seed set, better) candidates than naive. The metric is deterministic with fakes so it runs
in CI; with real models it produces meaningful numbers.

**Files:**
- Create: `sage/eval/__init__.py` (empty)
- Create: `sage/eval/run_ablation.py`
- Create: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

`tests/test_eval.py`:
```python
from sage.config import Settings
from sage.eval.run_ablation import ablate, AblationRow
from sage.providers.fakes import FakeEmbeddingProvider, FakeLLMProvider
from sage.vectorstore.base import Filter
from sage.vectorstore.faiss_store import FaissStore


def _store(emb):
    store = FaissStore(dim=emb.dim)
    rows = [("r1", "p1", "Acme", 80, "stays in at the gym sweatproof"),
            ("r2", "p2", "Bolt", 90, "audiophile studio sound")]
    ids, vecs, pl = [], [], []
    for rid, pid, brand, price, text in rows:
        ids.append(rid); vecs.append(emb.embed([text])[0])
        pl.append({"review_id": rid, "product_id": pid, "brand": brand, "category": "earbuds",
                   "price": price, "avg_rating": 4.4, "review_count": 100, "text": text})
    store.upsert(ids, vecs, pl)
    return store


def test_ablate_reports_per_query_candidate_sets():
    emb = FakeEmbeddingProvider(dim=32)
    store = _store(emb)
    rows = ablate(
        queries=["earbuds for the gym"], flt=Filter(category="earbuds"), embedder=emb,
        store=store, make_hyde_llm=lambda: FakeLLMProvider(responses=["stays in at the gym"]),
        settings=Settings(),
    )
    assert len(rows) == 1
    r = rows[0]
    assert isinstance(r, AblationRow)
    assert r.query == "earbuds for the gym"
    assert isinstance(r.naive_products, list) and isinstance(r.hyde_products, list)
    assert 0.0 <= r.jaccard <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`sage/eval/run_ablation.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sage.config import Settings
from sage.hyde import embedding_text
from sage.providers.base import EmbeddingProvider, LLMProvider
from sage.retrieve import retrieve_candidates
from sage.vectorstore.base import Filter, VectorStore


@dataclass
class AblationRow:
    query: str
    naive_products: list[str]
    hyde_products: list[str]
    jaccard: float


def _product_ids(text, embedder, store, flt, settings) -> list[str]:
    cands = retrieve_candidates(
        text, embedder=embedder, store=store, flt=flt, top_k=settings.top_k,
        max_candidates=settings.max_candidates, max_snippets=settings.max_snippets,
    )
    return [c.product_id for c in cands]


def ablate(
    *,
    queries: list[str],
    flt: Filter | None,
    embedder: EmbeddingProvider,
    store: VectorStore,
    make_hyde_llm: Callable[[], LLMProvider],
    settings: Settings,
) -> list[AblationRow]:
    rows: list[AblationRow] = []
    for q in queries:
        naive = _product_ids(q, embedder, store, flt, settings)
        hyde_text = embedding_text(q, mode="hyde", generator=make_hyde_llm())
        hyde = _product_ids(hyde_text, embedder, store, flt, settings)
        ns, hs = set(naive), set(hyde)
        union = ns | hs
        jaccard = len(ns & hs) / len(union) if union else 1.0
        rows.append(AblationRow(query=q, naive_products=naive, hyde_products=hyde, jaccard=jaccard))
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/eval/__init__.py sage/eval/run_ablation.py tests/test_eval.py
git commit -m "feat: add naive-vs-HyDE ablation"
```

---

## Task 16: CLI wiring + docker-compose + README

Ties the library together behind real providers/stores using the factory, and adds the
operational glue. The CLI is thin — it builds real components from `Settings` and calls the
same pipeline the tests exercise.

**Files:**
- Create: `sage/cli.py`
- Create: `sage/runtime.py` (builds real store + providers from Settings)
- Create: `docker-compose.yml`
- Create: `README.md`
- Test: `tests/test_runtime.py`

- [ ] **Step 1: Write the failing test**

`tests/test_runtime.py`:
```python
from sage.config import Settings
from sage.runtime import build_store


def test_build_store_faiss():
    s = Settings(vector_backend="faiss", embedding_provider="fake")
    store = build_store(s, dim=16)
    assert store.dim == 16
    # smoke: upsert + search round-trips
    store.upsert(["a"], [[0.5] * 16], [{"product_id": "p", "category": "earbuds", "price": 10,
                                        "review_id": "a", "brand": "X", "avg_rating": 4,
                                        "review_count": 1, "text": "t"}])
    assert store.search([0.5] * 16, top_k=1)[0].id == "a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runtime.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`sage/runtime.py`:
```python
from __future__ import annotations

from sage.config import Settings
from sage.vectorstore.base import VectorStore


def build_store(settings: Settings, dim: int) -> VectorStore:
    if settings.vector_backend == "faiss":
        from sage.vectorstore.faiss_store import FaissStore

        return FaissStore(dim=dim)
    if settings.vector_backend == "qdrant":
        from sage.vectorstore.qdrant_store import QdrantStore

        return QdrantStore(dim=dim, url=settings.qdrant_url, collection=settings.collection)
    raise ValueError(f"unknown vector backend: {settings.vector_backend}")
```

`sage/cli.py`:
```python
from __future__ import annotations

import json

import typer

from sage.config import Settings
from sage.eval.run_ablation import ablate
from sage.ingest import ingest
from sage.pipeline import recommend
from sage.providers.factory import assert_distinct_families, build_embedder, build_llm
from sage.runtime import build_store
from sage.vectorstore.base import Filter

app = typer.Typer(help="Review-Sage: HyDE + bias-controlled judge purchase advisor")


def _embedder_and_store():
    settings = Settings()
    embedder = build_embedder(settings)
    store = build_store(settings, dim=embedder.dim)
    return settings, embedder, store


@app.command()
def seed(path: str) -> None:
    """Ingest a JSONL review file into the configured vector store."""
    settings, embedder, store = _embedder_and_store()
    n = ingest(path, embedder=embedder, store=store)
    typer.echo(f"ingested {n} reviews into {settings.vector_backend}")


@app.command()
def recommend_cmd(
    query: str,
    mode: str = "hyde",
    bias: bool = True,
    price_max: float = typer.Option(None),
    category: str = typer.Option("earbuds"),
) -> None:
    """Rank products for a query."""
    settings, embedder, store = _embedder_and_store()
    hyde_llm = build_llm("hyde", settings)
    judge_llm = build_llm("judge", settings)
    assert_distinct_families(
        hyde_family=hyde_llm.family, judge_family=judge_llm.family,
        allow_same=settings.allow_same_family_judge,
    )
    res = recommend(
        query, mode=mode, bias_on=bias, flt=Filter(category=category, price_max=price_max),
        embedder=embedder, store=store, hyde_llm=hyde_llm, judge_llm=judge_llm, settings=settings,
    )
    for rp in res.ranked:
        typer.echo(f"#{rp.rank} {rp.candidate.brand} ({rp.candidate.product_id}) — {rp.reasoning}")


@app.command()
def eval_cmd(queries_file: str, category: str = "earbuds") -> None:
    """Run the naive-vs-HyDE ablation over a newline-delimited queries file."""
    settings, embedder, store = _embedder_and_store()
    queries = [q.strip() for q in open(queries_file, encoding="utf-8") if q.strip()]
    rows = ablate(
        queries=queries, flt=Filter(category=category), embedder=embedder, store=store,
        make_hyde_llm=lambda: build_llm("hyde", settings), settings=settings,
    )
    typer.echo(json.dumps([r.__dict__ for r in rows], indent=2))


if __name__ == "__main__":
    app()
```

`docker-compose.yml`:
```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

`README.md`: document setup (`pip install -e ".[dev,gemini,anthropic]"`), required env
vars per provider, `docker compose up -d qdrant`, `python -m sage.cli seed data/reviews.jsonl`,
`python -m sage.cli recommend-cmd "best gym earbuds" --price-max 100`, and how to switch
providers/backends via `SAGE_*` env vars. Note the self-preference default (Gemini generator
+ Claude judge) and how to override with `SAGE_ALLOW_SAME_FAMILY_JUDGE=true`.

- [ ] **Step 4: Run tests and full suite to verify**

Run: `python -m pytest -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sage/cli.py sage/runtime.py docker-compose.yml README.md tests/test_runtime.py
git commit -m "feat: add CLI, runtime wiring, docker-compose, and README"
```

---

## Final verification

- [ ] Run the full suite: `python -m pytest -v` — all green.
- [ ] Run ruff: `ruff check sage tests` — clean.
- [ ] Manual smoke (fakes, no keys): set `SAGE_EMBEDDING_PROVIDER=fake SAGE_HYDE_PROVIDER=fake SAGE_JUDGE_PROVIDER=fake SAGE_VECTOR_BACKEND=faiss`, then `python -m sage.cli seed tests/fixtures/reviews_sample.jsonl` and `python -m sage.cli recommend-cmd "gym earbuds"` — observe a ranked list. (Note: in-process FAISS does not persist across CLI invocations; persistence across processes requires Qdrant. This is expected and documented.)
- [ ] Confirm self-preference guard: default Settings has `hyde_provider=gemini`, `judge_provider=anthropic` → distinct families, no error.

> **Process-boundary note:** with the FAISS backend the index lives in memory, so `seed` and `recommend-cmd` in separate CLI runs do not share data. For a real end-to-end CLI demo use the Qdrant backend (`docker compose up -d qdrant`, `SAGE_VECTOR_BACKEND=qdrant`). The API layer (Plan 2) holds one store instance for the process lifetime, so seeding via the API or a long-lived process works with either backend.
