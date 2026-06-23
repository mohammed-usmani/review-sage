# Review-Sage — Design Spec

**Date:** 2026-06-23
**Status:** Approved (ready for implementation planning)

## 1. Summary

Review-Sage is a purchase-decision advisor for headphones/earbuds. A user asks
in plain language ("best wireless earbuds for the gym under $100"); the system
retrieves real product reviews using **HyDE** (Hypothetical Document
Embeddings), groups them into candidate products, and ranks the candidates with
an **LLM-as-judge** that runs under explicit **bias controls**. Results are
returned with cited real-review snippets and the judge's reasoning.

It is built as a first-class teaching example of a HyDE-based system: the value
of HyDE (vs naive query embedding) and of bias controls (vs a naive judge) is
made **visible** via a compare-modes toggle and an offline ablation script.

### Why HyDE + LLM-as-judge belong together here

- **HyDE** earns its keep because there is a strong vocabulary mismatch between
  casual queries ("good for the gym") and dense review prose. Embedding a
  hypothetical *ideal review* retrieves better candidates than embedding the
  bare query.
- **LLM-as-judge** is first-class (not bolted on) because the system retrieves
  multiple candidates and must **rank** them. Ranking is exactly where position
  bias, verbosity bias, brand bias, and self-preference bias bite — so bias
  controls have real work to do.

## 2. Scope

### In scope (v1)
- Single product category: **headphones/earbuds**.
- Offline, reproducible corpus: **Amazon Reviews** public dataset (McAuley/UCSD)
  subset, seeded once.
- HyDE retrieval **and** naive retrieval (for comparison).
- LLM-as-judge listwise ranking of candidate products.
- Four bias controls: **position, verbosity/length, brand-blinding,
  self-preference**.
- Pluggable **providers**: Gemini, Vertex AI, OpenAI, Claude — selectable per
  role (HyDE generator / judge / embeddings).
- Pluggable **vector store**: Qdrant (default, Docker) + FAISS (in-process
  fallback / tests).
- FastAPI backend + small React/Vite/Tailwind web UI.
- Compare-modes toggle (naive vs HyDE; bias on vs off) + offline ablation
  script.

### Out of scope (documented extension points)
- Multiple product categories (architecture supports it; it is data + config,
  not new code).
- Judge panel / quorum voting (single judge in v1; extension point left in
  `bias_controls`).
- Live scraping / freshness (corpus is offline and fixed).
- Auth, user accounts, persistence of user history.
- Recency bias and sycophancy controls (optional; not built in v1).

## 3. Architecture

Layered: a pure, testable **library core** (`sage/`), a thin **API** wrapper
(`api/`), and a small **web UI** (`web/`). The pipeline is the single
orchestrator; API, CLI, and eval script are all thin callers of it, so what is
measured is what ships.

```
review-sage/
├─ sage/                     # pure, testable library core (no web)
│  ├─ config.py              # pydantic-settings: per-role models, vector backend, top-k, filters, seed
│  ├─ providers/
│  │  ├─ base.py             # LLMProvider.generate(), EmbeddingProvider.embed()
│  │  ├─ gemini.py  openai.py  anthropic.py  vertex.py
│  │  ├─ fakes.py            # FakeLLMProvider / FakeEmbeddingProvider (deterministic, for tests)
│  │  └─ factory.py          # resolves a provider PER ROLE (hyde / judge / embedding)
│  ├─ vectorstore/
│  │  ├─ base.py             # VectorStore: upsert / search(filter=...)
│  │  ├─ qdrant_store.py     # default; metadata filtering
│  │  └─ faiss_store.py      # in-process fallback / tests
│  ├─ ingest.py              # Amazon Reviews subset → chunk → embed → upsert (brand stored as metadata)
│  ├─ hyde.py                # query → hypothetical "ideal review"; also `naive` passthrough mode
│  ├─ retrieve.py            # embed → search+filter → group chunks into CANDIDATE PRODUCTS + snippets
│  ├─ judge/
│  │  ├─ judge.py            # ranks candidate products vs query (listwise)
│  │  └─ bias_controls.py    # position / verbosity / brand-blind / self-preference
│  ├─ pipeline.py            # hyde → retrieve → judge → ranked result; mode flags (naive|hyde, bias on|off)
│  └─ eval/run_ablation.py   # offline: naive vs HyDE on seed queries
├─ api/                      # FastAPI: POST /recommend, POST /compare, GET /health
└─ web/                      # React+Vite+Tailwind UI
```

### Key boundaries
- `providers/` and `vectorstore/` are both swappable behind interfaces.
- Provider resolution is **per role**: the HyDE generator, the judge, and the
  embedding model are configured independently. This is what makes
  self-preference control possible (judge family ≠ generator family).
- `pipeline.py` is the only orchestrator. `api/`, a CLI, and `eval/` are thin
  wrappers.

## 4. Data flow (one `/recommend` call)

1. **Query in:** e.g. `"best wireless earbuds for the gym under $100"` +
   filters `{price_max: 100, category: earbuds}`.
2. **HyDE** (`hyde.py`): the generator LLM writes a hypothetical *ideal review*.
   In `naive` mode this step is skipped and the raw query string is used.
3. **Embed** the hypothetical doc (or raw query) — Gemini embeddings by default.
4. **Retrieve** (`retrieve.py`): vector search top-K review chunks with a Qdrant
   metadata filter (`price ≤ 100 AND category = earbuds`). Group chunks by
   `product_id` → **candidate products**, each carrying best snippets +
   aggregate rating / review-count.
5. **Judge** (`judge/`): listwise rank candidates against the query under the
   active bias controls; return ordered products + per-product reasoning + cited
   snippet ids.
6. **Out:** ranked recommendation rendered as cards with the real cited review
   snippets and an expandable judge rationale.

### Vector store metadata schema (per chunk)
`product_id`, `brand`, `category`, `price`, `avg_rating`, `review_count`,
`review_id`, `chunk_text`.

## 5. Bias controls (each toggleable, in `bias_controls.py`)

- **Position bias** → the judge runs over **N shuffled orderings** of the
  candidate list (N and shuffle seed from config); scores aggregated by
  mean-rank / Borda. A candidate that only wins in one ordering loses. This is
  the visible demonstration of order-independence.
- **Verbosity / length bias** → candidate evidence is **truncated to an equal
  budget** (top-M snippets, capped chars) before judging, and `review_count` is
  passed as a **normalized** signal rather than a raw count — so a product with
  4,000 reviews cannot bulldoze a great product with 40.
- **Brand-blinding** → brand names are **stripped from snippet text and hidden
  from the judge** (replaced with `Brand-A/B/C`); the real brand is revealed
  only in the final UI render, after ranking. Brand remains available as filter
  metadata but never reaches the judge prompt.
- **Self-preference** → the factory **asserts judge family ≠ HyDE-generator
  family** (Claude judge vs Gemini generator by default); if config violates it,
  the run **warns** at runtime. A panel/quorum judge is left as a documented
  extension point.

The pipeline accepts `bias_controls: on|off` so `/compare` can show
verdict-with vs verdict-without side by side.

## 6. Providers & default models (all overridable via config)

| Role            | Default model              | Rationale                                   |
|-----------------|----------------------------|---------------------------------------------|
| Embeddings      | Gemini `gemini-embedding-001` | Stated default; swappable to any provider |
| HyDE generator  | Gemini Flash               | Cheap, fast hypothetical-doc generation     |
| Judge           | Claude (Haiku/Sonnet)      | Different family → self-preference control by default |

Provider SDKs (`google-genai`, `openai`, `anthropic`) are imported **lazily**,
so only the keys for providers actually used are required. Vertex AI is a
distinct adapter from Gemini (AI Studio) for the Google path.

## 7. API surface

- `POST /recommend` — body: `{query, filters{price_max, category}, mode:
  naive|hyde, bias_controls: on|off}` → ranked products + cited snippets + judge
  reasoning.
- `POST /compare` — runs both relevant modes; returns side-by-side result for
  the UI compare toggle.
- `GET /health` — liveness.

Ingestion is a CLI / `make seed` command, not an HTTP endpoint.

## 8. Web UI (React + Vite + Tailwind, minimal)

- Search box + filter controls (price max, sub-category).
- Ranked product cards, each showing cited real-review snippets.
- Expandable judge reasoning per card.
- **Compare-modes toggle**: naive vs HyDE retrieval, and bias-on vs bias-off
  judging, side by side.

## 9. Testing & determinism

- **Injectable fake providers.** `FakeLLMProvider` / `FakeEmbeddingProvider`
  implement the same interfaces and return scripted, deterministic outputs. The
  full pipeline, judge, and every bias control are tested with **zero API
  calls** — fast, free, deterministic.
- **Per-bias-control tests:**
  - position → verdict identical across input orderings;
  - verbosity → a padded candidate does not outrank a concise better one;
  - brand-blind → the judge prompt contains no real brand string;
  - self-preference → factory raises/warns when judge family == generator family.
- **VectorStore** tested against FAISS in-process (no Docker in CI); a shared
  contract test runs against both FAISS and Qdrant so backends stay
  interchangeable.
- **Determinism:** fakes make logic deterministic; real-model paths use
  `temperature=0` and a config-seeded shuffle so position-bias aggregation is
  reproducible.
- **Eval script** runs on the seed-query set with fakes in CI (smoke) and with
  real models when run manually.

## 10. Tech stack

- Backend: Python 3.11+, FastAPI, pydantic-settings, httpx; provider SDKs
  loaded lazily.
- Vector: Qdrant (docker-compose) + faiss-cpu.
- Data: Amazon Reviews headphones subset, ingested once via `ingest.py`.
- Web: React + Vite + Tailwind.
- Tooling: ruff, pytest, docker-compose, Makefile (`seed`, `api`, `web`,
  `test`, `eval`).

## 11. Build order (high level — detailed in the implementation plan)

1. Provider interfaces + fakes + factory (per-role resolution).
2. VectorStore interface + FAISS backend + contract test; Qdrant backend.
3. Ingestion (corpus subset → chunk → embed → upsert).
4. HyDE module (hyde + naive modes).
5. Retrieve (search + group into candidates).
6. Judge + bias controls (TDD per control).
7. Pipeline orchestrator + mode flags.
8. FastAPI layer.
9. Web UI.
10. Eval/ablation script.
