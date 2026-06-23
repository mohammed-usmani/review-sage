# Review-Sage

A headphones purchase advisor that retrieves real reviews via **HyDE** (Hypothetical Document Embeddings) and ranks candidate products with a **bias-controlled LLM-as-judge**.

Rather than embedding the raw user query, Review-Sage first prompts an LLM to generate a *hypothetical* review that a perfect product would have, then embeds that synthetic review for retrieval. This closes the query-document vocabulary gap. A separate judge LLM then scores and ranks the retrieved candidates while controlling for positional bias (candidates are presented in multiple shuffled orderings). A self-preference guard ensures the judge and the HyDE generator come from different model families by default.

---

## Install

```bash
pip install -e ".[dev]"
```

Add the extras for whichever providers you want:

```bash
pip install -e ".[gemini]"      # Google Gemini embeddings + LLM
pip install -e ".[openai]"      # OpenAI embeddings + LLM
pip install -e ".[anthropic]"   # Anthropic LLM (for judge)
```

You can combine them: `pip install -e ".[dev,gemini,anthropic]"`

---

## Provider configuration

All settings are controlled via `SAGE_*` environment variables (or a `.env` file in the project root). Key variables and their defaults:

| Variable | Default | Description |
|---|---|---|
| `SAGE_EMBEDDING_PROVIDER` | `gemini` | Embedding provider: `gemini`, `openai`, or `fake` |
| `SAGE_HYDE_PROVIDER` | `gemini` | HyDE generator provider: `gemini`, `openai`, `anthropic`, or `fake` |
| `SAGE_JUDGE_PROVIDER` | `anthropic` | Judge provider: `gemini`, `openai`, `anthropic`, or `fake` |
| `SAGE_VECTOR_BACKEND` | `qdrant` | Vector store: `qdrant` or `faiss` |
| `SAGE_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `SAGE_COLLECTION` | `reviews` | Qdrant collection name |
| `SAGE_TOP_K` | `40` | Candidate pool size for retrieval |
| `SAGE_MAX_CANDIDATES` | `6` | Maximum distinct products to rank |
| `SAGE_ALLOW_SAME_FAMILY_JUDGE` | `false` | Allow the judge to share the generator's model family (bypasses the self-preference guard; emits a warning). The guard is **active** by default. |

### Self-preference guard

By default, Review-Sage requires the judge LLM to be from a **different model family** than the HyDE generator. This prevents a model from unfairly favouring responses it would itself generate. If you want to use the same provider for both roles, set:

```bash
SAGE_ALLOW_SAME_FAMILY_JUDGE=true
```

### API keys

Set the standard SDK credentials for your chosen provider:

- **Google / Gemini**: set the credentials expected by the `google-genai` SDK (e.g. `GOOGLE_API_KEY` or application-default credentials).
- **OpenAI**: set `OPENAI_API_KEY`.
- **Anthropic**: set `ANTHROPIC_API_KEY`.

---

## Vector store

### Qdrant (default — recommended for multi-command workflows)

```bash
docker compose up -d qdrant
```

Qdrant persists data to `./qdrant_storage` on the host. Use this when running `seed` and `recommend-cmd` as separate CLI invocations.

### FAISS (in-process, no Docker required)

```bash
export SAGE_VECTOR_BACKEND=faiss
```

FAISS is an in-memory store. It does **not** persist data across separate CLI processes. If you seed with FAISS and then run `recommend-cmd` in a new shell, the store will be empty. FAISS is best suited for single-process usage (e.g. scripts) or smoke tests.

---

## Usage

### Seed the vector store

```bash
python -m sage.cli seed data/reviews.jsonl
```

Ingests every review in the JSONL file into the configured vector store. Each line must be a JSON object with fields: `review_id`, `product_id`, `brand`, `category`, `price`, `avg_rating`, `review_count`, `text`.

### Recommend products

```bash
python -m sage.cli recommend-cmd "best gym earbuds" --price-max 100
```

Options:
- `--mode` — `hyde` (default) or `naive` (embed the raw query directly)
- `--bias/--no-bias` — enable/disable positional-bias controls (default: enabled)
- `--price-max FLOAT` — filter to products at or below this price
- `--category TEXT` — filter by product category (default: `earbuds`)

### Run the ablation eval

```bash
python -m sage.cli eval-cmd queries.txt
```

Compares naive vs. HyDE retrieval for each query in `queries.txt` (one query per line). Prints a JSON array with `query`, `naive_products`, `hyde_products`, and `jaccard` similarity per query.

---

## Testing

```bash
python -m pytest
```

The Qdrant contract test is skipped unless you set `SAGE_RUN_QDRANT=1` with a running Qdrant instance:

```bash
SAGE_RUN_QDRANT=1 python -m pytest
```

---

## Running with zero API keys (fakes)

You can run the whole pipeline locally without any API keys using fake providers
and FAISS. The fake judge emits valid ranking JSON, so retrieval **and** judging
both run. Because FAISS is in-memory, `seed` and `recommend` must happen in the
**same process** — so drive it from a short Python script rather than two
separate CLI calls:

```bash
SAGE_EMBEDDING_PROVIDER=fake \
SAGE_HYDE_PROVIDER=fake \
SAGE_JUDGE_PROVIDER=fake \
SAGE_VECTOR_BACKEND=faiss \
python - <<'PY'
from sage.config import Settings
from sage.providers.factory import build_embedder, build_llm
from sage.runtime import build_store
from sage.ingest import ingest
from sage.pipeline import recommend
from sage.vectorstore.base import Filter

s = Settings()
emb = build_embedder(s)
store = build_store(s, dim=emb.dim)
ingest("tests/fixtures/reviews_sample.jsonl", embedder=emb, store=store)
res = recommend("best gym earbuds", mode="hyde", bias_on=True,
                flt=Filter(category="earbuds"), embedder=emb, store=store,
                hyde_llm=build_llm("hyde", s), judge_llm=build_llm("judge", s), settings=s)
print([(r.rank, r.candidate.product_id, r.candidate.brand) for r in res.ranked])
PY
```

To use the **CLI** `recommend-cmd` with all-fake providers, also set
`SAGE_ALLOW_SAME_FAMILY_JUDGE=true` and use the persistent Qdrant backend (so
`seed` and `recommend-cmd` share data across processes): all fakes share the
`fake` family, which would otherwise trip the self-preference guard that
`recommend-cmd` enforces.
