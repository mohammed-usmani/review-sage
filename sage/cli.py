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
