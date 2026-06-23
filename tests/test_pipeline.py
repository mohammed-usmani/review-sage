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
    from sage.providers.fakes import FakeLLMProvider
    res = recommend(
        "best gym earbuds", mode="hyde", bias_on=True, flt=Filter(category="earbuds"),
        embedder=emb, store=store,
        hyde_llm=FakeLLMProvider(responses=["earbuds that stay in at the gym, sweatproof"]),
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


def test_compare_uses_fresh_llm_per_combination():
    # Each combo must get its own LLM instance (scripted fakes would otherwise
    # exhaust). Guards against a future refactor hoisting the factory call.
    emb, store = _build()
    from sage.providers.fakes import FakeLLMProvider

    judge_instances = []

    def make_judge():
        j = RankByProductLLM()
        judge_instances.append(j)
        return j

    compare(
        "best gym earbuds", flt=Filter(category="earbuds"), embedder=emb, store=store,
        make_hyde_llm=lambda: FakeLLMProvider(responses=["stays in at gym"]),
        make_judge_llm=make_judge, settings=Settings(),
    )
    assert len(judge_instances) == 4  # one per combination
    assert len({id(j) for j in judge_instances}) == 4  # all distinct objects
