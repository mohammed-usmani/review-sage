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
