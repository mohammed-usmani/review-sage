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
