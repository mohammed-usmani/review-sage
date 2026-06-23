import pytest

from sage.config import Settings
from sage.providers.factory import (
    build_llm,
    build_role_llms,
    SelfPreferenceError,
    assert_distinct_families,
)


def test_assert_distinct_families_raises_on_same():
    with pytest.raises(SelfPreferenceError):
        assert_distinct_families(hyde_family="gemini", judge_family="gemini", allow_same=False)


def test_assert_distinct_families_allows_when_opted_in():
    # should not raise, but must warn
    with pytest.warns(UserWarning):
        assert_distinct_families(hyde_family="gemini", judge_family="gemini", allow_same=True)


def test_assert_distinct_families_silent_when_different():
    # different families: no raise, no warning
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert_distinct_families(hyde_family="gemini", judge_family="anthropic", allow_same=False)


def test_build_llm_uses_fake_when_requested():
    s = Settings(hyde_provider="fake")
    llm = build_llm("hyde", s)
    assert llm.family == "fake"


def test_build_role_llms_enforces_self_preference():
    # both fake -> same family -> guard fires (library-level, not just CLI)
    s = Settings(hyde_provider="fake", judge_provider="fake")
    with pytest.raises(SelfPreferenceError):
        build_role_llms(s)


def test_build_role_llms_returns_pair_when_same_family_allowed():
    s = Settings(hyde_provider="fake", judge_provider="fake", allow_same_family_judge=True)
    with pytest.warns(UserWarning):
        hyde_llm, judge_llm = build_role_llms(s)
    assert hyde_llm.family == "fake" and judge_llm.family == "fake"


def test_build_llm_fake_judge_emits_valid_ranking_json():
    import json

    s = Settings(judge_provider="fake")
    judge = build_llm("judge", s)
    out = judge.generate("PRODUCT p1: brand=X\nPRODUCT p2: brand=Y")
    data = json.loads(out)
    assert [item["product_id"] for item in data["ranking"]] == ["p1", "p2"]
