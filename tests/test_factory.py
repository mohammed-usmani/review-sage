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
