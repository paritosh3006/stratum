"""Tests for judge backends and the calibration gate on CalibrationRegistry."""

import pytest

from stratum.judges import Calibration, CalibrationRegistry, get_judge
from stratum.judges.backends import OllamaJudge, StubJudge


class TestStubJudge:
    def test_faithfulness_full_coverage(self):
        j = StubJudge()
        r = j.judge_faithfulness(
            "The grace period is 30 days.",
            ["The grace period is 30 days, allowed for premium payment."],
            "en",
        )
        assert r.score == 1.0

    def test_faithfulness_partial_coverage(self):
        j = StubJudge()
        r = j.judge_faithfulness(
            "The grace period is 30 days and it involves dragons.",
            ["A grace period of 30 days is allowed."],
            "en",
        )
        assert 0.0 < r.score < 1.0

    def test_faithfulness_no_context_is_zero_not_a_crash(self):
        r = StubJudge().judge_faithfulness("some answer", [], "en")
        assert r.score == 0.0

    def test_correctness_matches_reference_scores_high(self):
        j = StubJudge()
        r = j.judge_correctness(
            "The grace period is 30 days.",
            "The grace period for renewal premium payment is 30 days.",
            "en",
        )
        assert r.score >= 2.0

    def test_correctness_unrelated_answer_scores_zero(self):
        j = StubJudge()
        r = j.judge_correctness("completely unrelated text", "the grace period is 30 days", "en")
        assert r.score == 0.0

    def test_correctness_score_is_on_the_0_3_rubric(self):
        j = StubJudge()
        r = j.judge_correctness("the grace period is 30 days", "the grace period is 30 days", "en")
        assert r.score in (0.0, 1.0, 2.0, 3.0)

    def test_deterministic(self):
        j = StubJudge()
        a = j.judge_faithfulness("answer text", ["context text"], "en")
        b = j.judge_faithfulness("answer text", ["context text"], "en")
        assert a.score == b.score

    def test_judge_id_is_stable(self):
        assert StubJudge().judge_id == "stub-token-overlap"


class TestOllamaJudgeParsing:
    def test_clean_json(self):
        score, reasoning = OllamaJudge._parse('{"score": 0.8, "reasoning": "mostly supported"}')
        assert score == 0.8 and reasoning == "mostly supported"

    def test_json_wrapped_in_prose(self):
        # format=json constrains many models, but not all of them, and some
        # wrap the object in a sentence anyway.
        raw = 'Here you go:\n{"score": 2, "reasoning": "close enough"}\nHope that helps!'
        score, reasoning = OllamaJudge._parse(raw)
        assert score == 2.0 and reasoning == "close enough"

    def test_missing_score_key_raises(self):
        with pytest.raises(ValueError, match="score"):
            OllamaJudge._parse('{"answer": 2}')

    def test_unparseable_response_raises(self):
        with pytest.raises(ValueError, match="parseable"):
            OllamaJudge._parse("the score is pretty good honestly")

    def test_unreachable_host_raises_clear_error(self):
        judge = OllamaJudge(host="http://localhost:1", timeout=1.0)
        with pytest.raises(RuntimeError, match="not reachable"):
            judge._generate("test prompt")

    def test_judge_id_includes_model(self):
        assert OllamaJudge(model="qwen2.5:7b").judge_id == "ollama:qwen2.5:7b"


class TestGetJudge:
    def test_default_is_stub(self):
        assert isinstance(get_judge(), StubJudge)
        assert isinstance(get_judge("stub"), StubJudge)

    def test_ollama_variants(self):
        assert isinstance(get_judge("ollama"), OllamaJudge)
        j = get_judge("ollama:llama3.1:8b")
        assert isinstance(j, OllamaJudge) and j.model == "llama3.1:8b"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_judge("not-a-real-judge")


class TestCalibrationRegistryJudgeIdGate:
    """A kappa measured against one model must not apply to a different one."""

    def _registry(self, judge_id="stub-token-overlap", kappa=0.8) -> CalibrationRegistry:
        reg = CalibrationRegistry()
        reg.register(Calibration(
            language="hi-Deva", metric="faithfulness", kappa=kappa,
            n_labelled=20, judge_id=judge_id,
        ))
        return reg

    def test_permits_when_judge_id_matches(self):
        reg = self._registry()
        assert reg.permits("hi-Deva", "faithfulness", "stub-token-overlap")

    def test_denies_when_judge_id_differs(self):
        reg = self._registry(judge_id="ollama:qwen2.5:7b")
        assert not reg.permits("hi-Deva", "faithfulness", "stub-token-overlap")

    def test_no_judge_id_given_skips_the_check(self):
        # Backward-compatible: callers that don't know/care about judge
        # identity still get the kappa-threshold behaviour alone.
        reg = self._registry(judge_id="ollama:qwen2.5:7b")
        assert reg.permits("hi-Deva", "faithfulness")

    def test_untrustworthy_kappa_denied_regardless_of_judge_id(self):
        reg = self._registry(kappa=0.2)
        assert not reg.permits("hi-Deva", "faithfulness", "stub-token-overlap")

    def test_uncalibrated_language_denied(self):
        reg = self._registry()
        assert not reg.permits("ta-Taml", "faithfulness", "stub-token-overlap")
