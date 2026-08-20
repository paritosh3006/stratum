"""Tests for the language-expectation model: `EvalItem` fields, the
`LanguageDetector` backends, the `check_output_language` evaluator, and
stage-aware attribution through `Harness`.

The motivating bug: a user writes in Hindi, the product answers in Bodo
(brx). Both are Devanagari, so a script-range check alone cannot catch it
— that's why `ScriptRangeDetector` must never assert `language`, and why
a definitive `fail` for that case needs a real language-identifying
backend (`IndicLIDLanguageDetector`, or the deterministic `StubLanguageDetector`
standing in for one here).
"""

from __future__ import annotations

import pytest

from stratum.dataset import Dataset, EvalItem
from stratum.endpoint import CallableEndpoint, RagResponse
from stratum.harness import Harness
from stratum.language import LanguageGuess, ScriptRangeDetector, StubLanguageDetector
from stratum.language.backends import IndicLIDLanguageDetector
from stratum.metrics.language import check_output_language


def _item(**kwargs):
    defaults = dict(id="i1", language="hi-Deva", slice="parallel_core", query="q")
    defaults.update(kwargs)
    return EvalItem(**defaults)


# ---------------------------------------------------------------------------
# EvalItem field defaulting — backward compatibility
# ---------------------------------------------------------------------------

def test_expected_defaults_to_query_language_when_absent():
    item = _item(query_language="hi-Deva")
    assert item.expected_answer_language is None
    assert item.effective_expected_answer_language == "hi-Deva"


def test_effective_expected_is_none_when_neither_field_set():
    item = _item()
    assert item.query_language is None
    assert item.expected_answer_language is None
    assert item.effective_expected_answer_language is None


def test_explicit_expected_answer_language_wins_over_query_language():
    item = _item(query_language="hi-Deva", expected_answer_language="en-Latn")
    assert item.effective_expected_answer_language == "en-Latn"


def test_dataset_predating_the_fields_parses_unchanged():
    ds = Dataset(items=[_item()])
    assert ds.items[0].effective_expected_answer_language is None


# ---------------------------------------------------------------------------
# ScriptRangeDetector — must identify script, never claim a language
# ---------------------------------------------------------------------------

def test_script_detector_never_sets_language():
    d = ScriptRangeDetector()
    for text in ("यह हिंदी में है", "This is English", "এটি বাংলা লেখা"):
        assert d.detect(text).language is None


def test_script_detector_identifies_devanagari():
    guess = ScriptRangeDetector().detect("यह एक वाक्य है")
    assert guess.script == "Deva"


def test_script_detector_identifies_latin():
    guess = ScriptRangeDetector().detect("this is english text")
    assert guess.script == "Latn"


def test_script_detector_cannot_tell_hindi_from_bodo():
    # Both Devanagari — the whole point of the production bug this feature
    # was built for. The script detector must report the same script for
    # both rather than pretending it can tell them apart.
    hi = ScriptRangeDetector().detect("प्रीमियम कितना है")
    brx = ScriptRangeDetector().detect("बर'था थार दं")
    assert hi.script == brx.script == "Deva"
    assert hi.language is None and brx.language is None


# ---------------------------------------------------------------------------
# StubLanguageDetector — deterministic, offline, test-controllable
# ---------------------------------------------------------------------------

def test_stub_detector_is_deterministic():
    d = StubLanguageDetector(default="en")
    assert d.detect("some answer text") == d.detect("some answer text")


def test_stub_detector_overrides_specific_text():
    d = StubLanguageDetector(overrides={"बर'था थार दं": "brx"}, default="en")
    assert d.detect("बर'था थार दं").language == "brx"
    assert d.detect("anything else").language == "en"


# ---------------------------------------------------------------------------
# IndicLIDLanguageDetector — mocked model output, no network
# ---------------------------------------------------------------------------

def test_indiclid_adapts_a_mocked_predict_fn():
    def fake_predict(text: str) -> tuple[str, float]:
        return ("brx", 0.91)

    d = IndicLIDLanguageDetector(predict_fn=fake_predict)
    guess = d.detect("कोई भी टेक्स्ट")
    assert guess.language == "brx"
    assert guess.confidence == 0.91
    assert guess.detector_id == "indiclid"


def test_indiclid_refuses_construction_without_predict_fn():
    with pytest.raises(RuntimeError):
        IndicLIDLanguageDetector()


# ---------------------------------------------------------------------------
# check_output_language — the reusable evaluator
# ---------------------------------------------------------------------------

def test_same_language_passes():
    result = check_output_language(LanguageGuess(language="hi"), "hi-Deva")
    assert result.outcome == "pass"


def test_hindi_expected_bodo_detected_fails():
    result = check_output_language(LanguageGuess(language="brx"), "hi-Deva")
    assert result.outcome == "fail"
    assert result.detected == "brx"
    assert result.expected == "hi-Deva"


def test_explicit_cross_lingual_pass_when_english_detected():
    result = check_output_language(LanguageGuess(language="en"), "en-Latn")
    assert result.outcome == "pass"


def test_script_only_mismatch_is_a_confident_fail():
    result = check_output_language(LanguageGuess(language=None, script="Latn"), "hi-Deva")
    assert result.outcome == "fail"


def test_script_only_match_is_inconclusive_not_pass():
    result = check_output_language(LanguageGuess(language=None, script="Deva"), "hi-Deva")
    assert result.outcome == "inconclusive"
    assert not result.is_measured


def test_no_expectation_is_not_applicable():
    result = check_output_language(LanguageGuess(language="en"), None)
    assert result.outcome == "not_applicable"
    assert not result.is_measured


# ---------------------------------------------------------------------------
# Harness wiring: report carries wrong_output_language with both values,
# and the attributed stage comes from configuration, not a hardcoded branch.
# ---------------------------------------------------------------------------

def _cross_lingual_dataset(expected: str) -> Dataset:
    return Dataset(items=[EvalItem(
        id="x1", language="hi-Deva", slice="parallel_core",
        query="प्रीमियम कितना है", query_language="hi-Deva",
        expected_answer_language=expected,
    )])


def _endpoint_answering_in(fixed_answer: str) -> CallableEndpoint:
    def rag(query, language, *, context_chunk_ids=None, answer_override=None):
        return RagResponse(answer=fixed_answer)
    return CallableEndpoint(rag)


def test_report_records_wrong_output_language_with_both_values():
    ds = _cross_lingual_dataset(expected="hi-Deva")
    h = Harness(
        _endpoint_answering_in("बर'था थार दं"), ds,
        language_detector=StubLanguageDetector(default="brx"),
    )
    r = h.run()
    assert r.taxonomy.get("wrong_output_language", 0) == 1
    failure = next(f for f in r.failures if f.cls == "wrong_output_language")
    assert "brx" in failure.detail
    assert "hi-Deva" in failure.detail


def test_explicit_cross_lingual_expectation_passes_end_to_end():
    ds = _cross_lingual_dataset(expected="en-Latn")
    h = Harness(
        _endpoint_answering_in("the premium is due"), ds,
        language_detector=StubLanguageDetector(default="en"),
    )
    r = h.run()
    assert r.taxonomy.get("wrong_output_language", 0) == 0


def test_backward_compatible_without_a_configured_detector():
    # No language_detector passed: Harness defaults to ScriptRangeDetector,
    # which never asserts `pass` — same-script answers stay inconclusive,
    # never a manufactured failure, for a feature the caller never opted
    # into beyond declaring expected_answer_language.
    ds = _cross_lingual_dataset(expected="hi-Deva")
    h = Harness(_endpoint_answering_in("प्रीमियम राशि है"), ds)
    r = h.run()
    assert r.taxonomy.get("wrong_output_language", 0) == 0


def test_stage_attribution_is_configured_not_hardcoded():
    """Same failure, two Harness configurations, two different stages —
    the stage comes from Harness configuration, never a branch in core
    keyed on product or use-case name."""
    ds = _cross_lingual_dataset(expected="hi-Deva")

    h_default = Harness(
        _endpoint_answering_in("बर'था थार दं"), ds,
        language_detector=StubLanguageDetector(default="brx"),
    )
    h_voice = Harness(
        _endpoint_answering_in("बर'था थार दं"), ds,
        language_detector=StubLanguageDetector(default="brx"),
        output_language_stage="s0",
    )

    fail_default = next(f for f in h_default.run().failures if f.cls == "wrong_output_language")
    fail_voice = next(f for f in h_voice.run().failures if f.cls == "wrong_output_language")

    assert fail_default.stage == "s4"
    assert fail_voice.stage == "s0"
