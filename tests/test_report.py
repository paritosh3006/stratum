"""Tests for Report.evaluate_gates.

A gate whose metric was never observed — glossary_adherence with no
glossary loaded is the motivating case — must not silently report "pass".
`fail_on_unevaluated` (default True) is what makes that a failure instead.
"""

import pytest

from stratum.report import Gate, LanguageResult, Report


def _report(*metrics: dict) -> Report:
    """One report, one language per metrics dict passed."""
    languages = [
        LanguageResult(language=f"lang{i}", n_items=10, metrics=m)
        for i, m in enumerate(metrics)
    ]
    return Report(
        system_label="test", baseline_language="lang0", n_items=10,
        languages=languages,
    )


class TestVacuousPass:
    def test_metric_missing_entirely_fails_by_default(self):
        # glossary_adherence was never computed at all — no key, not even
        # a zero-n Estimate — the case from a run with no --glossary.
        r = _report({"placeholder_integrity": {"value": 100.0, "n": 10}})
        r.evaluate_gates([Gate(metric="glossary_adherence", min_absolute=85.0)])
        gate = r.gates[0]
        assert gate.passed is False
        assert gate.observed is None
        assert "no observations" in gate.skipped_reason

    def test_metric_present_with_zero_n_fails_by_default(self):
        # The actual shape ref-004 produced: an Estimate with n=0 and
        # value=None, present in the metrics dict rather than absent from
        # it. metric_value() must treat this the same as "missing".
        r = _report({
            "glossary_adherence": {"value": None, "n": 0, "ci95": None, "precise": False},
        })
        r.evaluate_gates([Gate(metric="glossary_adherence", min_absolute=85.0)])
        gate = r.gates[0]
        assert gate.passed is False
        assert gate.observed is None

    def test_overall_status_is_failed(self):
        r = _report({"placeholder_integrity": {"value": 100.0, "n": 10}})
        r.evaluate_gates([Gate(metric="glossary_adherence", min_absolute=85.0)])
        assert r.status == "failed"

    def test_no_language_matches_gate_scope_also_fails(self):
        r = _report({"placeholder_integrity": {"value": 100.0, "n": 10}})
        r.evaluate_gates([
            Gate(metric="placeholder_integrity", languages=["does-not-exist"],
                 min_absolute=100.0),
        ])
        assert r.gates[0].passed is False

    def test_terminal_render_does_not_crash_on_unevaluated_gate(self):
        # observed is None on a failed gate — render_terminal used to format
        # it with `:.1f`, which raises on None.
        r = _report({"placeholder_integrity": {"value": 100.0, "n": 10}})
        r.evaluate_gates([Gate(metric="glossary_adherence", min_absolute=85.0)])
        out = r.render_terminal()
        assert "SKIPPED" in out
        assert "glossary_adherence" in out

    def test_fail_on_unevaluated_false_lets_it_pass(self):
        r = _report({"placeholder_integrity": {"value": 100.0, "n": 10}})
        r.evaluate_gates(
            [Gate(metric="glossary_adherence", min_absolute=85.0)],
            fail_on_unevaluated=False,
        )
        gate = r.gates[0]
        assert gate.passed is True
        assert "no observations" in gate.skipped_reason
        assert r.status == "passed"


class TestEvaluatedGatesUnaffected:
    """The fix must not change behaviour for gates that do have data."""

    def test_passing_gate_still_passes(self):
        r = _report({"placeholder_integrity": {"value": 100.0, "n": 10}})
        r.evaluate_gates([Gate(metric="placeholder_integrity", min_absolute=100.0)])
        gate = r.gates[0]
        assert gate.passed is True
        assert gate.skipped_reason is None

    def test_failing_gate_still_fails_with_observed_value(self):
        r = _report({"numeral_integrity": {"value": 40.0, "n": 10}})
        r.evaluate_gates([Gate(metric="numeral_integrity", min_absolute=95.0)])
        gate = r.gates[0]
        assert gate.passed is False
        assert gate.observed == pytest.approx(40.0)
        assert gate.failing_language == "lang0"

    def test_worst_language_is_reported_when_several_fail(self):
        r = _report(
            {"numeral_integrity": {"value": 40.0, "n": 10}},
            {"numeral_integrity": {"value": 20.0, "n": 10}},
        )
        r.evaluate_gates([Gate(metric="numeral_integrity", min_absolute=95.0)])
        gate = r.gates[0]
        assert gate.observed == pytest.approx(20.0)
        assert gate.failing_language == "lang1"

    def test_experimental_language_excluded_note_unaffected(self):
        languages = [
            LanguageResult(
                language="lang0", n_items=10, verified=True,
                metrics={"placeholder_integrity": {"value": 100.0, "n": 10}},
            ),
            LanguageResult(
                language="lang1", n_items=10, verified=False,
                metrics={"placeholder_integrity": {"value": 10.0, "n": 10}},
            ),
        ]
        r = Report(system_label="t", baseline_language="lang0", n_items=10,
                    languages=languages)
        r.evaluate_gates([Gate(metric="placeholder_integrity", min_absolute=100.0)])
        gate = r.gates[0]
        assert gate.passed is True
        assert gate.skipped_reason == "experimental languages excluded"
