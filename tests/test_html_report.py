"""Tests for html_report.py.

Everything here builds a Report the way `Report.model_validate_json` would
hand one back — plain dicts for `cascades`, never touching
`cascade_objects` — because that is the only shape `stratum html
<report.json>` ever has to work with, and a test built against the live
`Cascade` dataclass could pass while the regenerate-from-JSON path is
broken.
"""

import json

import pytest

from stratum import html_report as hr
from stratum.report import LanguageResult, Report


def _stage(stage, label, points, *, isolated=True, noise=False, note=""):
    return {
        "stage": stage, "label": label, "points_lost": points,
        "ci95": [points - 1, points + 1] if points is not None else None,
        "isolated": isolated, "indistinguishable_from_zero": noise, "note": note,
    }


def _cascade(language, baseline_score, final_score, by_stage, baseline_language="en"):
    return {
        "language": language, "baseline_language": baseline_language,
        "method": "nested_oracle_ladder", "baseline_score": baseline_score,
        "final_score": final_score, "total_loss": round(baseline_score - final_score, 2),
        "by_stage": by_stage,
        "sums_correctly": True,
        "dominant_stage": max(
            (s for s in by_stage if s["points_lost"] and s["points_lost"] > 0 and not s["indistinguishable_from_zero"]),
            key=lambda s: s["points_lost"], default={"stage": None},
        )["stage"],
        "warnings": [],
    }


def _report(**kw) -> Report:
    defaults = dict(
        system_label="test-run", baseline_language="en", n_items=10,
        passes_run=["standard"], languages=[], cascades=[],
    )
    defaults.update(kw)
    return Report(**defaults)


def _lang_result(language, score, ci, delta=None, delta_ci=None, n=20, verified=True):
    return LanguageResult(
        language=language, n_items=n,
        quality={"value": score, "ci95": ci, "n": n, "precise": True},
        answer_quality=score,
        delta={"value": delta, "ci95": delta_ci} if delta is not None else None,
        delta_vs_baseline=delta,
        metrics={}, verified=verified,
    )


class TestBuildHeadline:
    def test_names_dominant_stage_and_share(self):
        by_stage = [
            _stage("s0_s1_input_query", "Input + query processing", 30.0),
            _stage("s2_retrieval", "Retrieval", None),
            _stage("s3_generation", "Generation", None),
            _stage("s4_rendering", "Output rendering", 2.0, noise=True),
        ]
        cascade = _cascade("hi-Latn", 92.0, 60.0, by_stage)
        report = _report(cascades=[cascade])
        headline = hr.build_headline(report)
        assert "Roman-Hindi" in headline
        assert "32" in headline  # total_loss
        assert "input + query processing" in headline.lower()
        assert "%" in headline

    def test_measured_loss_qualifier_when_stage_unmeasurable(self):
        by_stage = [
            _stage("s0_s1_input_query", "Input + query processing", 10.0),
            _stage("s2_retrieval", "Retrieval", None),
            _stage("s3_generation", "Generation", None),
            _stage("s4_rendering", "Output rendering", 5.0),
        ]
        cascade = _cascade("hi-Deva", 90.0, 75.0, by_stage)
        report = _report(cascades=[cascade])
        headline = hr.build_headline(report)
        assert "measured loss" in headline

    def test_no_dominant_stage_says_so(self):
        by_stage = [
            _stage("s0_s1_input_query", "Input + query processing", 2.0, noise=True),
            _stage("s4_rendering", "Output rendering", 1.0, noise=True),
        ]
        cascade = _cascade("hi-Deva", 90.0, 87.0, by_stage)
        report = _report(cascades=[cascade])
        headline = hr.build_headline(report)
        assert "no single stage" in headline

    def test_no_cascades_at_all(self):
        report = _report(cascades=[])
        assert "no cross-language comparison" in hr.build_headline(report)

    def test_no_language_worse_than_baseline(self):
        by_stage = [_stage("s0_s1_input_query", "Input + query processing", -5.0)]
        cascade = _cascade("hi-Deva", 80.0, 85.0, by_stage)
        report = _report(cascades=[cascade])
        headline = hr.build_headline(report)
        assert "No language scored below" in headline


class TestLanguageCards:
    def test_baseline_status(self):
        report = _report(languages=[_lang_result("en", 90.0, [80.0, 100.0])])
        cards = hr.language_cards(report)
        assert cards[0]["status"] == "baseline"

    def test_over_target_is_critical(self):
        report = _report(languages=[
            _lang_result("en", 90.0, [80.0, 100.0]),
            _lang_result("hi-Latn", 55.0, [40.0, 70.0], delta=-35.0, delta_ci=[-50.0, -20.0]),
        ])
        cards = hr.language_cards(report)
        assert cards[1]["status"] == "critical"

    def test_ci_crossing_zero_is_noise(self):
        report = _report(languages=[
            _lang_result("en", 90.0, [80.0, 100.0]),
            _lang_result("hi-Deva", 85.0, [70.0, 100.0], delta=-5.0, delta_ci=[-15.0, 5.0]),
        ])
        cards = hr.language_cards(report)
        assert cards[1]["status"] == "noise"

    def test_experimental_flag_carried(self):
        report = _report(languages=[_lang_result("ta-Taml", 70.0, [60.0, 80.0], verified=False)])
        cards = hr.language_cards(report)
        assert cards[0]["verified"] is False


class TestSegmentsForWaterfall:
    def test_measured_stage_depth(self):
        by_stage = [_stage("s0_s1_input_query", "Input + query processing", 12.0)]
        cascade = _cascade("hi-Deva", 90.0, 78.0, by_stage)
        segs = hr._segments_for(cascade)
        assert segs[0]["depth_start"] == 0.0
        assert segs[0]["depth_end"] == 12.0
        assert segs[0]["kind"] == "measured"

    def test_void_run_sized_by_remainder(self):
        by_stage = [
            _stage("s0_s1_input_query", "Input + query processing", 10.0),
            _stage("s2_retrieval", "Retrieval", None),
            _stage("s3_generation", "Generation", None),
            _stage("s4_rendering", "Output rendering", 5.0),
        ]
        cascade = _cascade("hi-Deva", 90.0, 60.0, by_stage)  # total_loss = 30
        segs = hr._segments_for(cascade)
        void = next(s for s in segs if s["kind"] == "void")
        # 30 total - 10 - 5 measured = 15 for the combined S2+S3 void
        assert void["depth_end"] - void["depth_start"] == pytest.approx(15.0)
        assert len(void["stages"]) == 2

    def test_negative_segment_flagged(self):
        by_stage = [_stage("s0_s1_input_query", "Input + query processing", -3.0)]
        cascade = _cascade("hi-Deva", 90.0, 93.0, by_stage)
        segs = hr._segments_for(cascade)
        assert segs[0]["negative"] is True
        assert segs[0]["depth_end"] < segs[0]["depth_start"]

    def test_noise_and_combined_flags_propagate(self):
        by_stage = [_stage("s0_s1_input_query", "Input + query processing", 4.0, isolated=False, noise=True)]
        cascade = _cascade("hi-Deva", 90.0, 86.0, by_stage)
        seg = hr._segments_for(cascade)[0]
        assert seg["noise"] is True
        assert seg["combined"] is True


class TestWaterfallSvg:
    def test_empty_cascades_returns_placeholder(self):
        out = hr.render_waterfall_svg([])
        assert "svg" not in out
        assert "No cascade data" in out

    def test_total_loss_sign_is_negated_for_display(self):
        # Regression: total_loss is positive-means-worse; displaying it
        # with a leading '+' reads as an improvement (the exact bug
        # attribution.py's own render_cascade warns about in its source).
        by_stage = [_stage("s0_s1_input_query", "Input + query processing", 20.0)]
        cascade = _cascade("hi-Latn", 90.0, 70.0, by_stage)
        svg = hr.render_waterfall_svg([cascade])
        assert "-20 pts" in svg
        assert "+20 pts" not in svg

    def test_not_measured_text_present_for_void(self):
        by_stage = [
            _stage("s0_s1_input_query", "Input + query processing", 10.0),
            _stage("s2_retrieval", "Retrieval", None),
            _stage("s4_rendering", "Output rendering", 5.0),
        ]
        cascade = _cascade("hi-Deva", 90.0, 60.0, by_stage)
        svg = hr.render_waterfall_svg([cascade])
        assert "not measured" in svg

    def test_dominant_segment_gets_accent_stroke(self):
        by_stage = [_stage("s0_s1_input_query", "Input + query processing", 20.0)]
        cascade = _cascade("hi-Latn", 90.0, 70.0, by_stage)
        svg = hr.render_waterfall_svg([cascade])
        assert hr.PALETTE["accent"] in svg


class TestReportHistory:
    def test_reads_sibling_reports_sorted(self, tmp_path):
        for i, ts in enumerate(["2026-01-02T00:00:00", "2026-01-01T00:00:00"]):
            d = tmp_path / f"run-{i}"
            d.mkdir()
            (d / "report.json").write_text(json.dumps({
                "generated_at": ts, "system_label": f"run-{i}",
                "languages": [{"language": "en", "quality": {"value": 90.0}}],
            }))
        history = hr.scan_report_history(tmp_path)
        assert [h["system_label"] for h in history] == ["run-1", "run-0"]

    def test_excludes_given_path(self, tmp_path):
        d1, d2 = tmp_path / "run-a", tmp_path / "run-b"
        d1.mkdir(); d2.mkdir()
        for d in (d1, d2):
            (d / "report.json").write_text(json.dumps({
                "generated_at": "2026-01-01T00:00:00", "system_label": d.name, "languages": [],
            }))
        history = hr.scan_report_history(tmp_path, exclude=d1)
        assert [h["system_label"] for h in history] == ["run-b"]

    def test_skips_malformed_reports(self, tmp_path):
        d = tmp_path / "broken"
        d.mkdir()
        (d / "report.json").write_text("{not valid json")
        assert hr.scan_report_history(tmp_path) == []

    def test_missing_directory_returns_empty(self, tmp_path):
        assert hr.scan_report_history(tmp_path / "does-not-exist") == []


class TestRenderHtmlSelfContained:
    """The one non-negotiable structural requirement: no external requests."""

    def _full_report(self) -> Report:
        by_stage = [
            _stage("s0_s1_input_query", "Input + query processing", 20.0),
            _stage("s2_retrieval", "Retrieval", None),
            _stage("s3_generation", "Generation", None),
            _stage("s4_rendering", "Output rendering", 3.0, noise=True),
        ]
        return _report(
            languages=[
                _lang_result("en", 90.0, [80.0, 100.0]),
                _lang_result("hi-Latn", 65.0, [50.0, 80.0], delta=-25.0, delta_ci=[-40.0, -10.0]),
            ],
            cascades=[_cascade("hi-Latn", 90.0, 65.0, by_stage)],
            taxonomy={"over_refusal": 3},
            warnings=["some caveat"],
        )

    def test_no_external_references(self):
        html = hr.render_html(self._full_report())
        for token in ("http://", "https://", "<link ", "src=\"http", "@import"):
            assert token not in html

    def test_both_tabs_present(self):
        html = hr.render_html(self._full_report())
        assert 'id="tab-overview"' in html and 'id="tab-detail"' in html

    def test_is_a_complete_document(self):
        html = hr.render_html(self._full_report())
        assert html.strip().startswith("<!doctype html>")
        assert "</html>" in html

    def test_works_from_json_roundtrip_not_just_live_object(self):
        # The regenerate-from-report.json path: cascade_objects is excluded
        # from serialisation, so this must not depend on it.
        report = self._full_report()
        reloaded = Report.model_validate_json(report.model_dump_json())
        assert reloaded.cascade_objects == []
        html = hr.render_html(reloaded)
        assert "Roman-Hindi" in html or "hi-Latn" in html

    def test_write_html_creates_file(self, tmp_path):
        path = hr.write_html(self._full_report(), tmp_path / "out" / "report.html")
        assert path.exists()
        assert "<!doctype html>" in path.read_text(encoding="utf-8")
