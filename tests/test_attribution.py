import pytest

from stratum.attribution import build_cascade, render_cascade


def _ladder(standard, oq, oc, oa, baseline, n=20):
    """Build per-item score lists at each rung with a constant per-item value."""
    return dict(
        pass_item_scores={
            "standard": [standard] * n,
            "oracle_query": [oq] * n,
            "oracle_context": [oc] * n,
            "oracle_answer": [oa] * n,
        },
        baseline_item_scores=[baseline] * n,
        supported_passes=["standard", "oracle_query", "oracle_context", "oracle_answer"],
    )


class TestCascade:
    def test_losses_sum_to_total(self):
        c = build_cascade("hi", "en", **_ladder(0.5, 0.6, 0.7, 0.8, 0.9))
        assert c.total_loss == pytest.approx(40.0, abs=0.1)
        assert c.sums_correctly()

    def test_dominant_stage_identified(self):
        # All the loss sits in rendering.
        c = build_cascade("hi", "en", **_ladder(0.5, 0.5, 0.5, 0.5, 0.9))
        assert c.dominant.stage == "s4_rendering"

    def test_dominant_ignores_noise(self):
        c = build_cascade("hi", "en", **_ladder(0.9, 0.9, 0.9, 0.9, 0.9))
        assert c.dominant is None

    def test_missing_pass_produces_combined_band(self):
        c = build_cascade(
            "hi", "en",
            pass_item_scores={"standard": [0.5] * 20, "oracle_context": [0.8] * 20},
            baseline_item_scores=[0.9] * 20,
            supported_passes=["standard", "oracle_context"],
        )
        combined = [l for l in c.losses if not l.isolated]
        assert combined, "an unavailable rung must widen a band, not vanish"
        assert any("could not be isolated" in w for w in c.warnings)

    def test_unmeasurable_stage_is_not_guessed(self):
        c = build_cascade(
            "hi", "en",
            **_ladder(0.5, 0.6, 0.7, 0.8, 0.9),
            unmeasurable_stages=("s2_retrieval", "s3_generation"),
        )
        retrieval = next(l for l in c.losses if l.stage == "s2_retrieval")
        assert retrieval.points is None
        assert not c.sums_correctly()  # cannot claim completeness

    def test_negative_loss_reported_not_clamped(self):
        # Repairing query processing appears to hurt: must be shown, not hidden.
        c = build_cascade("hi", "en", **_ladder(0.7, 0.6, 0.7, 0.8, 0.9))
        s01 = next(l for l in c.losses if l.stage == "s0_s1_input_query")
        assert s01.is_negative

    def test_render_mentions_baseline_and_final(self):
        out = render_cascade(build_cascade("hi", "en", **_ladder(0.5, 0.6, 0.7, 0.8, 0.9)))
        assert "baseline" in out and "final" in out

    def test_dominant_share_is_of_measured_loss_when_stages_unmeasurable(self):
        # total_loss (baseline - standard) is 40, but s2/s3 are unmeasurable,
        # so the visible rungs (s0_s1=10, s4=40) do not sum to it. Sharing
        # the dominant stage against the raw total prints "100% of loss" for
        # a stage that, by construction, cannot account for the whole gap —
        # some of it may be hiding in the blank rungs.
        c = build_cascade(
            "hi", "en",
            **_ladder(0.5, 0.6, 0.7, 0.5, 0.9),
            unmeasurable_stages=("s2_retrieval", "s3_generation"),
        )
        assert c.total_loss == pytest.approx(40.0, abs=0.1)
        assert c.dominant.stage == "s4_rendering"

        out = render_cascade(c)
        assert "100% of loss" not in out
        # 40 / (10 + 40) measured-positive points = 80%, not the 100% a
        # denominator of the raw total would have printed.
        assert "80% of measured loss" in out


class TestSignConvention:
    def test_header_shows_loss_as_negative(self):
        # A language scoring below baseline must not read as an improvement.
        c = build_cascade("hi", "en", **_ladder(0.5, 0.6, 0.7, 0.8, 0.9))
        header = render_cascade(c).splitlines()[0]
        assert "-40.0 points" in header or "−40.0 points" in header

    def test_language_above_baseline_shows_positive(self):
        c = build_cascade("hi", "en", **_ladder(0.9, 0.9, 0.9, 0.9, 0.7))
        header = render_cascade(c).splitlines()[0]
        assert "+20.0 points" in header
