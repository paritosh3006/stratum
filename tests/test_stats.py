from stratum.stats import (
    bootstrap_difference, bootstrap_mean, bootstrap_paired_difference, crosses_zero,
)


class TestBootstrapMean:
    def test_point_estimate_matches_mean(self):
        e = bootstrap_mean([1.0, 0.0, 1.0, 1.0], scale=100.0)
        assert e.value == 75.0 and e.n == 4

    def test_nulls_dropped_not_zeroed(self):
        # A metric that does not apply must not drag the mean down.
        assert bootstrap_mean([1.0, None, 1.0]).value == 1.0

    def test_empty_is_none(self):
        assert bootstrap_mean([None, None]).value is None

    def test_single_observation_has_no_interval(self):
        e = bootstrap_mean([1.0])
        assert e.value is not None and e.ci_low is None

    def test_interval_contains_point(self):
        e = bootstrap_mean([1, 0, 1, 0, 1, 1, 0, 1], scale=100.0)
        assert e.ci_low <= e.value <= e.ci_high

    def test_more_data_narrows_interval(self):
        few = bootstrap_mean([1, 0] * 5, scale=100.0)
        many = bootstrap_mean([1, 0] * 100, scale=100.0)
        assert many.half_width < few.half_width

    def test_precision_flag_tracks_target(self):
        assert bootstrap_mean([1, 0] * 200, scale=100.0).is_precise
        assert not bootstrap_mean([1, 0, 1], scale=100.0).is_precise


class TestPairedDifference:
    def test_paired_is_tighter_than_unpaired(self):
        # Same items, consistent per-item gap: pairing should reveal the signal.
        a = [0.9, 0.7, 0.5, 0.3, 0.9, 0.7, 0.5, 0.3]
        b = [0.8, 0.6, 0.4, 0.2, 0.8, 0.6, 0.4, 0.2]
        paired = bootstrap_paired_difference(a, b, scale=100.0)
        unpaired = bootstrap_difference(a, b, scale=100.0)
        assert paired.half_width < unpaired.half_width

    def test_constant_gap_excludes_zero(self):
        a = [0.9] * 12
        b = [0.5] * 12
        assert not crosses_zero(bootstrap_paired_difference(a, b, scale=100.0))

    def test_no_effect_includes_zero(self):
        vals = [1.0, 0.0] * 10
        assert crosses_zero(bootstrap_paired_difference(vals, vals, scale=100.0))

    def test_unequal_pairs_dropped(self):
        assert bootstrap_paired_difference([1.0, None, 1.0], [0.0, 0.0, None]).n == 1
