import pytest

from stratum.stats import (
    bootstrap_difference, bootstrap_mean, bootstrap_paired_difference,
    cohens_kappa, crosses_zero,
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


class TestCohensKappa:
    def test_perfect_agreement_is_one(self):
        ratings = [0, 1, 2, 3, 0, 1, 2, 3]
        assert cohens_kappa(ratings, ratings) == 1.0

    def test_matches_textbook_unweighted_two_by_two(self):
        # Hand-computed: po=0.75, pe=0.5 -> kappa=(0.75-0.5)/(1-0.5)=0.5
        a = [1, 1, 1, 0, 0, 0, 1, 0]
        b = [1, 1, 0, 0, 0, 0, 1, 1]
        assert cohens_kappa(a, b, weights="none") == pytest.approx(0.5)

    def test_linear_weighting_penalises_distant_disagreement_more(self):
        # Same 4 items, same 4-vs-0 agreement count, but one disagreement
        # set is off-by-one and the other is off-by-three on the 0-3 scale.
        adjacent = cohens_kappa([0, 1, 2, 3], [1, 2, 3, 0])
        extreme = cohens_kappa([0, 0, 3, 3], [3, 3, 0, 0])
        assert adjacent > extreme

    def test_single_category_both_raters_is_perfect_agreement(self):
        # No disagreement was possible under this weighting to begin with.
        assert cohens_kappa([2, 2, 2, 2], [2, 2, 2, 2]) == 1.0

    def test_weight_normalisation_is_invariant_to_scale(self):
        # Linear/quadratic weights normalise by the observed span, but that
        # normalisation is a single constant through every term of both the
        # observed and expected sums — it cancels in their ratio. Doubling
        # every category label doubles every pairwise distance *and* the
        # span by the same factor, so kappa must come out identical.
        a, b = [0, 1, 2, 0, 1, 2], [1, 2, 0, 1, 2, 0]
        scaled_a, scaled_b = [v * 2 for v in a], [v * 2 for v in b]
        assert cohens_kappa(a, b) == pytest.approx(cohens_kappa(scaled_a, scaled_b))

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            cohens_kappa([0, 1], [0, 1, 2])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa([], [])
