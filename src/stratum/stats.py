"""Uncertainty quantification.

Every reported number carries its sample size and a confidence interval.
A score without error bars invites false precision, and this tool exists
to argue against exactly that habit.

Bootstrap is used rather than a normal approximation because most of the
metrics here are bounded proportions on small samples, where the normal
approximation is poor near 0 and 1.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its uncertainty."""

    value: float | None
    n: int
    ci_low: float | None = None
    ci_high: float | None = None

    @property
    def half_width(self) -> float | None:
        if self.ci_low is None or self.ci_high is None:
            return None
        return (self.ci_high - self.ci_low) / 2

    @property
    def is_precise(self) -> bool:
        """Whether the interval is tight enough to support a claim.

        The 5-point target used throughout the report is meaningless if
        the interval is wider than the target itself.
        """
        hw = self.half_width
        return hw is not None and hw <= 5.0

    def format(self, decimals: int = 1) -> str:
        if self.value is None:
            return "—"
        if self.ci_low is None:
            return f"{self.value:.{decimals}f}"
        return f"{self.value:.{decimals}f} ±{self.half_width:.{decimals}f}"

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "n": self.n,
            "ci95": [self.ci_low, self.ci_high]
            if self.ci_low is not None
            else None,
            "precise": self.is_precise,
        }


def bootstrap_mean(
    values: list[float | None],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    scale: float = 1.0,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap over the mean of the non-null values.

    Nulls are dropped rather than treated as zero: a metric that does not
    apply to an item (no placeholders present, no gold chunks) must not
    drag the mean down.
    """
    clean = [v for v in values if v is not None]
    n = len(clean)

    if n == 0:
        return Estimate(value=None, n=0)

    point = statistics.fmean(clean) * scale

    if n == 1:
        # One observation carries no information about spread. Say so
        # rather than emitting a zero-width interval.
        return Estimate(value=point, n=1)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        sample = [clean[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample) * scale)

    means.sort()
    alpha = (1 - confidence) / 2
    lo = means[int(alpha * iterations)]
    hi = means[min(int((1 - alpha) * iterations), iterations - 1)]

    return Estimate(value=round(point, 2), n=n, ci_low=round(lo, 2), ci_high=round(hi, 2))


def bootstrap_difference(
    a: list[float | None],
    b: list[float | None],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    scale: float = 1.0,
    seed: int = 0,
) -> Estimate:
    """CI for mean(a) - mean(b), resampling both independently.

    Used for deltas against the baseline language and for the per-stage
    losses in the cascade, where the question is not "is this number
    large" but "is this difference distinguishable from zero".
    """
    ca = [v for v in a if v is not None]
    cb = [v for v in b if v is not None]
    if not ca or not cb:
        return Estimate(value=None, n=0)

    point = (statistics.fmean(ca) - statistics.fmean(cb)) * scale
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(iterations):
        sa = statistics.fmean([ca[rng.randrange(len(ca))] for _ in range(len(ca))])
        sb = statistics.fmean([cb[rng.randrange(len(cb))] for _ in range(len(cb))])
        diffs.append((sa - sb) * scale)

    diffs.sort()
    alpha = (1 - confidence) / 2
    lo = diffs[int(alpha * iterations)]
    hi = diffs[min(int((1 - alpha) * iterations), iterations - 1)]

    return Estimate(
        value=round(point, 2),
        n=min(len(ca), len(cb)),
        ci_low=round(lo, 2),
        ci_high=round(hi, 2),
    )


def crosses_zero(est: Estimate) -> bool:
    """Whether a difference is indistinguishable from zero.

    The cascade uses this to mark stage losses that the sample cannot
    actually support, instead of printing a confident-looking number.
    """
    if est.ci_low is None or est.ci_high is None:
        return True
    return est.ci_low <= 0.0 <= est.ci_high


def cohens_kappa(a: list[int], b: list[int], *, weights: str = "linear") -> float:
    """Weighted agreement between two raters' categorical judgements.

    `weights="linear"` (the default, and what calibrate.py always uses)
    penalises a two-point disagreement on the 0-3 rubric twice as much as a
    one-point one. Plain nominal kappa (`weights="none"`) treats every
    disagreement as equally bad regardless of how far apart the ratings
    are — correct for unordered categories, wrong for an ordinal rubric
    where a human 3 and a judge 0 is a much worse disagreement than a human
    3 and a judge 2. `weights="quadratic"` penalises distance more steeply
    still, for callers who want large disagreements to dominate the score.

    Weight is computed from the actual rating *values*, not their rank
    among whatever happened to appear in this particular sample — a 0-vs-3
    disagreement is twice as bad as a 0-vs-1.5-away one, regardless of what
    other values occur. There is deliberately no parameter to pin a wider
    category domain than what was observed (e.g. forcing the rubric's full
    0-3 range even when a sample only ever hit 1s and 2s): linear and
    quadratic weights normalise by the observed span, and that
    normalisation is a single constant multiplied through every term of
    both the observed and expected sums alike, so it cancels in their
    ratio — pinning a wider span cannot change the resulting κ. Checked by
    computing it both ways rather than assumed; see
    `TestCohensKappa::test_weight_normalisation_is_invariant_to_scale`.
    """
    if len(a) != len(b):
        raise ValueError(f"rating lists must be the same length: {len(a)} != {len(b)}")
    if not a:
        raise ValueError("cannot compute kappa over zero ratings")

    cats = sorted(set(a) | set(b))
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a)
    lo, hi = cats[0], cats[-1]
    span = hi - lo

    def w(ci: int, cj: int) -> float:
        if weights == "none":
            return 0.0 if ci == cj else 1.0
        if weights == "linear":
            return abs(ci - cj) / span if span else 0.0
        if weights == "quadratic":
            return ((ci - cj) / span) ** 2 if span else 0.0
        raise ValueError(f"unknown weights: {weights!r}")

    observed = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[idx[x]][idx[y]] += 1

    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    po = sum(
        w(cats[i], cats[j]) * observed[i][j] for i in range(k) for j in range(k)
    ) / n
    pe = sum(
        w(cats[i], cats[j]) * row_totals[i] * col_totals[j]
        for i in range(k) for j in range(k)
    ) / (n * n)

    if pe == 0:
        # Every rating either agrees outright or falls on a weight-zero
        # pair (only possible when every rating that occurred is the same
        # single value) — no disagreement was possible under this
        # weighting to begin with, so there is nothing chance could have
        # produced instead. Agreement, not an undefined ratio.
        return 1.0

    return 1.0 - po / pe


def bootstrap_paired_difference(
    a: list[float | None],
    b: list[float | None],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    scale: float = 1.0,
    seed: int = 0,
) -> Estimate:
    """CI for the mean per-item difference, resampling items rather than arms.

    The cascade compares the *same* item under two passes, so the two arms are
    paired. Resampling them independently discards that pairing and inflates
    the interval enough to mark real effects as noise — which is the wrong
    error to make in a tool whose purpose is telling signal from noise.

    Items where either arm is missing are dropped, since no difference exists
    for them to contribute.
    """
    pairs = [
        (x, y) for x, y in zip(a, b) if x is not None and y is not None
    ]
    n = len(pairs)
    if n == 0:
        return Estimate(value=None, n=0)

    diffs = [(x - y) * scale for x, y in pairs]
    point = statistics.fmean(diffs)

    if n == 1:
        return Estimate(value=round(point, 2), n=1)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        means.append(statistics.fmean([diffs[rng.randrange(n)] for _ in range(n)]))

    means.sort()
    alpha = (1 - confidence) / 2
    lo = means[int(alpha * iterations)]
    hi = means[min(int((1 - alpha) * iterations), iterations - 1)]

    return Estimate(value=round(point, 2), n=n, ci_low=round(lo, 2), ci_high=round(hi, 2))
