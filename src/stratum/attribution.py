"""Loss attribution — the cascade.

A single end-to-end score tells you a language is worse. It does not tell
you which stage made it worse, and that is the only thing you can act on.

The method is a nested oracle ladder. Each pass repairs one more stage
using known-good inputs, so the passes form a monotone chain from the
system as it really is to the baseline language:

    standard        nothing repaired
    oracle_query    S0 + S1 repaired  (gold baseline query goes to retrieval)
    oracle_context  S0..S2 repaired   (gold chunks go to generation)
    oracle_answer   S0..S3 repaired   (gold answer goes to rendering only)
    baseline        everything        (the baseline-language run)

Each stage's loss is the gap between adjacent rungs, so the losses sum to
the total by construction — there is no residual to explain away:

    S0+S1 = score(oracle_query)   - score(standard)
    S2    = score(oracle_context) - score(oracle_query)
    S3    = score(oracle_answer)  - score(oracle_context)
    S4    = score(baseline)       - score(oracle_answer)

Two honesty constraints are enforced here rather than left to the reader.

First, the ladder is not guaranteed monotone on a finite sample. Repairing
a stage can appear to *hurt* through noise alone. Such rungs are reported
as measured, flagged `noisy`, never clamped to zero — clamping would hide
sampling error and inflate the neighbouring stage.

Second, a stage whose oracle pass the endpoint does not support is marked
`not_isolated` and its loss is merged into an explicitly named combined
band. It is never silently attributed to an adjacent stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .stats import Estimate, bootstrap_paired_difference, crosses_zero

# The ladder, in repair order. Each entry names the stages that the pass
# repairs relative to the pass before it.
LADDER: list[tuple[str, str, str]] = [
    # (pass_name, stage_key, human label)
    ("standard", "", ""),
    ("oracle_query", "s0_s1_input_query", "Input + query processing"),
    ("oracle_context", "s2_retrieval", "Retrieval"),
    ("oracle_answer", "s3_generation", "Generation"),
    ("baseline", "s4_rendering", "Output rendering"),
]


@dataclass
class StageLoss:
    stage: str
    label: str
    estimate: Estimate
    isolated: bool = True
    note: str = ""

    @property
    def points(self) -> float | None:
        return self.estimate.value

    @property
    def is_noise(self) -> bool:
        """Whether the sample can distinguish this loss from zero."""
        return crosses_zero(self.estimate)

    @property
    def is_negative(self) -> bool:
        """Repairing this stage appeared to make things worse."""
        return self.estimate.value is not None and self.estimate.value < 0

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "label": self.label,
            "points_lost": self.points,
            "ci95": [self.estimate.ci_low, self.estimate.ci_high],
            "isolated": self.isolated,
            "indistinguishable_from_zero": self.is_noise,
            "note": self.note,
        }


@dataclass
class Cascade:
    language: str
    baseline_language: str
    standard_score: float | None
    baseline_score: float | None
    losses: list[StageLoss] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_loss(self) -> float | None:
        if self.baseline_score is None or self.standard_score is None:
            return None
        return round(self.baseline_score - self.standard_score, 2)

    @property
    def dominant(self) -> StageLoss | None:
        """The stage worth fixing first, if the sample supports naming one."""
        real = [
            l for l in self.losses
            if l.points is not None and l.points > 0 and not l.is_noise
        ]
        return max(real, key=lambda l: l.points) if real else None

    @property
    def has_unmeasurable(self) -> bool:
        return any(l.points is None for l in self.losses)

    def sums_correctly(self, tolerance: float | None = None) -> bool:
        """The ladder telescopes, so this should always hold. Asserted, not assumed.

        Tolerance scales with the number of rungs: every rung is rounded to two
        decimals, so the permissible drift is the accumulated rounding budget,
        not a fixed constant.
        """
        if tolerance is None:
            tolerance = 0.005 * (len(self.losses) + 2) + 1e-9
        if self.total_loss is None or self.has_unmeasurable:
            return False
        parts = [l.points for l in self.losses if l.points is not None]
        return abs(sum(parts) - self.total_loss) <= tolerance

    def as_dict(self) -> dict:
        return {
            "language": self.language,
            "baseline_language": self.baseline_language,
            "method": "nested_oracle_ladder",
            "baseline_score": self.baseline_score,
            "final_score": self.standard_score,
            "total_loss": self.total_loss,
            "by_stage": [l.as_dict() for l in self.losses],
            "sums_correctly": self.sums_correctly(),
            "dominant_stage": self.dominant.stage if self.dominant else None,
            "warnings": self.warnings,
        }


def build_cascade(
    language: str,
    baseline_language: str,
    *,
    pass_item_scores: dict[str, list[float | None]],
    baseline_item_scores: list[float | None],
    supported_passes: list[str],
    scale: float = 100.0,
    unmeasurable_stages: tuple[str, ...] = (),
) -> Cascade:
    """Compute the cascade for one language.

    `pass_item_scores` maps pass name to per-item composite scores, ordered
    consistently across passes so differences are paired. `baseline_item_scores`
    is the same for the baseline-language run.
    """
    from .stats import bootstrap_mean

    rungs: dict[str, list[float | None]] = dict(pass_item_scores)
    rungs["baseline"] = baseline_item_scores

    standard = bootstrap_mean(rungs.get("standard", []), scale=scale)
    baseline = bootstrap_mean(baseline_item_scores, scale=scale)

    cascade = Cascade(
        language=language,
        baseline_language=baseline_language,
        standard_score=standard.value,
        baseline_score=baseline.value,
    )

    # Walk the ladder, collapsing over any rung the endpoint cannot provide
    # so that a missing pass widens a band rather than corrupting a neighbour.
    previous_pass = "standard"
    pending_labels: list[str] = []
    pending_stages: list[str] = []

    for pass_name, stage_key, label in LADDER[1:]:
        available = pass_name == "baseline" or pass_name in supported_passes

        if stage_key in unmeasurable_stages:
            cascade.losses.append(
                StageLoss(
                    stage=stage_key,
                    label=label,
                    estimate=Estimate(value=None, n=0),
                    isolated=False,
                    note="not measurable without a calibrated judge — no "
                         "deterministic check reads answer content",
                )
            )
            previous_pass = pass_name
            continue

        if not available:
            pending_stages.append(stage_key)
            pending_labels.append(label)
            cascade.warnings.append(
                f"endpoint does not support '{pass_name}' — "
                f"{label.lower()} could not be isolated"
            )
            continue

        est = bootstrap_paired_difference(
            rungs.get(pass_name, []),
            rungs.get(previous_pass, []),
            scale=scale,
        )

        if pending_stages:
            combined_stage = "+".join(pending_stages + [stage_key])
            combined_label = " + ".join(pending_labels + [label])
            cascade.losses.append(
                StageLoss(
                    stage=combined_stage,
                    label=combined_label,
                    estimate=est,
                    isolated=False,
                    note="combined band — an intermediate oracle pass was unavailable",
                )
            )
            pending_stages, pending_labels = [], []
        else:
            note = ""
            if est.value is not None and est.value < 0:
                note = "negative: repairing this stage did not improve the score"
            cascade.losses.append(
                StageLoss(stage=stage_key, label=label, estimate=est, note=note)
            )

        previous_pass = pass_name

    if unmeasurable_stages:
        cascade.warnings.append(
            "some stages are not measurable without a judge; the remaining "
            "rungs absorb their share and are upper bounds"
        )
    elif not cascade.sums_correctly():
        cascade.warnings.append(
            "stage losses do not sum to the total — treat the breakdown as unreliable"
        )

    noisy = [l.label for l in cascade.losses if l.is_noise]
    if noisy:
        cascade.warnings.append(
            f"indistinguishable from zero at this sample size: {', '.join(noisy)}"
        )

    return cascade


def render_cascade(cascade: Cascade, width: int = 28) -> str:
    """Text waterfall. The headline artifact of the report."""
    lines: list[str] = []
    total = cascade.total_loss

    if total is None:
        return f"  {cascade.language} · cascade unavailable (missing baseline)"

    lines.append(
        # total_loss is a *loss*: positive means worse than baseline. Printing it
        # with a leading '+' read as an improvement, which inverted the finding.
        f"  {cascade.language} · {-total:+.1f} points vs {cascade.baseline_language}"
    )
    lines.append("")
    lines.append(f"    {'baseline':<34}{cascade.baseline_score:>7.1f}")

    scale_max = max(
        (abs(l.points) for l in cascade.losses if l.points is not None), default=1.0
    ) or 1.0

    # `total` is baseline - standard: the true end-to-end gap. When a stage
    # is `not measurable`, the visible rungs do not sum to it — some of the
    # gap is hiding in a blank rung. Sharing against `total` in that case
    # would print "100% of loss" for a stage that, by construction, cannot
    # account for more than the *measured* portion.
    measured_positive = sum(
        l.points for l in cascade.losses if l.points is not None and l.points > 0
    )

    running = cascade.baseline_score or 0.0
    for loss in cascade.losses:
        if loss.points is None:
            lines.append(f"    {loss.label:<34}{'—':>7}   not measured")
            continue

        running -= loss.points
        bar_len = int(abs(loss.points) / scale_max * width)
        bar = ("█" if loss.points >= 0 else "░") * max(bar_len, 1 if loss.points else 0)

        flags = []
        if not loss.isolated:
            flags.append("combined")
        if loss.is_noise:
            flags.append("noise")
        if loss.is_negative:
            flags.append("negative")
        flag = f"  [{', '.join(flags)}]" if flags else ""

        marker = ""
        dom = cascade.dominant
        if dom is not None and loss.stage == dom.stage:
            if cascade.has_unmeasurable:
                share = abs(loss.points / measured_positive) * 100 if measured_positive else 0
                marker = f"   <- {share:.0f}% of measured loss"
            else:
                share = abs(loss.points / total) * 100 if total else 0
                marker = f"   <- {share:.0f}% of loss"

        lines.append(
            f"    {loss.label:<34}{-loss.points:>7.1f}  {bar}{flag}{marker}"
        )

    lines.append(f"    {'final':<34}{cascade.standard_score:>7.1f}")

    if cascade.warnings:
        lines.append("")
        for w in cascade.warnings:
            lines.append(f"    ! {w}")

    return "\n".join(lines)
