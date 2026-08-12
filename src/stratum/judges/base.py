"""Judge backends.

Judged metrics are optional in stratum. The deterministic checks — placeholder
integrity, glossary adherence, numeral integrity, script detection, retrieval
delta — need no model, cost nothing, and are reproducible. They are the core.

A judge adds faithfulness and answer correctness on top. It is pluggable
because judge quality varies enormously by language, and a judge that has not
been calibrated against human labels for a given language is not evidence.

The contract enforced here: a judge cannot be used for a language without a
calibration record. `Calibration` carries the agreement score, and the report
marks any judged metric below the threshold as low-confidence rather than
printing it as fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Judgement:
    score: float
    reasoning: str = ""
    raw: str = ""


@dataclass(frozen=True)
class Calibration:
    """Agreement between this judge and human labels, for one language."""

    language: str
    metric: str
    kappa: float
    n_labelled: int
    judge_id: str

    #: Below this, the judge is not treated as evidence for the language.
    threshold: float = 0.60

    @property
    def is_trustworthy(self) -> bool:
        return self.kappa >= self.threshold

    @property
    def confidence(self) -> str:
        if self.kappa >= 0.75:
            return "high"
        if self.kappa >= self.threshold:
            return "moderate"
        return "low"

    def as_dict(self) -> dict:
        return {
            "language": self.language,
            "metric": self.metric,
            "kappa": round(self.kappa, 3),
            "n_labelled": self.n_labelled,
            "judge_id": self.judge_id,
            "confidence": self.confidence,
            "trustworthy": self.is_trustworthy,
        }


class JudgeBackend(Protocol):
    """Any model that can score an answer. Local, hosted, or human."""

    judge_id: str

    def judge_faithfulness(
        self, answer: str, context: list[str], language: str
    ) -> Judgement:
        """Fraction of the answer's atomic claims supported by the context."""
        ...

    def judge_correctness(
        self, answer: str, reference: str, language: str
    ) -> Judgement:
        """Agreement with the reference answer, on a 0-3 rubric."""
        ...


@dataclass
class CalibrationRegistry:
    """Gatekeeps judged metrics by language.

    Registered from `stratum calibrate` output. A judged metric for an
    uncalibrated language is withheld, not estimated — this is the whole
    reason the registry exists.
    """

    records: dict[tuple[str, str], Calibration] = field(default_factory=dict)

    def register(self, cal: Calibration) -> None:
        self.records[(cal.language, cal.metric)] = cal

    def get(self, language: str, metric: str) -> Calibration | None:
        return self.records.get((language, metric))

    def permits(self, language: str, metric: str, judge_id: str | None = None) -> bool:
        """Whether a judged score for (language, metric) may be trusted.

        `judge_id`, when given, must match the judge that earned the
        calibration record. A kappa measured against one model says
        nothing about a different one — swap `stratum run --judge`'s model
        after calibrating and the registry must stop permitting, not keep
        applying a trust score that no longer describes what's running.
        """
        cal = self.get(language, metric)
        if cal is None or not cal.is_trustworthy:
            return False
        if judge_id is not None and cal.judge_id != judge_id:
            return False
        return True

    def status(self, language: str, metric: str) -> str:
        cal = self.get(language, metric)
        if cal is None:
            return "uncalibrated"
        return cal.confidence
