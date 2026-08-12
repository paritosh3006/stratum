"""Report models and rendering.

The JSON is the contract; the terminal view is a rendering of it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Failure(BaseModel):
    item_id: str
    language: str
    cls: str
    stage: str
    slice: str
    query: str
    output: str
    detail: str


class LanguageResult(BaseModel):
    language: str
    n_items: int
    quality: dict[str, Any] = Field(default_factory=dict)
    answer_quality: float | None = None
    delta: dict[str, Any] | None = None
    delta_vs_baseline: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    latency_p50: float | None = None
    latency_p90: float | None = None

    #: False when the language's dataset and judge have not been checked by a
    #: native speaker. Such numbers are reported as experimental, never as fact.
    verified: bool = True

    def metric_value(self, name: str) -> float | None:
        m = self.metrics.get(name)
        return m.get("value") if isinstance(m, dict) else m


class Gate(BaseModel):
    metric: str
    languages: list[str] | str = "all"
    min_absolute: float | None = None
    max_regression: float | None = None

    observed: float | None = None
    failing_language: str | None = None
    passed: bool | None = None
    skipped_reason: str | None = None


class Report(BaseModel):
    schema_version: str = "2.0"
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    system_label: str
    baseline_language: str
    n_items: int
    passes_run: list[str] = Field(default_factory=lambda: ["standard"])
    languages: list[LanguageResult]
    cascades: list[dict] = Field(default_factory=list)
    taxonomy: dict[str, int] = Field(default_factory=dict)
    failures: list[Failure] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    gates: list[Gate] = Field(default_factory=list)

    #: Cascade objects kept out of serialisation; the dicts above are the contract.
    cascade_objects: list = Field(default_factory=list, exclude=True, repr=False)

    # ------------------------------------------------------------------
    def evaluate_gates(self, gates: list[Gate]) -> "Report":
        evaluated: list[Gate] = []
        for gate in gates:
            langs = (
                [l.language for l in self.languages]
                if gate.languages == "all"
                else list(gate.languages)
            )
            worst_val, worst_lang, passed, skipped = None, None, True, None

            for lr in self.languages:
                if lr.language not in langs:
                    continue
                if not lr.verified:
                    # Experimental languages inform, they do not block.
                    skipped = "experimental languages excluded"
                    continue
                val = lr.metric_value(gate.metric)
                if val is None:
                    continue
                if gate.min_absolute is not None and val < gate.min_absolute:
                    passed = False
                    if worst_val is None or val < worst_val:
                        worst_val, worst_lang = val, lr.language

            evaluated.append(gate.model_copy(update={
                "observed": worst_val,
                "failing_language": worst_lang,
                "passed": passed,
                "skipped_reason": skipped,
            }))
        self.gates = evaluated
        return self

    @property
    def status(self) -> str:
        return "failed" if any(g.passed is False for g in self.gates) else "passed"

    # ------------------------------------------------------------------
    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "report.json"
        payload = self.model_dump()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    def render_terminal(self, target_delta: float = -5.0) -> str:
        from .attribution import render_cascade

        L: list[str] = []
        L.append(f"stratum · {self.system_label}")
        L.append(
            f"{self.n_items} items · {len(self.languages)} language variants · "
            f"passes: {', '.join(self.passes_run)}"
        )
        L.append("")
        L.append(
            f"  {'LANGUAGE':<12}{'QUALITY':>14}   {'Δ vs baseline':<24}STATUS"
        )

        for lr in self.languages:
            q = lr.quality
            qs = (
                f"{q['value']:.1f} ±{(q['ci95'][1] - q['ci95'][0]) / 2:.1f}"
                if q.get("value") is not None and q.get("ci95")
                else (f"{q['value']:.1f}" if q.get("value") is not None else "—")
            )

            if lr.language == self.baseline_language:
                ds, status = "—", "baseline"
            elif lr.delta and lr.delta.get("value") is not None:
                d = lr.delta
                ds = f"{d['value']:+.1f} [{d['ci95'][0]:+.1f},{d['ci95'][1]:+.1f}]"
                if d["ci95"][1] < 0 and d["value"] < target_delta:
                    status = "OVER TARGET"
                elif d["ci95"][0] <= 0 <= d["ci95"][1]:
                    status = "indistinguishable"
                else:
                    status = "ok"
            else:
                ds, status = "—", ""

            if not lr.verified:
                status = f"{status} · experimental"

            L.append(f"  {lr.language:<12}{qs:>14}   {ds:<24}{status}")

        L.append("")
        L.append("  n per language: " + ", ".join(
            f"{lr.language}={lr.n_items}" for lr in self.languages
        ))

        # -- cascades ------------------------------------------------------
        if self.cascade_objects:
            L.append("")
            L.append("  WHERE THE QUALITY WENT")
            for c in self.cascade_objects:
                L.append("")
                L.append(render_cascade(c))
        elif len(self.passes_run) == 1:
            L.append("")
            L.append("  WHERE THE QUALITY WENT")
            L.append("    unavailable — endpoint supports no oracle passes")

        # -- taxonomy ------------------------------------------------------
        if self.taxonomy:
            L.append("")
            L.append("  FAILURE CLASSES")
            for cls, count in self.taxonomy.items():
                L.append(f"    {cls:<28}{count:>4}")

        # -- gates ---------------------------------------------------------
        if self.gates:
            L.append("")
            L.append("  GATES")
            for g in self.gates:
                if g.passed:
                    note = f"  ({g.skipped_reason})" if g.skipped_reason else ""
                    L.append(f"    . {g.metric:<30}pass{note}")
                else:
                    L.append(
                        f"    x {g.metric:<30}"
                        f"{g.failing_language} {g.observed:.1f}  (required {g.min_absolute:.0f})"
                    )

        if self.warnings:
            L.append("")
            L.append("  WARNINGS")
            for w in self.warnings:
                L.append(f"    ! {w}")

        L.append("")
        L.append(f"  RUN {self.status.upper()}")
        return "\n".join(L)
