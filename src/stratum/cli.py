"""Command line interface."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import typer

from . import calibrate as calibration_mod
from .dataset import Dataset
from .metrics.s4_rendering import Glossary
from .harness import Harness
from .judges import CalibrationRegistry, get_judge
from .report import Gate

app = typer.Typer(add_completion=False, help="Evaluate multilingual RAG, per stage.")


def _load_endpoint(spec: str):
    """Load `path/to/file.py:attr`. Returns the module too, not just the
    endpoint attribute — `run` uses it to look for a `glossary` an endpoint
    declares alongside itself, so a system whose rendering is checked
    against a specific glossary doesn't depend on the caller remembering to
    pass `--glossary` separately and keep it in sync."""
    path, _, attr = spec.partition(":")
    attr = attr or "endpoint"
    mod_spec = importlib.util.spec_from_file_location("_stratum_ep", path)
    module = importlib.util.module_from_spec(mod_spec)          # type: ignore[arg-type]
    mod_spec.loader.exec_module(module)                          # type: ignore[union-attr]
    return module, getattr(module, attr)


@app.command()
def run(
    endpoint: str = typer.Option(..., help="module.py:attr exposing an Endpoint"),
    dataset: Path = typer.Option(..., exists=True),
    glossary: Path = typer.Option(
        None, exists=True,
        help="overrides any glossary the endpoint module declares via a "
             "module-level `glossary` attribute",
    ),
    baseline: str = typer.Option("en"),
    k: int = typer.Option(5),
    judge: str = typer.Option(
        None,
        help="judge backend for S2/S3 (faithfulness/answer_correctness): "
             "stub, ollama, ollama:<model>. Has no effect without --calibration "
             "— an uncalibrated judge scores nothing, by design",
    ),
    calibration: Path = typer.Option(
        None, exists=True,
        help="calibration.json from `stratum calibrate`; without it S2/S3 stay "
             "'not measurable' even with --judge set",
    ),
    out: Path = typer.Option(None, help="directory for report.json"),
    label: str = typer.Option("system-under-test"),
    verified: str = typer.Option(
        None,
        help="comma-separated languages whose dataset and judge a native speaker "
             "has checked; all others are reported as experimental and excluded "
             "from gates",
    ),
    fail_on_unevaluated: bool = typer.Option(
        True,
        help="a gate whose metric was never observed (e.g. glossary_adherence "
             "with no --glossary loaded) fails the run instead of silently "
             "passing; disable only if that metric is deliberately not gated "
             "this run",
    ),
):
    ds = Dataset.from_jsonl(dataset)
    module, ep = _load_endpoint(endpoint)

    if glossary:
        gl = Glossary.from_dict(json.loads(glossary.read_text()))
    else:
        # Not every endpoint declares one — a system with no rendering
        # stage has nothing to check terminology against, and that's fine.
        gl = getattr(module, "glossary", None)

    jb = get_judge(judge) if judge else None
    cal_registry = (
        calibration_mod.load_registry(calibration) if calibration else CalibrationRegistry()
    )

    harness = Harness(
        endpoint=ep,
        dataset=ds,
        baseline_language=baseline,
        glossary=gl,
        k=k,
        judge=jb,
        calibration=cal_registry,
        verified_languages=[v.strip() for v in verified.split(",")] if verified else None,
    )
    report = harness.run(system_label=label)

    report.evaluate_gates([
        Gate(metric="placeholder_integrity", languages="all", min_absolute=100.0),
        Gate(metric="numeral_integrity", languages="all", min_absolute=95.0),
        Gate(metric="language_detection", languages="all", min_absolute=90.0),
        Gate(metric="glossary_adherence", languages="all", min_absolute=85.0),
    ], fail_on_unevaluated=fail_on_unevaluated)

    typer.echo(report.render_terminal())

    if out:
        path = report.save(out)
        typer.echo(f"\n  report -> {path}")

    raise typer.Exit(1 if report.status == "failed" else 0)


_RUBRIC_HINT = {
    "answer_correctness": "0=wrong  1=partial  2=mostly right  3=fully agrees with reference",
    "faithfulness": "0=unsupported  1=some support  2=mostly supported  3=fully supported by context",
}


def _prompt_score(metric: str) -> int | None:
    hint = _RUBRIC_HINT.get(metric, "0-3")
    while True:
        raw = typer.prompt(f"  your score ({hint}) — or 's' to skip")
        if raw.strip().lower() in ("s", "skip"):
            return None
        try:
            val = int(raw.strip())
        except ValueError:
            typer.echo("  enter an integer 0-3, or 's' to skip")
            continue
        if val not in (0, 1, 2, 3):
            typer.echo("  must be 0, 1, 2, or 3")
            continue
        return val


@app.command()
def calibrate(
    endpoint: str = typer.Option(..., help="module.py:attr exposing an Endpoint"),
    dataset: Path = typer.Option(..., exists=True),
    judge: str = typer.Option("stub", help="judge backend: stub, ollama, ollama:<model>"),
    sample_size: int = typer.Option(10, help="items per language to sample for labelling"),
    seed: int = typer.Option(0),
    metrics: str = typer.Option("faithfulness,answer_correctness"),
    labels: Path = typer.Option(
        Path("calibration_labels.jsonl"),
        help="hand labels are appended here as you go — rerunning the same "
             "command resumes from whatever's already labelled instead of "
             "starting over",
    ),
    out: Path = typer.Option(Path("calibration.json")),
    threshold: float = typer.Option(
        0.60, help="kappa a language/metric needs to be trusted by `stratum run`"
    ),
    interactive: bool = typer.Option(
        True,
        help="prompt for hand labels; --no-interactive only recomputes "
             "calibration.json from whatever is already in --labels",
    ),
):
    """Hand-label a judge sample, compute Cohen's kappa per language, save
    a registry `stratum run --calibration` can gate judged metrics on."""
    ds = Dataset.from_jsonl(dataset)
    _, ep = _load_endpoint(endpoint)
    jb = get_judge(judge)
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]

    already = {(r.item_id, r.metric) for r in calibration_mod.load_label_records(labels)}

    if interactive:
        items = calibration_mod.sample_items(ds, sample_size, seed=seed)
        harness = Harness(ep, ds)
        candidates = calibration_mod.build_candidates(harness, items, jb)

        def _judged_for(c: calibration_mod.LabelCandidate, metric: str):
            if metric == "faithfulness":
                return c.judge_faithfulness
            if metric == "answer_correctness":
                return c.judge_correctness
            raise ValueError(f"unknown metric: {metric!r} — expected 'faithfulness' or 'answer_correctness'")

        pending = sum(
            1 for c in candidates for m in metric_list
            if _judged_for(c, m) is not None and (c.item.id, m) not in already
        )
        typer.echo(
            f"\n  {len(candidates)} candidates sampled, {len(already)} labels "
            f"already recorded, {pending} left to label\n"
        )

        try:
            for c in candidates:
                for metric in metric_list:
                    j = _judged_for(c, metric)
                    if j is None or (c.item.id, metric) in already:
                        continue

                    typer.echo(f"  --- {c.item.id} [{c.item.language}] · {metric} ---")
                    typer.echo(f"  query:     {c.item.query}")
                    if metric == "faithfulness":
                        ctx = " / ".join(c.response.retrieved_context)
                        typer.echo(f"  context:   {ctx[:400]}")
                    else:
                        typer.echo(f"  reference: {c.item.gold_answer}")
                    typer.echo(f"  answer:    {c.response.answer}")
                    typer.echo("")

                    score = _prompt_score(metric)
                    typer.echo("")
                    if score is None:
                        continue

                    calibration_mod.append_label_record(labels, calibration_mod.LabelRecord(
                        item_id=c.item.id, language=c.item.language, metric=metric,
                        human_score=score, judge_score=j.score, judge_id=jb.judge_id,
                    ))
                    already.add((c.item.id, metric))
        except KeyboardInterrupt:
            typer.echo(
                "\n\n  paused — labels recorded so far are saved; rerun the "
                "same command to resume."
            )

    records = calibration_mod.load_label_records(labels)
    if not records:
        typer.echo("\n  no labels recorded — nothing to calibrate.")
        raise typer.Exit(1)

    calibrations = calibration_mod.compute_calibration(records, threshold=threshold)
    registry = CalibrationRegistry()
    for c in calibrations.values():
        registry.register(c)
    saved = calibration_mod.save_registry(registry, out)

    typer.echo(f"\n  {'LANGUAGE':<12}{'METRIC':<20}{'N':>5}{'KAPPA':>8}  STATUS")
    for (language, metric), c in sorted(calibrations.items()):
        status = "trustworthy" if c.is_trustworthy else "below threshold"
        typer.echo(f"  {language:<12}{metric:<20}{c.n_labelled:>5}{c.kappa:>8.3f}  {status}")

    typer.echo(f"\n  calibration -> {saved}")


@app.command()
def compare(
    before: Path = typer.Argument(..., exists=True),
    after: Path = typer.Argument(..., exists=True),
):
    """Diff two runs. Regression gating reads this, not absolute thresholds."""
    a = json.loads((before / "report.json").read_text())
    b = json.loads((after / "report.json").read_text())

    idx_a = {l["language"]: l for l in a["languages"]}
    typer.echo(f"\n  {'METRIC':<40}{'BEFORE':>9}{'AFTER':>9}{'Δ':>9}")
    for lang_b in b["languages"]:
        lang_a = idx_a.get(lang_b["language"])
        if not lang_a:
            continue
        for metric, val_b in lang_b["metrics"].items():
            val_a = lang_a["metrics"].get(metric)
            if val_a is None or val_b is None or abs(val_b - val_a) < 0.05:
                continue
            name = f"{lang_b['language']} · {metric}"
            typer.echo(f"  {name:<40}{val_a:>9.1f}{val_b:>9.1f}{val_b - val_a:>+9.1f}")
    typer.echo("")


if __name__ == "__main__":
    app()
