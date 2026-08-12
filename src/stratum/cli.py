"""Command line interface."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import typer

from .dataset import Dataset
from .metrics.s4_rendering import Glossary
from .harness import Harness
from .report import Gate

app = typer.Typer(add_completion=False, help="Evaluate multilingual RAG, per stage.")


def _load_endpoint(spec: str):
    """Load `path/to/file.py:attr`."""
    path, _, attr = spec.partition(":")
    attr = attr or "endpoint"
    mod_spec = importlib.util.spec_from_file_location("_stratum_ep", path)
    module = importlib.util.module_from_spec(mod_spec)          # type: ignore[arg-type]
    mod_spec.loader.exec_module(module)                          # type: ignore[union-attr]
    return getattr(module, attr)


@app.command()
def run(
    endpoint: str = typer.Option(..., help="module.py:attr exposing an Endpoint"),
    dataset: Path = typer.Option(..., exists=True),
    glossary: Path = typer.Option(None, exists=True),
    baseline: str = typer.Option("en"),
    k: int = typer.Option(5),
    out: Path = typer.Option(None, help="directory for report.json"),
    label: str = typer.Option("system-under-test"),
    verified: str = typer.Option(
        None,
        help="comma-separated languages whose dataset and judge a native speaker "
             "has checked; all others are reported as experimental and excluded "
             "from gates",
    ),
):
    ds = Dataset.from_jsonl(dataset)
    gl = Glossary.from_dict(json.loads(glossary.read_text())) if glossary else None

    harness = Harness(
        endpoint=_load_endpoint(endpoint),
        dataset=ds,
        baseline_language=baseline,
        glossary=gl,
        k=k,
        verified_languages=[v.strip() for v in verified.split(",")] if verified else None,
    )
    report = harness.run(system_label=label)

    report.evaluate_gates([
        Gate(metric="placeholder_integrity", languages="all", min_absolute=100.0),
        Gate(metric="numeral_integrity", languages="all", min_absolute=95.0),
        Gate(metric="language_detection", languages="all", min_absolute=90.0),
        Gate(metric="glossary_adherence", languages="all", min_absolute=85.0),
    ])

    typer.echo(report.render_terminal())

    if out:
        path = report.save(out)
        typer.echo(f"\n  report -> {path}")

    raise typer.Exit(1 if report.status == "failed" else 0)


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
