"""Calibration: hand labels vs a judge, turned into a `CalibrationRegistry`.

Deliberately split from the CLI: everything here is pure/testable without
stdin. `stratum calibrate` (cli.py) is the interactive shell around
`sample_items` -> `build_candidates` -> (human labels each candidate) ->
`compute_calibration` -> `save_registry`.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset import Dataset, EvalItem
from .endpoint import RagResponse
from .harness import Harness
from .judges import Calibration, CalibrationRegistry, JudgeBackend, Judgement
from .stats import cohens_kappa

#: The rubric every human label and every judge score is discretized onto,
#: whichever metric it started as. Fixed rather than inferred from
#: whatever categories happen to appear in a given sample, so a kappa from
#: one calibration run means the same thing as a kappa from another.
RUBRIC = (0, 1, 2, 3)


@dataclass
class LabelCandidate:
    """One sampled item with its system output and the judge's own score —
    everything a human needs to see to label it, and everything
    `compute_calibration` needs once a human label is attached."""

    item: EvalItem
    response: RagResponse
    judge_faithfulness: Judgement | None
    judge_correctness: Judgement | None


@dataclass
class LabelRecord:
    """One human label for one metric on one candidate."""

    item_id: str
    language: str
    metric: str  # "faithfulness" | "answer_correctness"
    human_score: int  # on RUBRIC
    judge_score: float  # Judgement.score as returned: 0..1 for faithfulness, 0..3 for correctness
    judge_id: str


# --------------------------------------------------------------------------
# Sampling and running
# --------------------------------------------------------------------------

def sample_items(
    dataset: Dataset, n_per_language: int, *, seed: int = 0
) -> list[EvalItem]:
    """Up to `n_per_language` items per language, deterministic given `seed`.

    Stratified rather than a flat random sample over the whole dataset —
    kappa is computed per language, so a flat sample could easily leave a
    low-resource language with too few labels to mean anything while
    English gets labelled twenty times over.
    """
    rng = random.Random(seed)
    out: list[EvalItem] = []
    for lang in dataset.languages:
        pool = dataset.by_language(lang)
        rng.shuffle(pool)
        out.extend(pool[:n_per_language])
    return out


def build_candidates(
    harness: Harness, items: list[EvalItem], judge: JudgeBackend
) -> list[LabelCandidate]:
    """Run each item's standard pass and score it with the judge, so the
    human labels the same material the judge saw.

    Calls the judge directly rather than going through
    `Harness._run_item`'s own judge-gated scoring: at calibration time
    there is no registry yet — that is what this function's output feeds
    into — so `_judge_permits` would correctly refuse to score anything,
    which is not what we want while building the very labels that will
    populate the registry.
    """
    out: list[LabelCandidate] = []
    for item in items:
        res = harness._run_item(item, "standard")
        resp = res.response
        judgeable = resp.answer and not resp.refused

        jf = None
        if judgeable and resp.retrieved_context:
            jf = judge.judge_faithfulness(resp.answer, resp.retrieved_context, item.language)

        jc = None
        if judgeable and item.gold_answer:
            jc = judge.judge_correctness(resp.answer, item.gold_answer, item.language)

        out.append(LabelCandidate(
            item=item, response=resp, judge_faithfulness=jf, judge_correctness=jc
        ))
    return out


# --------------------------------------------------------------------------
# Discretization and kappa
# --------------------------------------------------------------------------

def discretize(metric: str, score: float) -> int:
    """Map a judge's raw score onto the shared 0-3 `RUBRIC`.

    `answer_correctness` is already 0-3 by the `JudgeBackend` contract
    (base.py); rounding only guards against a real model returning
    something like 2.5. `faithfulness` is a continuous 0-1 fraction by the
    same contract, scaled onto the same 4-point scale so both metrics'
    labels use one consistent category system for kappa — see
    `stats.cohens_kappa`'s docstring for why the specific scale chosen
    doesn't bias the result either way.
    """
    if metric == "faithfulness":
        raw = round(score * 3)
    elif metric == "answer_correctness":
        raw = round(score)
    else:
        raise ValueError(f"unknown metric: {metric!r}")
    return max(RUBRIC[0], min(RUBRIC[-1], raw))


def compute_calibration(
    records: list[LabelRecord], *, threshold: float = 0.60
) -> dict[tuple[str, str], Calibration]:
    """One `Calibration` per (language, metric) actually labelled.

    Registers everything measured, trustworthy or not — gating on
    `threshold` is `CalibrationRegistry.permits`'s job, not this one. A
    report should be able to show that a language *was* calibrated and
    simply didn't clear the bar, not have that distinction erased here.
    """
    groups: dict[tuple[str, str], list[LabelRecord]] = {}
    for r in records:
        groups.setdefault((r.language, r.metric), []).append(r)

    out: dict[tuple[str, str], Calibration] = {}
    for (language, metric), recs in groups.items():
        human = [r.human_score for r in recs]
        judged = [discretize(metric, r.judge_score) for r in recs]
        judge_ids = {r.judge_id for r in recs}
        if len(judge_ids) > 1:
            raise ValueError(
                f"labels for {language}/{metric} span multiple judges "
                f"({judge_ids}) — calibrate one judge at a time"
            )
        kappa = cohens_kappa(human, judged)
        out[(language, metric)] = Calibration(
            language=language, metric=metric, kappa=kappa,
            n_labelled=len(recs), judge_id=judge_ids.pop(), threshold=threshold,
        )
    return out


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_registry(registry: CalibrationRegistry, path: str | Path) -> Path:
    """Full-fidelity serialization — deliberately not `Calibration.as_dict()`,
    which rounds kappa to 3dp and drops `threshold` for report display."""
    path = Path(path)
    payload = {
        "calibrations": [
            {
                "language": c.language, "metric": c.metric, "kappa": c.kappa,
                "n_labelled": c.n_labelled, "judge_id": c.judge_id,
                "threshold": c.threshold,
            }
            for c in registry.records.values()
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_registry(path: str | Path) -> CalibrationRegistry:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    registry = CalibrationRegistry()
    for c in data["calibrations"]:
        registry.register(Calibration(**c))
    return registry


def append_label_record(path: str | Path, record: LabelRecord) -> None:
    """One JSON line per label, so an interrupted labelling session loses
    at most the label in progress, not the ones already entered."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_label_records(path: str | Path) -> list[LabelRecord]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(LabelRecord(**json.loads(line)))
    return records
