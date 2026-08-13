"""Validates stratum's attribution accuracy against itself.

Every other test in this repo checks that stratum computes its documented
formulas correctly. None of them ask the harder question: when a real
system has a real defect at a real stage, does the cascade actually *name
that stage*? A tool whose entire premise is "not just that quality dropped,
but where" needs to be checked against ground truth it can't see any other
way — which is what synthetic systems with a defect at a stage the test
author controls are for.

Design, end to end:

  - One fixed synthetic dataset (`_build_dataset`), en baseline + one
    synthetic non-English language ("xx-Synth" — deliberately not a real
    language code, so nobody mistakes a validation fixture for a claim
    about real-language quality). Every item's gold answer carries a
    unique 4-digit numeral, so a wrong-item answer is deterministically
    detectable by `numeral_integrity` without needing a judge.

  - `_make_endpoint(dataset, defect, severity)` builds one scripted
    CallableEndpoint per synthetic system. It honours all three oracle
    hooks correctly *except* at the one injected stage, so the ladder's
    counterfactual passes recover cleanly everywhere but there — the same
    contract docs/attribution.md asks of a real endpoint.

  - S0 and S1 get separate injection mechanisms (language misroute vs.
    degraded-translation) for narrative honesty about *why* a system might
    fail, but stratum's ladder only ever isolates a combined
    "s0_s1_input_query" rung — see attribution.py's LADDER. Both are
    graded against that combined stage, not against a nonexistent S0-only
    or S1-only rung. Getting this "wrong" would mean the suite is grading
    stratum against a resolution it doesn't have and never claimed to.

  - S2 is hardcoded judge-dependent (`harness.JUDGE_DEPENDENT_STAGES`)
    regardless of how obviously a synthetic defect would show up
    deterministically. Tested twice: once with a real calibration (kappa
    computed via the real compute_calibration/cohens_kappa pipeline
    against ground-truth-derived labels — synthetic labels are legitimate
    here because the "human" judgement being simulated is just "is this
    the item's own answer or a different item's", which is verifiable by
    construction, not a stand-in for real human review), and once without,
    to confirm the honest not-measured fallback rather than a guess.

  - S3 is not defect-injected at all (out of scope per the brief) but its
    same judge-dependency is noted in the reported limitations, since a
    reader of the accuracy report needs to know it exists.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from stratum import CallableEndpoint, Dataset, EvalItem, Harness, RagResponse
from stratum.calibrate import LabelRecord, compute_calibration
from stratum.endpoint import Capabilities
from stratum.judges import CalibrationRegistry
from stratum.judges.backends import StubJudge

BASELINE = "en"
SYNTH_LANG = "xx-Synth"
N_PAIRS = 10

#: Stage keys as attribution.py's LADDER actually names them — the only
#: vocabulary stratum's cascade can ever report.
S0_S1 = "s0_s1_input_query"
S2 = "s2_retrieval"
S4 = "s4_rendering"


# ---------------------------------------------------------------------------
# Fixed synthetic dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SynthItem:
    parallel_id: str
    index: int
    query_en: str
    query_xx: str
    chunk_id: str
    numeral: str
    answer: str
    context: str


#: Distinct topic words, one per item — not just a shared template with a
#: different number filled in. A judge (or a human) scoring by lexical/
#: token overlap cannot tell one item's answer from another's if the only
#: thing that differs is a digit inside otherwise-identical boilerplate;
#: the first version of this dataset used exactly that template and
#: StubJudge's calibration kappa came back near zero — not because the
#: judge is broken, but because the fixture gave it nothing to discriminate
#: on. Real answers about unrelated records don't share ten of eleven
#: words either.
_TOPICS = [
    "battery", "voltage", "pressure", "temperature", "humidity",
    "altitude", "velocity", "torque", "frequency", "amplitude",
    "viscosity", "resistance", "capacitance", "density", "salinity",
]


def _build_items(n_pairs: int = N_PAIRS) -> list[SynthItem]:
    items = []
    for i in range(n_pairs):
        numeral = str(1000 + i)  # 4-digit, no accidental substring overlap
        topic = _TOPICS[i % len(_TOPICS)]
        items.append(SynthItem(
            parallel_id=f"p{i:02d}",
            index=i,
            query_en=f"What is the recorded {topic} reading?",
            # Deliberately not real language text — this is a validation
            # fixture, not a claim about any real language's behaviour.
            # Different from query_en (required: the S0/S1 defect check is
            # literally "did the query arrive as the gold English text").
            query_xx=f"xx-{topic}-reading kaunsa hai?",
            chunk_id=f"chunk-{i:02d}",
            numeral=numeral,
            # Terse by design, not just for style: a wrong-item swap has to
            # read as *wrong* to a token-overlap judge, not just to a human.
            # "The recorded {topic} reading is {numeral} units." shares 5 of
            # 7 tokens with every other item's answer (same boilerplate,
            # different topic/numeral) — a completely wrong swap still
            # scored token F1=0.71 against the true reference, which lands
            # in StubJudge's ">=0.5 -> 2.0" bucket instead of "else -> 0.0".
            # Two content tokens and nothing else means a wrong swap shares
            # zero tokens with the reference, by construction.
            answer=f"{topic}: {numeral}",
            context=f"Sensor log entry — {topic}: {numeral}",
        ))
    return items


def _build_dataset(items: list[SynthItem]) -> Dataset:
    eval_items: list[EvalItem] = []
    for it in items:
        eval_items.append(EvalItem(
            id=f"en-{it.index:02d}", language=BASELINE, slice="parallel_core",
            parallel_id=it.parallel_id, query=it.query_en, gold_answer=it.answer,
            gold_chunk_ids=[it.chunk_id], numerals=[it.numeral], synthetic=True,
        ))
        eval_items.append(EvalItem(
            id=f"xx-{it.index:02d}", language=SYNTH_LANG, slice="parallel_core",
            parallel_id=it.parallel_id, query=it.query_xx, gold_answer=it.answer,
            gold_chunk_ids=[it.chunk_id], numerals=[it.numeral], synthetic=True,
        ))
    return Dataset(items=eval_items)


ITEMS = _build_items()
DATASET = _build_dataset(ITEMS)
BY_INDEX = {it.index: it for it in ITEMS}
#: For the test harness's own bookkeeping only — "which item is this
#: request about" — resolvable from either query form. NOT the "did the
#: endpoint understand this query" signal: that must come from comparing
#: against `EN_QUERIES` specifically, since a lookup that already contains
#: the xx-language form would make "understood" trivially true and defeat
#: the entire point of the S0/S1 defect check.
BY_QUERY = {it.query_en: it for it in ITEMS} | {it.query_xx: it for it in ITEMS}
EN_QUERIES = {it.query_en for it in ITEMS}


def _affected_parallel_ids(items: list[SynthItem], severity: float, seed: int) -> set[str]:
    """Deterministically pick which items actually exhibit the defect —
    `severity` a system with a milder version of the same bug, not a
    different bug. Fixed seed per call site, so re-running the suite
    reproduces the exact same 20 systems byte-for-byte."""
    import random
    ids = [it.parallel_id for it in items]
    random.Random(seed).shuffle(ids)
    n = round(severity * len(ids))
    return set(ids[:n])


def _capabilities() -> Capabilities:
    return Capabilities(
        accepts_query_override=True, accepts_context_override=True,
        accepts_answer_override=True,
    )


def _wrong_numeral(numeral: str) -> str:
    return str(int(numeral) + 9000)  # nowhere near any real item's numeral


# ---------------------------------------------------------------------------
# Defect: S0 — language misroute (query never resolves to the right item)
# ---------------------------------------------------------------------------

def _make_s0_endpoint(items: list[SynthItem], severity: float, seed: int) -> CallableEndpoint:
    affected = _affected_parallel_ids(items, severity, seed)
    by_chunk = {it.chunk_id: it for it in items}

    def rag(query, language, *, context_chunk_ids=None, answer_override=None):
        # Bookkeeping only — which item this request is *about*, so the
        # response can be constructed. Not the "did the endpoint understand
        # this query" signal; see EN_QUERIES below for that.
        it = BY_QUERY.get(query) or items[0]

        if context_chunk_ids is not None:
            target = by_chunk.get(context_chunk_ids[0], it)
            return RagResponse(
                answer=answer_override or target.answer,
                retrieved_chunk_ids=[target.chunk_id],
                retrieved_context=[target.context],
                detected_language=language,
            )

        # The actual simulated S0 failure: only the gold English query
        # text counts as "correctly identified/normalised". The item's own
        # xx-language query — real input, in the standard pass — does not,
        # by construction, regardless of whether a lookup could resolve it
        # for bookkeeping purposes.
        understood = query in EN_QUERIES
        defective = it.parallel_id in affected

        if language == BASELINE or not defective or understood:
            return RagResponse(
                answer=it.answer, retrieved_chunk_ids=[it.chunk_id],
                retrieved_context=[it.context], detected_language=language,
            )

        # Misrouted: treated as a different topic entirely, not a
        # near-miss — the record diametrically opposite in the set, so it
        # never coincides with `it` itself.
        wrong = items[(it.index + len(items) // 2) % len(items)]
        return RagResponse(
            answer=wrong.answer, retrieved_chunk_ids=[wrong.chunk_id],
            retrieved_context=[wrong.context], detected_language=language,
        )

    return CallableEndpoint(rag, capabilities=_capabilities())


# ---------------------------------------------------------------------------
# Defect: S1 — degraded translation (retrieval coarse-matches, detail lost)
# ---------------------------------------------------------------------------

def _make_s1_endpoint(items: list[SynthItem], severity: float, seed: int) -> CallableEndpoint:
    affected = _affected_parallel_ids(items, severity, seed)
    by_chunk = {it.chunk_id: it for it in items}

    def rag(query, language, *, context_chunk_ids=None, answer_override=None):
        it = BY_QUERY.get(query) or items[0]

        if context_chunk_ids is not None:
            target = by_chunk.get(context_chunk_ids[0], it)
            return RagResponse(
                answer=answer_override or target.answer,
                retrieved_chunk_ids=[target.chunk_id],
                retrieved_context=[target.context], detected_language=language,
            )

        understood = query == it.query_en
        defective = it.parallel_id in affected
        if language == BASELINE or not defective or understood:
            return RagResponse(
                answer=it.answer, retrieved_chunk_ids=[it.chunk_id],
                retrieved_context=[it.context], detected_language=language,
            )

        # Retrieval still lands on the right *record* (coarse lexical
        # match survives a rough translation) but the query's specific
        # detail — the numeral being asked about — was lost, so the
        # answer it renders substitutes the wrong value. This is the
        # "half-broken" case S0's complete-misroute doesn't cover, and
        # exactly why S1 (query processing) is its own conceptual stage
        # even though stratum's ladder can't isolate it from S0.
        degraded = it.answer.replace(it.numeral, _wrong_numeral(it.numeral))
        return RagResponse(
            answer=degraded, retrieved_chunk_ids=[it.chunk_id],
            retrieved_context=[it.context], detected_language=language,
        )

    return CallableEndpoint(rag, capabilities=_capabilities())


# ---------------------------------------------------------------------------
# Defect: S2 — retrieval only (query is understood; the wrong chunk comes back)
# ---------------------------------------------------------------------------

def _make_s2_endpoint(items: list[SynthItem], severity: float, seed: int) -> CallableEndpoint:
    affected = _affected_parallel_ids(items, severity, seed)
    by_chunk = {it.chunk_id: it for it in items}

    def rag(query, language, *, context_chunk_ids=None, answer_override=None):
        it = BY_QUERY.get(query) or items[0]

        if context_chunk_ids is not None:
            # oracle_context+ hands the gold chunk directly — retrieval is
            # bypassed entirely, so the defect (which only lives in the
            # retrieval step itself) cannot fire here by construction.
            target = by_chunk.get(context_chunk_ids[0], it)
            return RagResponse(
                answer=answer_override or target.answer,
                retrieved_chunk_ids=[target.chunk_id],
                retrieved_context=[target.context], detected_language=language,
            )

        defective = it.parallel_id in affected
        if language == BASELINE or not defective:
            return RagResponse(
                answer=it.answer, retrieved_chunk_ids=[it.chunk_id],
                retrieved_context=[it.context], detected_language=language,
            )

        # The query itself is understood fine (oracle_query alone doesn't
        # fix this — matches the real contract: S0+S1 repaired, S2 still
        # broken) but the retrieval step returns a neighbouring record's
        # chunk regardless of what was asked.
        wrong = items[(it.index + 1) % len(items)]
        return RagResponse(
            answer=wrong.answer, retrieved_chunk_ids=[wrong.chunk_id],
            retrieved_context=[wrong.context], detected_language=language,
        )

    return CallableEndpoint(rag, capabilities=_capabilities())


# ---------------------------------------------------------------------------
# Defect: S4 — rendering only (right content in, corrupted numeral out)
# ---------------------------------------------------------------------------

def _make_s4_endpoint(items: list[SynthItem], severity: float, seed: int) -> CallableEndpoint:
    affected = _affected_parallel_ids(items, severity, seed)
    by_chunk = {it.chunk_id: it for it in items}

    def rag(query, language, *, context_chunk_ids=None, answer_override=None):
        it = BY_QUERY.get(query) or items[0]
        if context_chunk_ids is not None:
            it = by_chunk.get(context_chunk_ids[0], it)

        raw = answer_override if answer_override is not None else it.answer
        defective = it.parallel_id in affected

        if language == BASELINE or not defective:
            return RagResponse(
                answer=raw, retrieved_chunk_ids=[it.chunk_id],
                retrieved_context=[it.context], detected_language=language,
            )

        # Rendering corrupts whatever content arrives — including an
        # oracle_answer override, which is the point: S4 is the one stage
        # oracle_answer does *not* repair, so this must still fire there.
        corrupted = raw.replace(it.numeral, _wrong_numeral(it.numeral))
        return RagResponse(
            answer=corrupted, retrieved_chunk_ids=[it.chunk_id],
            retrieved_context=[it.context], detected_language=language,
        )

    return CallableEndpoint(rag, capabilities=_capabilities())


# ---------------------------------------------------------------------------
# Judge calibration for S2 — required because JUDGE_DEPENDENT_STAGES masks
# s2_retrieval/s3_generation unconditionally, regardless of how obviously a
# synthetic defect would show up deterministically (see module docstring).
# ---------------------------------------------------------------------------

BY_PID = {it.parallel_id: it for it in ITEMS}


def _calibrate_s2_judge() -> CalibrationRegistry:
    """Real calibration via the real pipeline (sample -> judge -> kappa),
    not an asserted trust score. Uses a mid-severity S2 endpoint
    specifically so the label set has genuine right/wrong variance — an
    all-defective sample would make every ground-truth label "wrong" and
    the resulting kappa a degenerate edge case, not a real measurement.

    Ground truth is mechanical: "did this response literally return the
    record's own answer, or a different record's" — verifiable by
    construction from the synthetic data, which is exactly what makes it a
    legitimate substitute for hand-labelling *here* and nowhere else. Real
    calibration against a real system needs a real human; see
    calibrate.py / `stratum calibrate`.
    """
    from stratum.calibrate import build_candidates

    judge = StubJudge()
    endpoint = _make_s2_endpoint(ITEMS, severity=0.5, seed=99)
    harness = Harness(endpoint, DATASET, baseline_language=BASELINE)
    xx_items = [i for i in DATASET if i.language == SYNTH_LANG]

    candidates = build_candidates(harness, xx_items, judge)
    records: list[LabelRecord] = []
    for c in candidates:
        it = BY_PID[c.item.parallel_id]
        human = 3 if c.response.answer.strip() == it.answer.strip() else 0
        if c.judge_faithfulness is not None:
            records.append(LabelRecord(
                item_id=c.item.id, language=c.item.language, metric="faithfulness",
                human_score=human, judge_score=c.judge_faithfulness.score,
                judge_id=judge.judge_id,
            ))
        if c.judge_correctness is not None:
            records.append(LabelRecord(
                item_id=c.item.id, language=c.item.language, metric="answer_correctness",
                human_score=human, judge_score=c.judge_correctness.score,
                judge_id=judge.judge_id,
            ))

    registry = CalibrationRegistry()
    for cal in compute_calibration(records, threshold=0.60).values():
        registry.register(cal)
    return registry


# ---------------------------------------------------------------------------
# The 20 synthetic systems: 4 defect mechanisms x 5 severities.
#
# S0 and S1 are two different mechanisms graded against the same expected
# stage (S0_S1 — see module docstring); S2 and S4 each get their own. Seeds
# are a fixed function of (mechanism, severity index), not randomly chosen
# at test time, so re-running this file reproduces the exact same 20
# systems — which items are affected, in which order — byte for byte.
# ---------------------------------------------------------------------------

SEVERITIES: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6)

#: (mechanism name, expected reported stage, endpoint factory)
_MECHANISMS = [
    ("s0", S0_S1, _make_s0_endpoint),
    ("s1", S0_S1, _make_s1_endpoint),
    ("s2", S2, _make_s2_endpoint),
    ("s4", S4, _make_s4_endpoint),
]


@dataclass(frozen=True)
class SystemSpec:
    mechanism: str
    expected_stage: str
    severity: float
    seed: int
    factory: object


def _build_systems() -> list[SystemSpec]:
    systems = []
    for mech_idx, (mechanism, expected_stage, factory) in enumerate(_MECHANISMS):
        for sev_idx, severity in enumerate(SEVERITIES):
            systems.append(SystemSpec(
                mechanism=mechanism, expected_stage=expected_stage,
                severity=severity, seed=1000 + mech_idx * 100 + sev_idx,
                factory=factory,
            ))
    return systems


SYSTEMS = _build_systems()


@dataclass(frozen=True)
class SystemOutcome:
    mechanism: str
    expected_stage: str
    severity: float
    seed: int
    dominant_stage: str | None
    expected_stage_measured: bool
    expected_stage_noise: bool

    @property
    def correct(self) -> bool:
        # The rule the brief calls out explicitly: a "not measured" or
        # "noise" result is never counted as correct just because no other
        # stage outranked it. Only an actually-measured, non-noise dominant
        # stage that names the injected one counts.
        return self.dominant_stage == self.expected_stage


def _run_system(spec: SystemSpec, *, judge=None, calibration=None) -> SystemOutcome:
    endpoint = spec.factory(ITEMS, severity=spec.severity, seed=spec.seed)
    harness = Harness(
        endpoint, DATASET, baseline_language=BASELINE,
        judge=judge, calibration=calibration,
    )
    report = harness.run(system_label=f"synthetic-{spec.mechanism}-{spec.severity}")
    cascade = next(c for c in report.cascade_objects if c.language == SYNTH_LANG)
    dominant = cascade.dominant
    expected_loss = next(
        (l for l in cascade.losses if l.stage == spec.expected_stage), None
    )
    return SystemOutcome(
        mechanism=spec.mechanism,
        expected_stage=spec.expected_stage,
        severity=spec.severity,
        seed=spec.seed,
        dominant_stage=dominant.stage if dominant else None,
        expected_stage_measured=expected_loss is not None and expected_loss.points is not None,
        expected_stage_noise=expected_loss.is_noise if expected_loss else True,
    )


# ---------------------------------------------------------------------------
# Expected limitations — stated up front, not discovered by reading output.
# ---------------------------------------------------------------------------

EXPECTED_LIMITATIONS = """\
Documented, not incidental:

  - S0 and S1 are graded together as "s0_s1_input_query": stratum's ladder
    has no oracle pass that repairs one without the other, so it cannot
    report which of the two actually failed — only that the combined
    input/query-processing band did. Both injection mechanisms above are
    scored against that combined key, never against a nonexistent
    S0-only or S1-only rung.

  - S2 (retrieval) and S3 (generation) are unconditionally unmeasurable
    without a calibrated judge (harness.JUDGE_DEPENDENT_STAGES), even when
    a deterministic check would incidentally catch the defect. This suite
    calibrates a real judge (StubJudge, via the production
    build_candidates/compute_calibration/cohens_kappa pipeline) for
    answer_correctness on xx-Synth and uses that calibration for the S2
    mechanism above. faithfulness does NOT calibrate for S2 specifically
    — conceptually, not as a bug: a wrong-chunk answer is still faithful
    to the (wrong) chunk it was generated from, so faithfulness cannot
    see a retrieval defect. answer_correctness, scored against the true
    external reference regardless of what context was supplied, is the
    metric that actually detects it.

  - S3 is not defect-injected in this suite at all (out of scope per the
    brief's stage list) but shares S2's judge dependency — a reader
    should not infer S3 is validated here just because S2 is.
"""


def test_attribution_accuracy_on_20_synthetic_systems() -> None:
    assert len(SYSTEMS) == 20

    judge = StubJudge()
    calibration = _calibrate_s2_judge()
    assert calibration.permits(SYNTH_LANG, "answer_correctness", judge.judge_id), (
        "S2 calibration must actually clear the trustworthy threshold, or this "
        "test would be silently grading S2 against a judge that isn't permitted "
        "to score anything — see _calibrate_s2_judge's docstring."
    )

    outcomes = [_run_system(spec, judge=judge, calibration=calibration) for spec in SYSTEMS]

    correct = sum(o.correct for o in outcomes)
    total = len(outcomes)

    by_stage: dict[str, list[SystemOutcome]] = defaultdict(list)
    for o in outcomes:
        by_stage[o.expected_stage].append(o)
    accuracy_by_stage = {
        stage: sum(o.correct for o in items) / len(items)
        for stage, items in by_stage.items()
    }

    confusion: Counter[tuple[str, str]] = Counter(
        (o.mechanism, o.dominant_stage or "not_measured_or_noise") for o in outcomes
    )

    noise_or_not_measured = [
        o for o in outcomes if not o.expected_stage_measured or o.expected_stage_noise
    ]

    print(f"\nAttribution accuracy: {correct}/{total} ({correct / total:.0%})")
    print("\nBy expected stage:")
    for stage in sorted(accuracy_by_stage):
        items = by_stage[stage]
        n_correct = sum(o.correct for o in items)
        print(f"  {stage}: {n_correct}/{len(items)} ({n_correct / len(items):.0%})")
    print("\nConfusion matrix (injected mechanism -> reported dominant stage):")
    for (mechanism, reported), n in sorted(confusion.items()):
        print(f"  {mechanism} -> {reported}: {n}")
    if noise_or_not_measured:
        print("\nNoise / not-measured cases (excluded from 'correct', reported not hidden):")
        for o in noise_or_not_measured:
            print(
                f"  {o.mechanism} severity={o.severity} seed={o.seed}: "
                f"measured={o.expected_stage_measured} noise={o.expected_stage_noise} "
                f"dominant={o.dominant_stage}"
            )
    else:
        print("\nNoise / not-measured cases: none")
    print(f"\n{EXPECTED_LIMITATIONS}")

    # Never count "not measured" or "noise" as correct by accident: a
    # correct outcome must have an actually-measured, non-noise dominant
    # stage that names the injected one.
    for o in outcomes:
        if o.correct:
            assert o.dominant_stage is not None
            assert o.expected_stage_measured
            assert not o.expected_stage_noise

    # Defects are deliberately large (full wrong-item swaps; numeral
    # corruption ~9x any real value) so attribution should get the large
    # majority right. This is a floor, not an assertion of the exact
    # measured number — a real regression in attribution should fail this
    # loudly rather than the test silently tracking whatever it measures.
    assert correct / total >= 0.8, (
        f"only {correct}/{total} correct — confusion matrix: {dict(confusion)}"
    )


def test_s2_without_judge_calibration_is_expected_not_measured() -> None:
    """S2 is intentionally unmeasurable without a calibrated judge — the one
    case where "not measured" is the *correct*, expected outcome rather than
    a miss. Confirms the honest fallback, not a guess at S2 from a
    deterministic check that happens to see the defect.
    """
    endpoint = _make_s2_endpoint(ITEMS, severity=1.0, seed=1200)
    harness = Harness(endpoint, DATASET, baseline_language=BASELINE)  # no judge
    report = harness.run()
    cascade = next(c for c in report.cascade_objects if c.language == SYNTH_LANG)

    s2_loss = next(l for l in cascade.losses if l.stage == S2)
    s3_loss = next(l for l in cascade.losses if l.stage == "s3_generation")
    assert s2_loss.points is None
    assert s3_loss.points is None
    assert any("not measurable without a calibrated judge" in w or "judge" in w.lower()
               for w in cascade.warnings + report.warnings)

    # Dominant must never quietly become S2 when S2 cannot be measured —
    # either a genuinely measured stage takes it, or nothing does.
    assert cascade.dominant is None or cascade.dominant.stage != S2
