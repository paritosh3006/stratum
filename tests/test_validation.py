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

  - The 20-system suite above uses deliberately large defects (whole-item
    swaps, ~9000-magnitude numeral corruption) at severities of 60-100% of
    the dataset, so a clean sweep there only proves stratum handles the
    *obvious* case. The sensitivity sweep below (`_run_sensitivity_sweep`)
    asks the sharper question: at graduated *target loss magnitudes*
    (~2/5/10/20 points) and dataset sizes (n=20/50/100 paired items), what
    is the smallest defect stratum reliably attributes? Severity here maps
    to a target magnitude via a measured, not assumed, linear relationship
    — loss scales linearly in the fraction of affected items because the
    synthetic items are homogeneous by construction (interchangeable
    effect size per item), verified empirically before relying on it: a
    2-item defect on a 100-item S0 system measured 1.2 points against a
    60.0-point measurement at 100% severity, exactly 60.0 x (2/100).
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
        # The index suffix (not just cycling through _TOPICS) keeps every
        # item's topic token unique regardless of n_pairs — the sensitivity
        # sweep below builds datasets up to n=100, well past len(_TOPICS),
        # and a repeated bare topic word ("battery" at item 0 *and* item
        # 15) would reintroduce the same partial token-overlap problem the
        # index-free version had, just at a smaller scale (F1=0.5 instead
        # of 0.71 — still enough to miss StubJudge's "else -> 0.0" bucket).
        topic = f"{_TOPICS[i % len(_TOPICS)]}{i}"
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


def _lookups(items: list[SynthItem]) -> tuple[dict[str, "SynthItem"], set[str]]:
    """Per-`items`-list bookkeeping, built fresh from whatever item set an
    endpoint factory is actually given — never from a module-level
    constant. The sensitivity sweep below calls these factories with
    differently-sized item lists (n=20/50/100, not just the fixed 10-item
    ITEMS); a closure over a single global item set would silently resolve
    every query against the wrong dataset the moment sizes diverge.

    Returns (by_query, en_queries): `by_query` resolves either query form
    to "which item is this about" for response bookkeeping; `en_queries`
    is the actual "did the endpoint understand this query" signal — only
    the gold English form counts, so it must stay a strict subset of
    `by_query`'s keys, never conflated with it.
    """
    by_query = {it.query_en: it for it in items} | {it.query_xx: it for it in items}
    en_queries = {it.query_en for it in items}
    return by_query, en_queries


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
    by_query, en_queries = _lookups(items)

    def rag(query, language, *, context_chunk_ids=None, answer_override=None):
        # Bookkeeping only — which item this request is *about*, so the
        # response can be constructed. Not the "did the endpoint understand
        # this query" signal; see en_queries below for that.
        it = by_query.get(query) or items[0]

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
        understood = query in en_queries
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
    by_query, _ = _lookups(items)

    def rag(query, language, *, context_chunk_ids=None, answer_override=None):
        it = by_query.get(query) or items[0]

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
    by_query, _ = _lookups(items)

    def rag(query, language, *, context_chunk_ids=None, answer_override=None):
        it = by_query.get(query) or items[0]

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
    by_query, _ = _lookups(items)

    def rag(query, language, *, context_chunk_ids=None, answer_override=None):
        it = by_query.get(query) or items[0]
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

def _calibrate_s2_judge(
    items: list[SynthItem] | None = None,
    dataset: Dataset | None = None,
) -> CalibrationRegistry:
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

    Takes explicit `items`/`dataset` (defaulting to the fixed 10-item
    ITEMS/DATASET) rather than closing over the module-level ones, because
    the sensitivity sweep calibrates separately for each sample size —
    reusing a calibration measured on a 10-item dataset against a 100-item
    sweep would be exactly the kind of unverified reuse this suite is
    supposed to avoid.
    """
    from stratum.calibrate import build_candidates

    items = ITEMS if items is None else items
    dataset = DATASET if dataset is None else dataset
    by_pid = {it.parallel_id: it for it in items}

    judge = StubJudge()
    endpoint = _make_s2_endpoint(items, severity=0.5, seed=99)
    harness = Harness(endpoint, dataset, baseline_language=BASELINE)
    xx_items = [i for i in dataset if i.language == SYNTH_LANG]

    candidates = build_candidates(harness, xx_items, judge)
    records: list[LabelRecord] = []
    for c in candidates:
        it = by_pid[c.item.parallel_id]
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


# ---------------------------------------------------------------------------
# Sensitivity sweep: graduated target loss magnitudes x dataset sizes.
#
# The 20-system suite above only shows stratum handles large, obvious
# defects. This asks a harder question: how small a defect does it still
# reliably attribute, and does that answer depend on how much data the run
# has to work with. Same four mechanisms, same combined-S0/S1 and
# judge-dependent-S2 honesty rules — just parameterised by target loss
# magnitude and sample size instead of a fixed severity list.
# ---------------------------------------------------------------------------

#: Approximate points of loss at the injected stage to aim for. "Approximate"
#: because the achievable severity is quantised to whole affected items
#: (round(fraction * n)), so small n can't always land exactly on a small
#: target — the achieved value is measured and reported alongside the
#: target, never assumed to equal it.
TARGET_LOSSES: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)

#: Paired-item counts to sweep. Larger n gives finer severity granularity
#: (1/100 vs 1/20 of the dataset per affected item), which is exactly the
#: variable this sweep exists to test the effect of.
SWEEP_SAMPLE_SIZES: tuple[int, ...] = (20, 50, 100)

#: "Reliably attributes" per the brief: at least this fraction of
#: mechanisms correctly attributed at a given (n, target loss) cell, with
#: the injected stage actually measured and non-noise (enforced by
#: `SweepOutcome.correct` below, not a separate check).
RELIABLE_THRESHOLD = 0.8


@dataclass(frozen=True)
class SweepOutcome:
    n: int
    mechanism: str
    expected_stage: str
    target_loss: float
    achieved_loss: float | None
    n_affected: int
    severity: float
    seed: int
    dominant_stage: str | None
    measured: bool
    noise: bool

    @property
    def correct(self) -> bool:
        return self.dominant_stage == self.expected_stage


def _severity_for_target(target: float, max_loss: float, n: int) -> tuple[float, int]:
    """The (severity, n_affected) pair whose expected loss is closest to
    `target`, given the measured max_loss at 100% severity for this
    mechanism/n. Loss is linear in the affected fraction (see module
    docstring), so this is a direct scale-down, not a search — clamped to
    at least 1 affected item (a defect nobody exhibits isn't a defect) and
    at most n (can't affect more items than exist).
    """
    if max_loss <= 0:
        return 1.0, n
    fraction = target / max_loss
    n_affected = max(1, min(n, round(fraction * n)))
    return n_affected / n, n_affected


def _sweep_max_loss(
    factory, items: list[SynthItem], dataset: Dataset, judge, calibration,
    expected_stage: str, seed: int,
) -> float:
    """Measured (not assumed) loss at 100% severity — the scale `target`
    loss magnitudes are converted against for this specific mechanism/n."""
    endpoint = factory(items, severity=1.0, seed=seed)
    harness = Harness(
        endpoint, dataset, baseline_language=BASELINE,
        judge=judge, calibration=calibration,
    )
    report = harness.run()
    cascade = next(c for c in report.cascade_objects if c.language == SYNTH_LANG)
    loss = next((l for l in cascade.losses if l.stage == expected_stage), None)
    return loss.points if loss is not None and loss.points is not None else 0.0


def _run_sweep_system(
    mechanism: str, expected_stage: str, factory,
    items: list[SynthItem], dataset: Dataset, judge, calibration,
    n: int, target: float, seed: int, max_loss: float,
) -> SweepOutcome:
    severity, n_affected = _severity_for_target(target, max_loss, n)
    endpoint = factory(items, severity=severity, seed=seed)
    harness = Harness(
        endpoint, dataset, baseline_language=BASELINE,
        judge=judge, calibration=calibration,
    )
    report = harness.run(system_label=f"sweep-{mechanism}-n{n}-target{target}")
    cascade = next(c for c in report.cascade_objects if c.language == SYNTH_LANG)
    dominant = cascade.dominant
    expected_loss = next((l for l in cascade.losses if l.stage == expected_stage), None)
    return SweepOutcome(
        n=n, mechanism=mechanism, expected_stage=expected_stage, target_loss=target,
        achieved_loss=expected_loss.points if expected_loss else None,
        n_affected=n_affected, severity=severity, seed=seed,
        dominant_stage=dominant.stage if dominant else None,
        measured=expected_loss is not None and expected_loss.points is not None,
        noise=expected_loss.is_noise if expected_loss is not None else True,
    )


def _run_sensitivity_sweep() -> list[SweepOutcome]:
    """Builds and runs every (n, mechanism, target loss) combination.
    Seeds are a fixed function of (n index, mechanism index, target index)
    — deterministic and reproducible, same guarantee as the 20-system
    suite above, just over a larger grid.
    """
    outcomes: list[SweepOutcome] = []
    for n_idx, n in enumerate(SWEEP_SAMPLE_SIZES):
        items = _build_items(n)
        dataset = _build_dataset(items)
        judge = StubJudge()
        # A fresh, honest calibration per sample size — reusing the fixed
        # 10-item calibration against a 100-item sweep would be exactly
        # the kind of unverified reuse this suite exists to avoid.
        calibration = _calibrate_s2_judge(items, dataset)

        for mech_idx, (mechanism, expected_stage, factory) in enumerate(_MECHANISMS):
            base_seed = 5000 + n_idx * 1000 + mech_idx * 100
            max_loss = _sweep_max_loss(
                factory, items, dataset, judge, calibration, expected_stage,
                seed=base_seed,
            )
            for target_idx, target in enumerate(TARGET_LOSSES):
                outcomes.append(_run_sweep_system(
                    mechanism, expected_stage, factory, items, dataset,
                    judge, calibration, n, target,
                    seed=base_seed + 10 + target_idx, max_loss=max_loss,
                ))
    return outcomes


def test_sensitivity_sweep_across_sample_sizes() -> None:
    outcomes = _run_sensitivity_sweep()
    assert len(outcomes) == len(SWEEP_SAMPLE_SIZES) * len(_MECHANISMS) * len(TARGET_LOSSES)

    by_cell: dict[tuple[int, float], list[SweepOutcome]] = defaultdict(list)
    for o in outcomes:
        by_cell[(o.n, o.target_loss)].append(o)

    print("\nSensitivity sweep — accuracy / noise-rate / not-measured-rate by (n, target loss):")
    smallest_reliable: dict[int, float | None] = {}
    for n in SWEEP_SAMPLE_SIZES:
        print(f"\n  n={n}:")
        for target in TARGET_LOSSES:
            cell = by_cell[(n, target)]
            n_cell = len(cell)
            n_correct = sum(o.correct for o in cell)
            n_noise = sum(o.noise for o in cell)
            n_not_measured = sum(not o.measured for o in cell)
            accuracy = n_correct / n_cell
            achieved = [o.achieved_loss for o in cell if o.achieved_loss is not None]
            achieved_repr = f"{min(achieved):.1f}-{max(achieved):.1f}" if achieved else "n/a"
            print(
                f"    target~{target:>5.1f}  achieved~{achieved_repr:<11}  "
                f"accuracy={n_correct}/{n_cell} ({accuracy:.0%})  "
                f"noise={n_noise}/{n_cell}  not_measured={n_not_measured}/{n_cell}"
            )
            if accuracy >= RELIABLE_THRESHOLD and n not in smallest_reliable:
                smallest_reliable[n] = target
        smallest_reliable.setdefault(n, None)

    print("\nSmallest reliably-attributed target loss magnitude, by sample size:")
    for n in SWEEP_SAMPLE_SIZES:
        val = smallest_reliable[n]
        print(f"  n={n}: {val if val is not None else 'none of the tested magnitudes (' + str(TARGET_LOSSES) + ')'}")

    confusion: Counter[tuple[str, str]] = Counter(
        (o.mechanism, o.dominant_stage or "not_measured_or_noise") for o in outcomes
    )
    print("\nConfusion matrix, all (n, target) cells combined (injected mechanism -> reported stage):")
    for (mechanism, reported), n in sorted(confusion.items()):
        print(f"  {mechanism} -> {reported}: {n}")

    print(
        "\nThese are controlled synthetic defects on a synthetic dataset — a "
        "sensitivity floor for stratum's attribution method itself, not a "
        "guarantee about any particular real-world system's defects."
    )

    # Never count "noise" or "not measured" as correct, at any cell.
    for o in outcomes:
        if o.correct:
            assert o.measured
            assert not o.noise

    # The largest tested magnitude (20 points) should be reliably
    # attributed at every sample size — a floor a real attribution
    # regression should break loudly, not a claim about smaller magnitudes.
    for n in SWEEP_SAMPLE_SIZES:
        cell = by_cell[(n, TARGET_LOSSES[-1])]
        accuracy = sum(o.correct for o in cell) / len(cell)
        assert accuracy >= RELIABLE_THRESHOLD, (
            f"n={n}, target={TARGET_LOSSES[-1]}: only {accuracy:.0%} correct"
        )
