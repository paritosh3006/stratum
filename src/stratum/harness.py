"""The harness: run a dataset against an endpoint, score every stage."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .attribution import Cascade, build_cascade
from .dataset import Dataset, EvalItem
from .endpoint import Endpoint, RagResponse
from .judges import CalibrationRegistry, JudgeBackend
from .metrics import s2_retrieval as s2
from .metrics import s4_rendering as s4
from .report import Failure, LanguageResult, Report
from .stats import Estimate, bootstrap_mean, bootstrap_paired_difference

#: Metrics forming the outcome score — the number the cascade is computed on.
#:
#: Only properties of the *answer* appear here. This is not a stylistic choice.
#: An oracle pass repairs a stage by construction: `oracle_context` hands the
#: system the gold chunks, so retrieval recall becomes trivially perfect. If
#: recall contributed to the score, repairing retrieval would inflate its own
#: rung and the cascade would report a stage loss it manufactured itself.
#:
#: Intermediate metrics (recall, language detection) remain in the report as
#: diagnostics. They describe how a stage behaved; they do not price it.
#:
#: faithfulness/answer_correctness only ever land in an item's scores when a
#: judge is configured *and* `CalibrationRegistry.permits` the language for
#: that metric — otherwise `_composite` never sees the key and renormalises
#: over what's left, exactly like an item with no placeholders skips that
#: weight. No calibration anywhere means these two are absent everywhere,
#: which is what keeps S2/S3 "not measured" the default rather than a number
#: nobody calibrated.
OUTCOME_WEIGHTS: dict[str, float] = {
    "answered_correctly": 2.0,
    "placeholder_integrity": 1.0,
    "numeral_integrity": 1.0,
    "glossary_adherence": 1.0,
    "entity_preservation": 1.0,
    "faithfulness": 2.0,
    "answer_correctness": 2.0,
}

#: Diagnostics reported per language but excluded from the outcome score.
DIAGNOSTIC_METRICS = (
    "recall_at_k", "mrr", "ndcg_at_k", "language_detection", "over_refusal",
)

#: Stages whose loss cannot be measured by deterministic checks alone.
#:
#: Retrieval and generation influence the user-visible answer only through the
#: answer's *content*, and no deterministic check reads content — that needs a
#: judge. Until one is calibrated, these rungs are reported as not measurable
#: rather than being assigned a number the method cannot support.
JUDGE_DEPENDENT_STAGES = ("s2_retrieval", "s3_generation")


@dataclass
class ItemResult:
    item: EvalItem
    response: RagResponse
    pass_name: str = "standard"
    scores: dict[str, float | None] = field(default_factory=dict)
    composite: float | None = None
    failures: list[Failure] = field(default_factory=list)


class Harness:
    def __init__(
        self,
        endpoint: Endpoint,
        dataset: Dataset,
        *,
        baseline_language: str = "en",
        glossary: s4.Glossary | None = None,
        k: int = 5,
        judge: JudgeBackend | None = None,
        calibration: CalibrationRegistry | None = None,
        verified_languages: list[str] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.dataset = dataset
        self.baseline_language = baseline_language
        self.glossary = glossary
        self.k = k
        self.judge = judge
        self.calibration = calibration or CalibrationRegistry()
        self.verified_languages = verified_languages

    # ------------------------------------------------------------------
    def _judge_permits(self, language: str, metric: str) -> bool:
        if self.judge is None:
            return False
        return self.calibration.permits(language, metric, self.judge.judge_id)

    # ------------------------------------------------------------------
    def run(self, system_label: str = "system-under-test") -> Report:
        warnings = list(self.dataset.validate_parallelism(self.baseline_language))
        caps = getattr(self.endpoint, "capabilities", None)
        supported = caps.supported_passes if caps else ["standard"]

        if supported == ["standard"]:
            warnings.append(
                "endpoint declares no oracle-pass support — attribution unavailable; "
                "see docs/attribution.md for the three override hooks"
            )

        if self.judge is not None:
            permitted = [
                f"{lang}/{metric}"
                for lang in self.dataset.languages
                for metric in ("faithfulness", "answer_correctness")
                if self._judge_permits(lang, metric)
            ]
            if permitted:
                warnings.append(
                    f"judge {self.judge.judge_id!r} calibrated and scoring: "
                    f"{', '.join(permitted)}"
                )
            else:
                warnings.append(
                    f"judge {self.judge.judge_id!r} configured but not calibrated for "
                    f"any language/metric — S2/S3 remain not measured; run "
                    f"`stratum calibrate` first"
                )

        # -- standard pass over everything -------------------------------
        standard: list[ItemResult] = [self._run_item(i, "standard") for i in self.dataset]
        by_item: dict[str, ItemResult] = {r.item.id: r for r in standard}

        # -- oracle passes, only for items that have a baseline twin ------
        parallel = self.dataset.parallel_map()
        oracle: dict[str, dict[str, ItemResult]] = {p: {} for p in supported if p != "standard"}

        for pass_name in oracle:
            for pid, by_lang in parallel.items():
                gold = by_lang.get(self.baseline_language)
                if gold is None:
                    continue
                for lang, item in by_lang.items():
                    if lang == self.baseline_language:
                        continue
                    oracle[pass_name][item.id] = self._run_item(item, pass_name, gold=gold)

        # -- aggregate ----------------------------------------------------
        languages: list[LanguageResult] = []
        cascades: list[Cascade] = []
        baseline_items = [r for r in standard if r.item.language == self.baseline_language]
        baseline_composites = [r.composite for r in baseline_items]

        for lang in self.dataset.languages:
            lang_results = [r for r in standard if r.item.language == lang]
            lr = self._aggregate(lang, lang_results)
            lr.verified = (
                self.verified_languages is None or lang in self.verified_languages
            )
            if lang != self.baseline_language:
                # Paired on parallel_id: the same question in two languages.
                pairs_lang, pairs_base = [], []
                base_by_pid = {
                    r.item.parallel_id: r for r in baseline_items if r.item.parallel_id
                }
                for r in lang_results:
                    twin = base_by_pid.get(r.item.parallel_id or "")
                    if twin is not None:
                        pairs_lang.append(r.composite)
                        pairs_base.append(twin.composite)
                delta = bootstrap_paired_difference(pairs_lang, pairs_base, scale=100.0)
                lr.delta = delta.as_dict()
                lr.delta_vs_baseline = delta.value

                cascade = self._cascade_for(lang, parallel, by_item, oracle, supported)
                if cascade is not None:
                    cascades.append(cascade)
            languages.append(lr)

        taxonomy = Counter(f.cls for r in standard for f in r.failures)

        return Report(
            system_label=system_label,
            baseline_language=self.baseline_language,
            n_items=len(self.dataset),
            passes_run=supported,
            languages=sorted(languages, key=lambda x: x.language != self.baseline_language),
            cascades=[c.as_dict() for c in cascades],
            taxonomy=dict(taxonomy.most_common()),
            failures=[f for r in standard for f in r.failures],
            warnings=warnings,
            cascade_objects=cascades,
        )

    # ------------------------------------------------------------------
    def _cascade_for(
        self,
        language: str,
        parallel: dict[str, dict[str, EvalItem]],
        by_item: dict[str, ItemResult],
        oracle: dict[str, dict[str, ItemResult]],
        supported: list[str],
    ) -> Cascade | None:
        """Assemble paired per-item scores across the ladder for one language."""
        pids = [
            pid for pid, by_lang in parallel.items()
            if language in by_lang and self.baseline_language in by_lang
        ]
        if not pids:
            return None

        def scores_for(pass_name: str) -> list[float | None]:
            out: list[float | None] = []
            for pid in pids:
                item = parallel[pid][language]
                src = by_item if pass_name == "standard" else oracle.get(pass_name, {})
                res = src.get(item.id)
                out.append(res.composite if res else None)
            return out

        baseline_scores = [
            by_item[parallel[pid][self.baseline_language].id].composite for pid in pids
        ]

        pass_scores = {p: scores_for(p) for p in supported}

        judged = any(
            self._judge_permits(language, m)
            for m in ("faithfulness", "answer_correctness")
        )
        return build_cascade(
            language,
            self.baseline_language,
            pass_item_scores=pass_scores,
            baseline_item_scores=baseline_scores,
            supported_passes=supported,
            unmeasurable_stages=() if judged else JUDGE_DEPENDENT_STAGES,
        )

    # ------------------------------------------------------------------
    def _run_item(
        self, item: EvalItem, pass_name: str, gold: EvalItem | None = None
    ) -> ItemResult:
        """Run one item under one pass.

        Each oracle pass repairs one more stage using known-good input taken
        from the baseline-language twin.
        """
        query = item.query
        context_override = None
        answer_override = None

        if pass_name == "oracle_query" and gold is not None:
            query = gold.query                      # S0+S1 bypassed
        elif pass_name == "oracle_context" and gold is not None:
            query = gold.query
            context_override = item.gold_chunk_ids  # S2 also bypassed
        elif pass_name == "oracle_answer" and gold is not None:
            query = gold.query
            context_override = item.gold_chunk_ids
            answer_override = gold.gold_answer or gold.query  # S3 also bypassed

        resp = self.endpoint.query(
            query,
            item.language,
            context_chunk_ids=context_override,
            answer_override=answer_override,
        )
        res = ItemResult(item=item, response=resp, pass_name=pass_name)

        # -- S2 retrieval -------------------------------------------------
        res.scores["recall_at_k"] = s2.recall_at_k(
            resp.retrieved_chunk_ids, item.gold_chunk_ids, self.k
        )
        res.scores["mrr"] = s2.mrr(resp.retrieved_chunk_ids, item.gold_chunk_ids)
        res.scores["ndcg_at_k"] = s2.ndcg_at_k(
            resp.retrieved_chunk_ids, item.gold_chunk_ids, self.k
        )
        if pass_name == "standard" and item.gold_chunk_ids and res.scores["recall_at_k"] == 0.0:
            res.failures.append(self._fail(item, "retrieval_miss", "s2",
                                           "no gold chunk in top-k", resp.answer))

        # -- S0 language detection ----------------------------------------
        if resp.detected_language is not None:
            ok = resp.detected_language == item.language
            res.scores["language_detection"] = 1.0 if ok else 0.0
            if not ok and pass_name == "standard":
                res.failures.append(self._fail(
                    item, "script_misdetection", "s0",
                    f"detected {resp.detected_language}, expected {item.language}",
                    resp.answer))

        # -- S3 refusal behaviour -----------------------------------------
        if item.answerable:
            over = resp.refused
            res.scores["over_refusal"] = 1.0 if over else 0.0
            res.scores["answered_correctly"] = 0.0 if over else 1.0
            if over and pass_name == "standard":
                res.failures.append(self._fail(item, "over_refusal", "s3",
                                               "refused an answerable question", resp.answer))
        else:
            res.scores["answered_correctly"] = 1.0 if resp.refused else 0.0
            if not resp.refused and pass_name == "standard":
                res.failures.append(self._fail(item, "missed_refusal", "s3",
                                               "answered an unanswerable question", resp.answer))

        # -- S4 rendering ---------------------------------------------------
        ph = s4.check_placeholders(item.query, resp.answer, item.placeholders or None)
        if "no placeholders" not in ph.detail:
            res.scores["placeholder_integrity"] = float(ph.passed)
            if not ph.passed and pass_name == "standard":
                res.failures.append(self._fail(item, "placeholder_corruption", "s4",
                                               ph.detail, resp.answer))

        num = s4.check_numerals(item.query, resp.answer, item.numerals or None)
        if "no numerals" not in num.detail:
            res.scores["numeral_integrity"] = float(num.passed)
            if not num.passed and pass_name == "standard":
                res.failures.append(self._fail(item, "numeral_error", "s4",
                                               num.detail, resp.answer))

        if item.entities:
            ent = s4.check_entities(resp.answer, item.entities)
            res.scores["entity_preservation"] = float(ent.passed)
            if not ent.passed and pass_name == "standard":
                res.failures.append(self._fail(item, "entity_mangled", "s1",
                                               ent.detail, resp.answer))

        if self.glossary:
            gl = s4.check_glossary(item.query, resp.answer, self.glossary, item.language)
            if "no glossary" not in gl.detail:
                res.scores["glossary_adherence"] = float(gl.passed)
                if not gl.passed and pass_name == "standard":
                    res.failures.append(self._fail(item, "terminology_drift", "s4",
                                                   gl.detail, resp.answer))

        # -- S2/S3 judged content, only where calibration permits it --------
        # Runs on every pass, not just standard: the cascade computes a
        # per-pass delta, and comparing a composite that includes judged
        # content against one that doesn't would price the judge's opinion
        # as if it were a stage's loss. Skipped for refusals/empty answers
        # (nothing to judge) and when the endpoint or item doesn't supply
        # what the judge needs (retrieved_context, gold_answer).
        if resp.answer and not resp.refused:
            if resp.retrieved_context and self._judge_permits(item.language, "faithfulness"):
                j = self.judge.judge_faithfulness(resp.answer, resp.retrieved_context, item.language)
                res.scores["faithfulness"] = j.score
            if item.gold_answer and self._judge_permits(item.language, "answer_correctness"):
                j = self.judge.judge_correctness(resp.answer, item.gold_answer, item.language)
                res.scores["answer_correctness"] = j.score / 3.0  # 0-3 rubric -> 0..1 composite

        res.composite = self._composite(res.scores)
        return res

    # ------------------------------------------------------------------
    @staticmethod
    def _composite(scores: dict[str, float | None]) -> float | None:
        """Weighted mean over applicable outcome metrics, on 0..1.

        Only metrics that apply to the item participate, and the weights are
        renormalised over those present — so an item with no placeholders is
        not penalised, and is not silently scored on a different basis.
        """
        num = den = 0.0
        for metric, weight in OUTCOME_WEIGHTS.items():
            val = scores.get(metric)
            if val is None:
                continue
            num += val * weight
            den += weight
        return num / den if den else None

    @staticmethod
    def _fail(item: EvalItem, cls: str, stage: str, detail: str, output: str) -> Failure:
        return Failure(
            item_id=item.id, language=item.language, cls=cls, stage=stage,
            slice=item.slice, query=item.query, output=output[:300], detail=detail,
        )

    # ------------------------------------------------------------------
    def _aggregate(self, language: str, results: list[ItemResult]) -> LanguageResult:
        def est(key: str, scale: float = 100.0) -> Estimate:
            return bootstrap_mean([r.scores.get(key) for r in results], scale=scale)

        metrics = {
            name: est(name).as_dict()
            for name in (
                "recall_at_k", "language_detection", "over_refusal",
                "placeholder_integrity", "numeral_integrity",
                "entity_preservation", "glossary_adherence", "answered_correctly",
                "faithfulness", "answer_correctness",
            )
        }
        metrics["mrr"] = est("mrr", scale=1.0).as_dict()
        metrics["ndcg_at_k"] = est("ndcg_at_k", scale=1.0).as_dict()

        composite = bootstrap_mean([r.composite for r in results], scale=100.0)

        latencies = [r.response.latency_ms.get("end_to_end", 0.0) for r in results]
        latencies.sort()

        return LanguageResult(
            language=language,
            n_items=len(results),
            quality=composite.as_dict(),
            answer_quality=composite.value,
            metrics=metrics,
            latency_p50=round(latencies[len(latencies) // 2], 1) if latencies else None,
            latency_p90=round(latencies[int(len(latencies) * 0.9)], 1) if latencies else None,
        )
