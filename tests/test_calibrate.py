"""Tests for calibrate.py's pure logic — everything except the interactive
CLI shell (cli.py's `calibrate` command owns stdin, tested by inspection
above in the smoke-test rather than here)."""

import json

import pytest

from stratum import calibrate as cal
from stratum.dataset import Dataset, EvalItem
from stratum.endpoint import CallableEndpoint, Capabilities, RagResponse
from stratum.harness import Harness
from stratum.judges import CalibrationRegistry
from stratum.judges.backends import StubJudge


def _item(id_, language, parallel_id="p1", **kw) -> EvalItem:
    defaults = dict(
        slice="parallel_core", query="What is the grace period?",
        gold_answer="The grace period is 30 days.", gold_chunk_ids=["c1"],
    )
    defaults.update(kw)
    return EvalItem(id=id_, language=language, parallel_id=parallel_id, **defaults)


def _dataset() -> Dataset:
    return Dataset(items=[
        _item("en1", "en"), _item("en2", "en", parallel_id="p2",
                                   query="What is the co-payment?"),
        _item("hi1", "hi-Deva"), _item("hi2", "hi-Deva", parallel_id="p2",
                                        query="सह-भुगतान क्या है?"),
        _item("ta1", "ta-Taml", parallel_id="p3"),
    ])


def _fake_rag(query, language, *, context_chunk_ids=None, answer_override=None):
    context = ["The grace period is 30 days, allowed for premium payment."]
    retrieved = context_chunk_ids if context_chunk_ids is not None else ["c1"]
    answer = answer_override if answer_override is not None else "The grace period is 30 days."
    return RagResponse(
        answer=answer, retrieved_chunk_ids=retrieved, retrieved_context=context,
        detected_language=language,
    )


def _endpoint() -> CallableEndpoint:
    return CallableEndpoint(_fake_rag, capabilities=Capabilities(
        accepts_query_override=True, accepts_context_override=True,
        accepts_answer_override=True,
    ))


class TestSampleItems:
    def test_respects_per_language_cap(self):
        items = cal.sample_items(_dataset(), n_per_language=1, seed=0)
        by_lang = {}
        for i in items:
            by_lang.setdefault(i.language, 0)
            by_lang[i.language] += 1
        assert all(count <= 1 for count in by_lang.values())

    def test_covers_every_language_present(self):
        items = cal.sample_items(_dataset(), n_per_language=5, seed=0)
        assert {i.language for i in items} == {"en", "hi-Deva", "ta-Taml"}

    def test_deterministic_given_seed(self):
        a = cal.sample_items(_dataset(), n_per_language=1, seed=3)
        b = cal.sample_items(_dataset(), n_per_language=1, seed=3)
        assert [i.id for i in a] == [i.id for i in b]

    def test_different_seeds_can_differ(self):
        # Not guaranteed in general, but true for this fixture (2 items for
        # en/hi-Deva) often enough that a stable seed pair is fine to pin.
        a = cal.sample_items(_dataset(), n_per_language=1, seed=0)
        b = cal.sample_items(_dataset(), n_per_language=1, seed=1)
        assert [i.id for i in a] != [i.id for i in b]


class TestBuildCandidates:
    def test_scores_faithfulness_and_correctness(self):
        harness = Harness(_endpoint(), _dataset())
        items = [i for i in _dataset() if i.language == "en"]
        candidates = cal.build_candidates(harness, items, StubJudge())
        assert all(c.judge_faithfulness is not None for c in candidates)
        assert all(c.judge_correctness is not None for c in candidates)

    def test_no_gold_answer_skips_correctness_only(self):
        ds = Dataset(items=[_item("x1", "en", gold_answer=None)])
        harness = Harness(_endpoint(), ds)
        candidates = cal.build_candidates(harness, list(ds), StubJudge())
        assert candidates[0].judge_faithfulness is not None
        assert candidates[0].judge_correctness is None

    def test_refused_answer_is_not_judged(self):
        def _refusing_rag(query, language, **kw):
            return RagResponse(answer="", retrieved_chunk_ids=[], refused=True)
        ep = CallableEndpoint(_refusing_rag)
        harness = Harness(ep, _dataset())
        items = [i for i in _dataset() if i.language == "en"][:1]
        candidates = cal.build_candidates(harness, items, StubJudge())
        assert candidates[0].judge_faithfulness is None
        assert candidates[0].judge_correctness is None


class TestDiscretize:
    def test_faithfulness_scales_onto_rubric(self):
        assert cal.discretize("faithfulness", 0.0) == 0
        assert cal.discretize("faithfulness", 1.0) == 3
        assert cal.discretize("faithfulness", 0.5) == round(0.5 * 3)

    def test_correctness_rounds_and_clips(self):
        assert cal.discretize("answer_correctness", 2.4) == 2
        assert cal.discretize("answer_correctness", 2.6) == 3
        assert cal.discretize("answer_correctness", 5.0) == 3  # clipped
        assert cal.discretize("answer_correctness", -1.0) == 0  # clipped

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            cal.discretize("nonsense", 0.5)


class TestComputeCalibration:
    def test_perfect_agreement_gives_kappa_one(self):
        records = [
            cal.LabelRecord("i1", "hi-Deva", "faithfulness", 3, 1.0, "stub-token-overlap"),
            cal.LabelRecord("i2", "hi-Deva", "faithfulness", 0, 0.0, "stub-token-overlap"),
            cal.LabelRecord("i3", "hi-Deva", "faithfulness", 2, 0.6, "stub-token-overlap"),
        ]
        result = cal.compute_calibration(records)
        c = result[("hi-Deva", "faithfulness")]
        assert c.kappa == pytest.approx(1.0)
        assert c.n_labelled == 3
        assert c.judge_id == "stub-token-overlap"

    def test_groups_independently_per_language_and_metric(self):
        records = [
            cal.LabelRecord("i1", "hi-Deva", "faithfulness", 3, 1.0, "j"),
            cal.LabelRecord("i2", "hi-Latn", "faithfulness", 0, 1.0, "j"),  # disagreement
        ]
        result = cal.compute_calibration(records)
        assert set(result) == {("hi-Deva", "faithfulness"), ("hi-Latn", "faithfulness")}

    def test_threshold_is_carried_onto_the_record(self):
        records = [cal.LabelRecord("i1", "en", "faithfulness", 3, 1.0, "j")]
        c = cal.compute_calibration(records, threshold=0.9)[("en", "faithfulness")]
        assert c.threshold == 0.9

    def test_mixed_judge_ids_for_same_group_raises(self):
        records = [
            cal.LabelRecord("i1", "en", "faithfulness", 3, 1.0, "judge-a"),
            cal.LabelRecord("i2", "en", "faithfulness", 3, 1.0, "judge-b"),
        ]
        with pytest.raises(ValueError, match="multiple judges"):
            cal.compute_calibration(records)

    def test_registers_low_kappa_too_not_just_trustworthy(self):
        # compute_calibration doesn't gate — CalibrationRegistry.permits does.
        records = [
            cal.LabelRecord("i1", "en", "faithfulness", 3, 0.0, "j"),
            cal.LabelRecord("i2", "en", "faithfulness", 0, 1.0, "j"),
        ]
        c = cal.compute_calibration(records)[("en", "faithfulness")]
        assert not c.is_trustworthy
        reg = CalibrationRegistry()
        reg.register(c)
        assert reg.get("en", "faithfulness") is not None
        assert not reg.permits("en", "faithfulness")


class TestRegistryPersistence:
    def test_roundtrip_preserves_full_precision_and_threshold(self, tmp_path):
        records = [
            cal.LabelRecord("i1", "hi-Deva", "faithfulness", 2, 0.61, "j"),
            cal.LabelRecord("i2", "hi-Deva", "faithfulness", 1, 0.2, "j"),
            cal.LabelRecord("i3", "hi-Deva", "faithfulness", 3, 0.9, "j"),
        ]
        calibrations = cal.compute_calibration(records, threshold=0.42)
        registry = CalibrationRegistry()
        for c in calibrations.values():
            registry.register(c)

        path = cal.save_registry(registry, tmp_path / "calibration.json")
        loaded = cal.load_registry(path)

        original = registry.get("hi-Deva", "faithfulness")
        restored = loaded.get("hi-Deva", "faithfulness")
        assert restored == original  # frozen dataclass: field-wise equality
        assert restored.threshold == 0.42

    def test_saved_file_is_readable_json(self, tmp_path):
        records = [cal.LabelRecord("i1", "en", "faithfulness", 3, 1.0, "j")]
        registry = CalibrationRegistry()
        for c in cal.compute_calibration(records).values():
            registry.register(c)
        path = cal.save_registry(registry, tmp_path / "cal.json")
        data = json.loads(path.read_text())
        assert data["calibrations"][0]["language"] == "en"


class TestLabelRecordFile:
    def test_append_then_load_roundtrips(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        r1 = cal.LabelRecord("i1", "en", "faithfulness", 3, 1.0, "j")
        r2 = cal.LabelRecord("i2", "hi-Deva", "answer_correctness", 1, 0.3, "j")
        cal.append_label_record(path, r1)
        cal.append_label_record(path, r2)
        loaded = cal.load_label_records(path)
        assert loaded == [r1, r2]

    def test_missing_file_returns_empty_list(self, tmp_path):
        assert cal.load_label_records(tmp_path / "does_not_exist.jsonl") == []

    def test_resumable_across_separate_writes(self, tmp_path):
        # Simulates two separate `stratum calibrate` invocations.
        path = tmp_path / "labels.jsonl"
        cal.append_label_record(path, cal.LabelRecord("i1", "en", "faithfulness", 3, 1.0, "j"))
        first_session = cal.load_label_records(path)
        cal.append_label_record(path, cal.LabelRecord("i2", "en", "faithfulness", 2, 0.7, "j"))
        second_session = cal.load_label_records(path)
        assert len(first_session) == 1
        assert len(second_session) == 2
