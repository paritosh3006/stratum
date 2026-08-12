from pathlib import Path

from stratum import Dataset, Harness

ROOT = Path(__file__).resolve().parents[1]


def _endpoint():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mock", ROOT / "examples/mock_endpoint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.endpoint


def _report(verified=None):
    import json
    from stratum.metrics.s4_rendering import Glossary
    ds = Dataset.from_jsonl(ROOT / "datasets/demo.jsonl")
    gl = Glossary.from_dict(json.loads((ROOT / "datasets/glossary.json").read_text()))
    return Harness(_endpoint(), ds, glossary=gl, verified_languages=verified).run()


def test_dataset_loads():
    ds = Dataset.from_jsonl(ROOT / "datasets/demo.jsonl")
    assert len(ds) == 100
    assert "ta-Taml" in ds.languages


def test_finds_every_planted_bug():
    # The mock endpoint has five known bugs. All five must be detected.
    tax = _report().taxonomy
    for cls in ("placeholder_corruption", "terminology_drift",
                "script_misdetection", "numeral_error", "over_refusal"):
        assert tax.get(cls, 0) > 0, f"missed planted bug: {cls}"


def test_baseline_has_no_delta():
    r = _report()
    en = next(l for l in r.languages if l.language == "en")
    assert en.delta_vs_baseline in (0.0, None)


def test_tamil_flagged_over_target():
    r = _report()
    ta = next(l for l in r.languages if l.language == "ta-Taml")
    assert ta.delta_vs_baseline < -5.0


def test_report_roundtrips_json(tmp_path):
    import json
    p = _report().save(tmp_path)
    data = json.loads(p.read_text())
    assert data["schema_version"] == "2.0"
    assert len(data["languages"]) == 4


def test_cascade_computed_for_each_non_baseline_language():
    r = _report()
    langs = {c["language"] for c in r.cascades}
    assert langs == {"hi-Deva", "hi-Latn", "ta-Taml"}


def test_rendering_dominates_for_tamil():
    # Every planted Tamil bug is an output-rendering bug, so the cascade must
    # point there. This is the property the whole tool exists to deliver.
    r = _report()
    ta = next(c for c in r.cascades if c["language"] == "ta-Taml")
    assert ta["dominant_stage"] == "s4_rendering"


def test_every_metric_carries_uncertainty():
    r = _report()
    for lr in r.languages:
        for name, m in lr.metrics.items():
            assert "n" in m and "ci95" in m, name


def test_experimental_language_excluded_from_gates():
    from stratum import Gate
    r = _report(verified=["en", "hi-Deva", "hi-Latn"])
    r.evaluate_gates([Gate(metric="placeholder_integrity", min_absolute=100.0)])
    gate = r.gates[0]
    assert gate.skipped_reason is not None


# --------------------------------------------------------------------------
# Judge wiring: the mock endpoint above has no retrieved_context and its
# gold_answer is the question restated, neither suited to judging — a
# small self-contained fixture stands in for the S2/S3 judged-metric tests.
# --------------------------------------------------------------------------

def _judge_dataset():
    from stratum.dataset import EvalItem

    def item(id_, language, parallel_id="p1"):
        return EvalItem(
            id=id_, language=language, slice="parallel_core", parallel_id=parallel_id,
            query="What is the grace period?",
            gold_answer="The grace period is 30 days.", gold_chunk_ids=["c1"],
        )

    return Dataset(items=[item("en1", "en"), item("hi1", "hi-Deva")])


def _judge_endpoint():
    from stratum.endpoint import CallableEndpoint, Capabilities, RagResponse

    def rag(query, language, *, context_chunk_ids=None, answer_override=None):
        context = ["The grace period is 30 days, allowed for premium payment."]
        retrieved = context_chunk_ids if context_chunk_ids is not None else ["c1"]
        answer = answer_override if answer_override is not None else "The grace period is 30 days."
        return RagResponse(
            answer=answer, retrieved_chunk_ids=retrieved, retrieved_context=context,
            detected_language=language,
        )

    return CallableEndpoint(rag, capabilities=Capabilities(
        accepts_query_override=True, accepts_context_override=True,
        accepts_answer_override=True,
    ))


def test_no_judge_leaves_judged_scores_unset():
    h = Harness(_judge_endpoint(), _judge_dataset())
    item = list(_judge_dataset())[0]
    res = h._run_item(item, "standard")
    assert "faithfulness" not in res.scores
    assert "answer_correctness" not in res.scores


def test_judge_without_calibration_leaves_judged_scores_unset():
    from stratum.judges.backends import StubJudge
    h = Harness(_judge_endpoint(), _judge_dataset(), judge=StubJudge())
    item = list(_judge_dataset())[0]
    res = h._run_item(item, "standard")
    assert "faithfulness" not in res.scores
    assert "answer_correctness" not in res.scores


def test_judge_with_trustworthy_calibration_populates_judged_scores():
    from stratum.judges import Calibration, CalibrationRegistry
    from stratum.judges.backends import StubJudge

    reg = CalibrationRegistry()
    for metric in ("faithfulness", "answer_correctness"):
        reg.register(Calibration(
            language="hi-Deva", metric=metric, kappa=0.8, n_labelled=10,
            judge_id="stub-token-overlap",
        ))

    h = Harness(_judge_endpoint(), _judge_dataset(), judge=StubJudge(), calibration=reg)
    hi_item = next(i for i in _judge_dataset() if i.language == "hi-Deva")
    res = h._run_item(hi_item, "standard")
    assert "faithfulness" in res.scores
    assert "answer_correctness" in res.scores
    # answer_correctness is stored on the 0..1 composite scale, not the 0-3 rubric.
    assert 0.0 <= res.scores["answer_correctness"] <= 1.0


def test_calibration_for_a_different_judge_id_does_not_apply():
    from stratum.judges import Calibration, CalibrationRegistry
    from stratum.judges.backends import StubJudge

    reg = CalibrationRegistry()
    reg.register(Calibration(
        language="hi-Deva", metric="faithfulness", kappa=0.9, n_labelled=10,
        judge_id="ollama:qwen2.5:7b",  # calibrated for a different judge
    ))

    h = Harness(_judge_endpoint(), _judge_dataset(), judge=StubJudge(), calibration=reg)
    hi_item = next(i for i in _judge_dataset() if i.language == "hi-Deva")
    res = h._run_item(hi_item, "standard")
    assert "faithfulness" not in res.scores


def test_cascade_s2_s3_unmeasured_without_calibration():
    h = Harness(_judge_endpoint(), _judge_dataset())
    r = h.run()
    cascade = r.cascades[0]
    s2 = next(s for s in cascade["by_stage"] if s["stage"] == "s2_retrieval")
    assert s2["points_lost"] is None


def test_cascade_s2_s3_measured_with_calibration():
    from stratum.judges import Calibration, CalibrationRegistry
    from stratum.judges.backends import StubJudge

    reg = CalibrationRegistry()
    for metric in ("faithfulness", "answer_correctness"):
        reg.register(Calibration(
            language="hi-Deva", metric=metric, kappa=0.8, n_labelled=10,
            judge_id="stub-token-overlap",
        ))
    h = Harness(_judge_endpoint(), _judge_dataset(), judge=StubJudge(), calibration=reg)
    r = h.run()
    cascade = r.cascades[0]
    s2 = next(s for s in cascade["by_stage"] if s["stage"] == "s2_retrieval")
    assert s2["points_lost"] is not None


def test_run_warns_when_judge_configured_but_uncalibrated():
    from stratum.judges.backends import StubJudge
    h = Harness(_judge_endpoint(), _judge_dataset(), judge=StubJudge())
    r = h.run()
    assert any("not calibrated" in w for w in r.warnings)


def test_run_warns_which_language_metric_pairs_are_scoring():
    from stratum.judges import Calibration, CalibrationRegistry
    from stratum.judges.backends import StubJudge

    reg = CalibrationRegistry()
    reg.register(Calibration(
        language="hi-Deva", metric="faithfulness", kappa=0.8, n_labelled=10,
        judge_id="stub-token-overlap",
    ))
    h = Harness(_judge_endpoint(), _judge_dataset(), judge=StubJudge(), calibration=reg)
    r = h.run()
    assert any("hi-Deva/faithfulness" in w for w in r.warnings)
