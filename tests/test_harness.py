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
