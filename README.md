# stratum

**Find out where your multilingual RAG system breaks — and which stage is causing it.**

[![PyPI](https://img.shields.io/badge/pypi-v0.1.0-blue)](https://pypi.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

---

## The problem

Built a RAG system. It works well in English. Then you added Hindi, Tamil, Indonesian, Swahili.

Does it still work?

Nobody knows. Teams ship translation layers on top of English RAG and find out from support tickets. Existing eval frameworks (RAGAS, DeepEval, TruLens) are English-first and give you **one end-to-end number** — which tells you something is wrong, but not what.

`stratum-eval` gives you a **per-stage, per-language breakdown**, so instead of:

> Tamil scores 62%

you get:

> Tamil loses 22 points: 4 at query translation, 2 at retrieval, 1 at generation, **15 at output rendering**

One of those is actionable. The other is not.

---

## What it refuses to claim

Most eval tools report a number for everything they are asked about. stratum
withholds numbers the method cannot support, which is the point:

- Stages that deterministic checks cannot see are marked **not measured**, not
  estimated. Retrieval and generation affect the answer only through its
  content, and reading content needs a calibrated judge.
- Losses whose confidence interval spans zero are flagged **noise**, and the
  "fix this first" signal stays empty rather than naming a stage the sample
  cannot support.
- A language whose dataset has not been checked by a native speaker is reported
  as **experimental** and excluded from CI gates.
- Judged metrics are withheld entirely for any language without a calibration
  record, and reported with Cohen's κ alongside where one exists.

Every number carries its sample size and a bootstrap confidence interval.

## Install

```bash
pip install stratum-eval
```

---

## 60-second start

```python
from stratum import Harness, Endpoint

harness = Harness(
    endpoint=Endpoint(url="http://localhost:8000/query"),
    dataset="datasets/insurance-hi-ta.jsonl",
    languages=["en", "hi", "ta"],
    baseline_language="en",
)

report = harness.run()
report.save("reports/")          # writes report.json, report.html, report.md
print(report.summary())
```

Or from the CLI:

```bash
stratum run --config configs/baseline.yaml --out reports/
stratum compare reports/run-001 reports/run-014
```

---

## What it measures

The pipeline is decomposed into stages. Each is scored independently.

| Stage | Measures | Example failure it catches |
|-------|----------|---------------------------|
| **S0** Input handling | Script/language detection, transliteration accuracy | `mera loan approve hua kya` classified as English, not Roman-Hindi |
| **S1** Query processing | Entity preservation, retrieval-equivalence vs gold query | Policy number mangled during translation |
| **S2** Retrieval | Recall@k, MRR, nDCG — reported as Δ vs baseline language | Hindi query retrieves different chunks than the same English query |
| **S3** Generation | Claim-level faithfulness, answer correctness, refusal precision/recall | Model answers confidently from an unretrieved fact |
| **S4** Output rendering | Glossary adherence, placeholder integrity, numeral/date integrity, grammatical agreement | `{amount}` translated to `{राशि}` — breaks downstream string formatting |
| **S5** Voice *(optional)* | Entity-WER, TTS numeral/acronym expansion, per-stage latency | `₹1,50,000` spoken as "ek panch shunya shunya shunya" |

---

## The key idea: loss attribution

A single score can't tell you where quality drained away. `stratum-eval` runs **counterfactual passes** that bypass one stage at a time using known-good inputs:

| Pass | What it does | What the delta tells you |
|------|-------------|-------------------------|
| `oracle_query` | Feeds the gold baseline-language query straight to retrieval | Loss caused by S0 + S1 |
| `oracle_context` | Feeds gold chunks straight to generation | Loss caused by S2 |
| `oracle_answer` | Pushes the gold answer through output rendering only | Loss caused by S4 |

Subtract the deltas and you get the **degradation cascade** — the headline chart in every report.

---

## Judge calibration

Any metric backed by an LLM judge ships with an agreement score against human labels, per language.

```bash
stratum calibrate --dataset datasets/insurance-hi-ta.jsonl --sample 50 --lang hi
```

Cohen's κ is printed next to every judged metric in the report. If κ < 0.6 for a language, those numbers are flagged **low-confidence** rather than quietly reported. An unvalidated judge is a vibe with a decimal point.

---

## Validation: does the cascade name the right stage?

Every other test in this repo checks that stratum computes its documented formulas correctly. `tests/test_validation.py` checks the harder claim: given a system with a real, known defect at a known stage, does the cascade actually attribute failure *there*.

Defects are injected by mechanism — S0 language misroute, S1 degraded translation, S2 wrong-chunk retrieval, S4 rendering corruption — into synthetic RAG systems with known ground truth, so `dominant_stage` can be checked exactly rather than eyeballed. A second suite sweeps *how small a defect stratum still reliably catches*, since a tool only proven on obvious failures hasn't proven much:

**Smallest injected-stage loss magnitude Stratum reliably attributes (≥ 80% correct, measured, non-noise), by dataset size:**

| Paired items (n) | Smallest reliably-attributed loss |
|---|---|
| 20 | ~20 points |
| 50 | ~5 points |
| 100 | ~5 points |

More data resolves smaller defects — a 20-item dataset only reliably names the stage behind a ~20-point loss, where 50 and 100 items both get there at ~5 points (100 items shows a further partial edge at ~2 points that 50 doesn't, just not enough to clear the 80% bar). Reproduce with `pytest tests/test_validation.py -k sensitivity -s`.

As a secondary, easier check: on 20 large, obvious synthetic defects (whole wrong-item swaps, numeral corruption far outside any real value, at 60-100% severity), **Stratum identified the injected dominant failure stage in 20/20 cases** — the floor the original test asserts, not evidence the ladder resolves subtler defects; that's what the sensitivity sweep above is for. Reproduce with `pytest tests/test_validation.py -k 20_synthetic_systems -s`.

**Both numbers describe controlled, synthetic defects on a synthetic dataset — a sensitivity floor for the attribution *method*, not a guarantee about what any particular real-world system's real defects will look like or how small a real regression this will catch.**

What the suite is honest about rather than glossing over:

- **S0 and S1 are graded as one combined stage**, `s0_s1_input_query`, because that's all the ladder can isolate — there's no oracle pass that repairs one without the other. Both injection mechanisms above are scored against that combined key, never against a nonexistent S0-only or S1-only rung.
- **S2 and S3 are unconditionally unmeasurable without a calibrated judge**, even when a deterministic check would incidentally catch the defect. A dedicated test confirms the fallback is honest: an uncalibrated S2 run reports the stage as `not measured`, never guesses, and neither `not measured` nor `noise` is ever counted as a correct attribution anywhere in this suite, at any dataset size.
- **Faithfulness doesn't calibrate for retrieval defects** — a wrong-chunk answer is still faithful to the (wrong) chunk it came from, so faithfulness can't see a retrieval failure. `answer_correctness`, scored against the true external reference regardless of which context was supplied, is the metric that actually detects S2 — and the one this suite calibrates, freshly, at each dataset size.
- **S3 is not defect-injected** in this suite at all (out of scope for this pass); its shared judge-dependency with S2 is noted, not tested.

---

## CI integration

```yaml
# .github/workflows/eval.yml
- run: stratum run --config configs/ci.yaml --fail-under configs/thresholds.yaml
```

```yaml
# configs/thresholds.yaml
gates:
  - metric: retrieval.recall_at_5
    languages: [hi, ta]
    max_regression: 2.0        # percentage points vs last green run
  - metric: rendering.placeholder_integrity
    languages: all
    min_absolute: 100.0        # placeholders are a bug class, not a score
```

A PR that quietly degrades Tamil fails the build, the same way a unit test would.

---

## Repository layout

```
stratum-eval/
├── src/stratum/
│   ├── harness.py              # orchestrates runs, passes, retries
│   ├── endpoint.py             # adapters: HTTP, LangChain, LlamaIndex, custom callable
│   ├── passes/
│   │   ├── standard.py
│   │   ├── oracle_query.py
│   │   ├── oracle_context.py
│   │   └── oracle_answer.py
│   ├── metrics/
│   │   ├── s0_input.py         # script detection, transliteration
│   │   ├── s1_query.py         # entity preservation, retrieval-equivalence
│   │   ├── s2_retrieval.py     # recall@k, mrr, ndcg
│   │   ├── s3_generation.py    # faithfulness, correctness, refusal
│   │   ├── s4_rendering.py     # glossary, placeholders, numerals, agreement
│   │   └── s5_voice.py         # entity-WER, TTS probes, latency
│   ├── judges/
│   │   ├── base.py
│   │   ├── llm_judge.py
│   │   └── calibration.py      # human-agreement, Cohen's κ
│   ├── attribution.py          # the cascade computation
│   ├── taxonomy.py             # auto-tagging of failed items
│   └── report/
│       ├── schema.py           # versioned report contract
│       ├── html.py
│       └── markdown.py
├── datasets/
│   ├── README.md               # how to build a question set that isn't translationese
│   ├── schema.json
│   └── insurance-hi-ta.jsonl   # reference dataset, 150 items/language
├── configs/
│   ├── baseline.yaml           # naive: translate → English RAG → translate back
│   ├── ci.yaml
│   └── thresholds.yaml
├── examples/
│   ├── sample_report.md
│   ├── langchain_endpoint.py
│   └── notebooks/
├── docs/
│   ├── metrics.md              # every metric: definition + known weakness
│   ├── attribution.md
│   ├── dataset-design.md
│   └── adding-a-language.md
└── tests/
```

---

## Datasets

The most common way to get flattering, useless numbers is to machine-translate your English eval set. That produces *translationese* — clean, well-formed queries no real user types.

The reference dataset uses a deliberate mix. See [`docs/dataset-design.md`](docs/dataset-design.md).

| Slice | Share | Why |
|-------|-------|-----|
| Parallel core (human-verified) | 40% | Enables clean cross-language comparison and attribution |
| Native-authored | 25% | Real phrasing: indirect, code-mixed, honorific |
| Term-heavy | 10% | Forces glossary terms into play |
| Unanswerable / adversarial | 15% | Refusal behaviour — both misses *and* over-refusal |
| Multi-hop / comparative | 10% | Retrieval depth |

Hindi is cross-cut by script (Devanagari / Roman / mixed) and reported separately.

---

## Bring your own language

`stratum-eval` is not Indic-specific. A language needs four things, none of which require code changes:

1. A dataset file following [`datasets/schema.json`](datasets/schema.json)
2. A glossary (optional, enables S4 terminology metrics)
3. An agreement probe set (optional, enables grammatical-agreement checks)
4. ~50 human-labelled items for judge calibration

See [`docs/adding-a-language.md`](docs/adding-a-language.md).

---

## Status

`v0.1.0` — early. The metric definitions in `docs/metrics.md` are the stable contract; internals will move.

Contributions especially welcome for: new language packs, endpoint adapters, and failure classes we haven't named yet.

## License

Apache 2.0
