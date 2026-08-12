# stratum-eval — Run Report

**Run ID** `run-014` · **System** `insurance-assistant v2.3` · **Date** 2026-08-11
**Corpus** 412 policy documents · **Dataset** `insurance-hi-ta.jsonl` (150 items/language)
**Baseline language** English · **Config hash** `a9f3c1e` · **Dataset hash** `77b204d`

---

## 1. Scorecard

Overall quality, per language. Δ is against the baseline language. Target: within 5 points.

| Language | Script | Answer Quality | Δ | Retrieval R@5 | Faithfulness | Refusal F1 | Status |
|---|---|---|---|---|---|---|---|
| English | Latin | 84.2 ±3.1 | — | 88.0 | 91.5 | 0.86 | baseline |
| Hindi | Devanagari | 81.1 ±3.4 | −3.1 | 86.4 | 89.2 | 0.83 | ✅ within target |
| Hindi | Roman | 71.8 ±3.9 | −12.4 | 74.1 | 87.0 | 0.79 | ⚠️ over target |
| Tamil | Tamil | 62.4 ±4.0 | −21.8 | 85.2 | 88.1 | 0.81 | ❌ over target |

> **Read this first:** Tamil's retrieval and faithfulness are near-baseline. The loss is *not* in the RAG core. See §2.

*Confidence intervals are 95% bootstrap over n=150. Judged metrics carry calibration scores — see §6.*

---

## 2. Degradation cascade

Where the quality actually drains away. Points lost at each stage, computed from counterfactual passes.

### Tamil — 21.8 points lost

```
Baseline (English)                                    84.2
                                                         │
S0  Input handling            −0.3  ▏                 83.9
S1  Query processing          −4.1  ██▌               79.8
S2  Retrieval                 −1.8  █                 78.0
S3  Generation                −0.9  ▌                 77.1
S4  Output rendering         −14.7  █████████▎        62.4  ◀── 67% of total loss
                                                         │
Final (Tamil)                                         62.4
```

**Verdict:** two thirds of the Tamil loss happens *after* the correct answer has been generated. Retrieval tuning would recover at most 1.8 points. Fix rendering first.

### Hindi (Roman) — 12.4 points lost

```
S0  Input handling            −7.2  ████▊             ◀── script misdetection
S1  Query processing          −3.1  ██
S2  Retrieval                 −1.4  ▉
S3  Generation                −0.4  ▏
S4  Output rendering          −0.3  ▏
```

**Verdict:** opposite problem. Roman-script Hindi is being misclassified at the door, so the query never reaches the pipeline in usable form. Everything downstream is fine.

---

## 3. Stage detail

### S0 — Input handling

| Metric | en | hi-Deva | hi-Latn | ta |
|---|---|---|---|---|
| Language detection accuracy | 99.3 | 98.7 | **71.3** | 99.1 |
| Script detection accuracy | 100 | 99.3 | **68.0** | 100 |
| Transliteration exact-match (entities) | — | 94.0 | 81.3 | 91.3 |

### S1 — Query processing

| Metric | hi-Deva | hi-Latn | ta |
|---|---|---|---|
| Retrieval-equivalence (Jaccard@5 vs gold query) | 0.87 | 0.72 | 0.79 |
| Entity preservation rate | 96.0 | 88.7 | 89.3 |
| Numeric preservation rate | 98.7 | 95.3 | 97.3 |

### S2 — Retrieval

| Metric | en | hi-Deva | hi-Latn | ta |
|---|---|---|---|---|
| Recall@5 | 88.0 | 86.4 | 74.1 | 85.2 |
| Recall@10 | 93.3 | 92.0 | 82.7 | 91.3 |
| MRR | 0.79 | 0.77 | 0.64 | 0.76 |
| nDCG@10 | 0.81 | 0.79 | 0.67 | 0.78 |

### S3 — Generation

| Metric | en | hi-Deva | hi-Latn | ta |
|---|---|---|---|---|
| Faithfulness (% grounded claims) | 91.5 | 89.2 | 87.0 | 88.1 |
| Answer correctness (0–3 rubric) | 2.51 | 2.43 | 2.29 | 2.38 |
| Refusal precision | 0.89 | 0.86 | 0.82 | 0.84 |
| Refusal recall | 0.83 | 0.80 | 0.76 | 0.78 |
| **Over-refusal rate** | 4.0 | 5.3 | 8.7 | 6.0 |

### S4 — Output rendering ◀ *primary loss stage*

| Metric | hi-Deva | hi-Latn | ta |
|---|---|---|---|
| **Glossary adherence** | 88.0 | 86.7 | **61.3** |
| **Placeholder integrity** | 100 | 100 | **78.0** ❌ |
| Numeral / date / unit integrity | 97.3 | 96.0 | 84.0 |
| Grammatical agreement (probe set) | 92.0 | 91.3 | 88.7 |
| Adequacy (1–5, judged) | 4.31 | 4.22 | 3.44 |

> Placeholder integrity below 100 is treated as a **bug class, not a score**. 33 of 150 Tamil outputs would fail downstream string formatting.

### S5 — Voice

| Metric | en | hi-Deva | ta |
|---|---|---|---|
| WER | 8.1 | 14.2 | 17.6 |
| **Entity-WER** | 6.4 | 19.8 | 24.1 |
| TTS numeral expansion correctness | 98.0 | 88.0 | 79.3 |
| TTS acronym expansion correctness | 96.0 | 74.0 | 68.7 |

*Entity-WER is the metric that matters — errors on words that change what gets retrieved.*

---

## 4. Failure taxonomy

Every failed item auto-tagged. Counts across all non-baseline languages.

| Class | Count | Share | Stage |
|---|---|---|---|
| Terminology drift | 71 | 26% | S4 |
| Placeholder corruption | 33 | 12% | S4 |
| Script misdetection | 31 | 11% | S0 |
| Retrieval miss | 28 | 10% | S2 |
| Entity mangled in query | 24 | 9% | S1 |
| Numeral / unit error | 22 | 8% | S4 |
| Over-refusal | 19 | 7% | S3 |
| Unfaithful claim | 17 | 6% | S3 |
| Agreement error | 15 | 5% | S4 |
| Other | 14 | 5% | — |

### Examples

**Placeholder corruption** · `ta` · item `ins-0231`
```
Template   Your EMI of ₹{amount} is due on {date}
Output     உங்கள் EMI ₹{தொகை} {தேதி} அன்று செலுத்த வேண்டும்
Problem    Placeholder identifiers translated. Downstream .format() raises KeyError.
```

**Terminology drift** · `ta` · item `ins-0088`
```
Term       "premium"
Rendered   3 distinct forms across 11 occurrences
Expected   single approved glossary form
Problem    Same concept named three ways in one answer. User cannot map to policy doc.
```

**Script misdetection** · `hi-Latn` · item `ins-0402`
```
Query      mera knee operation cover hoga kya
Detected   en (confidence 0.61)
Expected   hi-Latn
Problem    Routed to English pipeline. Retrieved chunks unrelated to the question.
```

**Over-refusal** · `hi-Deva` · item `ins-0117`
```
Query      क्या मेरी पॉलिसी में दांतों का इलाज शामिल है?
Response   refused — "information not available"
Expected   answerable; gold chunk was retrieved at rank 2
Problem    Model refused despite having the evidence in context.
```

*Full failure log: `report.json → failures[]`*

---

## 5. Baseline comparison

Same dataset, same corpus, two pipelines.

| | Naive baseline¹ | This system | Δ |
|---|---|---|---|
| Hindi (Deva) | 68.4 | 81.1 | **+12.7** |
| Hindi (Roman) | 52.1 | 71.8 | **+19.7** |
| Tamil | 59.8 | 62.4 | +2.6 |
| Placeholder integrity (avg) | 71.0 | 92.7 | **+21.7** |
| Glossary adherence (avg) | 44.2 | 78.7 | **+34.5** |

¹ *MT → English RAG → MT back, no glossary, no placeholder protection, no script detection.*

Tamil's small gain is consistent with §2: the pipeline's improvements are concentrated in stages that aren't Tamil's bottleneck.

---

## 6. Judge calibration

Human agreement on a 50-item stratified sample per language.

| Judged metric | hi | ta | Confidence |
|---|---|---|---|
| Faithfulness | κ = 0.78 | κ = 0.71 | high |
| Answer correctness | κ = 0.69 | κ = 0.64 | high |
| Adequacy | κ = 0.66 | **κ = 0.52** | ⚠️ low for `ta` |

> Tamil adequacy scores are **low-confidence** and should not be used for gating until the judge is re-calibrated or replaced with a native-speaker panel.

---

## 7. Latency & cost

Per-stage, p50 / p90 in ms.

| Stage | en | hi | ta |
|---|---|---|---|
| S0 input | 12 / 31 | 18 / 44 | 19 / 47 |
| S1 query | 4 / 9 | 210 / 480 | 224 / 510 |
| S2 retrieval | 88 / 190 | 91 / 198 | 90 / 195 |
| S3 generation | 1,240 / 2,900 | 1,310 / 3,050 | 1,290 / 3,010 |
| S4 rendering | 6 / 14 | 240 / 560 | 251 / 590 |
| **End-to-end** | **1,350 / 3,144** | **1,869 / 4,332** | **1,874 / 4,352** |

Cost per query: `en` $0.0031 · `hi` $0.0048 · `ta` $0.0049

---

## 8. Gates

| Gate | Result |
|---|---|
| `retrieval.recall_at_5` max regression 2.0pp — `hi`, `ta` | ✅ pass |
| `rendering.placeholder_integrity` = 100 — all | ❌ **fail** (`ta` at 78.0) |
| `generation.faithfulness` min 85.0 — all | ✅ pass |

**Run status: FAILED** — 1 gate breached.

---

## 9. Recommended actions

Ordered by points recoverable per unit of effort.

1. **Protect placeholders before translation** (`ta`) — mask identifiers, translate, restore. Recovers ~22pp on placeholder integrity, unblocks the failing gate.
2. **Enforce glossary at render time** (`ta`) — constrained decoding or post-hoc term substitution against the approved list. Est. +8–11pp answer quality.
3. **Add Roman-script detection** (`hi-Latn`) — dedicated classifier ahead of language routing. Est. +7pp, largest single win for Hindi.
4. **Re-calibrate the Tamil adequacy judge** — current κ = 0.52 makes S4 Tamil numbers unreliable; fix before trusting action 2's measurement.
5. Retrieval tuning — **deprioritise**. Worth ≤2pp for Tamil.

---

<sub>Generated by `stratum-eval v0.1.0` · reproduce: `stratum run --config configs/run-014.yaml`</sub>
