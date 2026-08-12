# Report format

Every run emits the same content through three surfaces. The JSON is the contract; terminal and HTML are renderings of it.

| Surface | File | Audience |
|---|---|---|
| Terminal | stdout | The developer who just ran it |
| Markdown / HTML | `report.md`, `report.html` | Sharing, PR comments, portfolio |
| JSON | `report.json` | CI gates, diffing, downstream tooling |

---

## 1. Terminal

Optimised for the first ten seconds. Headline, cascade, gate result, next action. Nothing else.

```
stratum-eval v0.1.0 · run-014 · insurance-assistant v2.3
150 items × 4 language variants · 412 docs · 6m 12s

  LANGUAGE        QUALITY      Δ      STATUS
  en (Latn)          84.2      —      baseline
  hi (Deva)          81.1   -3.1      ok
  hi (Latn)          71.8  -12.4      over target
  ta (Taml)          62.4  -21.8      OVER TARGET

  ta · where the 21.8 points went
    S0 input          -0.3  ▏
    S1 query          -4.1  ██▌
    S2 retrieval      -1.8  █
    S3 generation     -0.9  ▌
    S4 rendering     -14.7  █████████▎   <- 67% of loss

  GATES
    x rendering.placeholder_integrity   ta 78.0  (required 100)
    . retrieval.recall_at_5             pass
    . generation.faithfulness           pass

  RUN FAILED · 1 gate breached

  Top fix: protect placeholders before translation (ta)
           ~22pp recoverable, unblocks the gate

  Full report -> reports/run-014/report.html
```

Flags: `--quiet` (gates only), `--json` (machine output), `--no-color`.

---

## 2. JSON contract

Versioned. Additive changes bump minor; field removals bump major.

```jsonc
{
  "schema_version": "1.0",
  "run": {
    "id": "run-014",
    "started_at": "2026-08-11T09:14:22Z",
    "duration_s": 372,
    "system_label": "insurance-assistant v2.3",
    "config_hash": "a9f3c1e",
    "dataset_hash": "77b204d",
    "stratum_version": "0.1.0",
    "baseline_language": "en"
  },

  "languages": [
    {
      "code": "ta",
      "script": "Taml",
      "n_items": 150,
      "headline": {
        "answer_quality": 62.4,
        "ci95": [58.4, 66.4],
        "delta_vs_baseline": -21.8,
        "status": "over_target"
      },
      "stages": {
        "s0_input": {
          "language_detection_accuracy": 99.1,
          "script_detection_accuracy": 100.0,
          "transliteration_exact_match": 91.3
        },
        "s1_query": {
          "retrieval_equivalence_jaccard_at_5": 0.79,
          "entity_preservation_rate": 89.3,
          "numeric_preservation_rate": 97.3
        },
        "s2_retrieval": {
          "recall_at_5": 85.2,
          "recall_at_10": 91.3,
          "mrr": 0.76,
          "ndcg_at_10": 0.78
        },
        "s3_generation": {
          "faithfulness": 88.1,
          "answer_correctness": 2.38,
          "refusal_precision": 0.84,
          "refusal_recall": 0.78,
          "over_refusal_rate": 6.0
        },
        "s4_rendering": {
          "glossary_adherence": 61.3,
          "placeholder_integrity": 78.0,
          "numeral_integrity": 84.0,
          "agreement_accuracy": 88.7,
          "adequacy": 3.44
        },
        "s5_voice": {
          "wer": 17.6,
          "entity_wer": 24.1,
          "tts_numeral_correctness": 79.3,
          "tts_acronym_correctness": 68.7
        }
      },
      "attribution": {
        "method": "counterfactual_oracle",
        "baseline_score": 84.2,
        "final_score": 62.4,
        "total_loss": 21.8,
        "by_stage": {
          "s0_input": 0.3,
          "s1_query": 4.1,
          "s2_retrieval": 1.8,
          "s3_generation": 0.9,
          "s4_rendering": 14.7
        },
        "unexplained_residual": 0.0
      },
      "calibration": {
        "faithfulness":       { "kappa": 0.71, "n": 50, "confidence": "high" },
        "answer_correctness": { "kappa": 0.64, "n": 50, "confidence": "high" },
        "adequacy":           { "kappa": 0.52, "n": 50, "confidence": "low" }
      },
      "latency_ms": {
        "s0_input":      { "p50": 19,   "p90": 47   },
        "s1_query":      { "p50": 224,  "p90": 510  },
        "s2_retrieval":  { "p50": 90,   "p90": 195  },
        "s3_generation": { "p50": 1290, "p90": 3010 },
        "s4_rendering":  { "p50": 251,  "p90": 590  },
        "end_to_end":    { "p50": 1874, "p90": 4352 }
      },
      "cost_per_query_usd": 0.0049
    }
  ],

  "taxonomy": [
    { "class": "terminology_drift",     "stage": "s4", "count": 71, "share": 0.26 },
    { "class": "placeholder_corruption","stage": "s4", "count": 33, "share": 0.12 },
    { "class": "script_misdetection",   "stage": "s0", "count": 31, "share": 0.11 }
  ],

  "failures": [
    {
      "item_id": "ins-0231",
      "language": "ta",
      "class": "placeholder_corruption",
      "stage": "s4",
      "slice": "term_heavy",
      "input": "Your EMI of ₹{amount} is due on {date}",
      "output": "உங்கள் EMI ₹{தொகை} {தேதி} அன்று செலுத்த வேண்டும்",
      "expected_behaviour": "placeholder identifiers preserved verbatim",
      "detail": "Downstream .format() raises KeyError"
    }
  ],

  "comparison": {
    "against": "naive_mt_baseline",
    "deltas": { "hi-Deva": 12.7, "hi-Latn": 19.7, "ta": 2.6 }
  },

  "gates": [
    {
      "metric": "rendering.placeholder_integrity",
      "languages": ["ta"],
      "rule": "min_absolute",
      "threshold": 100.0,
      "observed": 78.0,
      "passed": false
    }
  ],

  "status": "failed"
}
```

### Design rules

- **Every score carries its n and its CI.** A number without sampling error invites false precision.
- **Every judged metric carries its κ.** Uncalibrated scores are marked `low`, never silently reported.
- **Attribution must sum.** `by_stage` totals must equal `total_loss` within tolerance; any gap is surfaced as `unexplained_residual` rather than hidden.
- **Failures are structured, not strings.** Every failed item is tagged with class, stage, and dataset slice so it can be grouped and diffed.

---

## 3. HTML dashboard

Four views, in this order. Order is the argument.

**View 1 — Degradation cascade.** Waterfall per language, baseline on the left, final on the right, stage losses between. This is the headline; everything else is supporting evidence.

**View 2 — Language scorecard.** One row per language × script variant. Colour against target, CIs shown, not hidden.

**View 3 — Failure taxonomy.** Tag counts with two clickable examples each. Filterable by stage, class, and dataset slice.

**View 4 — Baseline comparison.** Naive pipeline vs this one, same dataset. One bar chart.

Side panel: latency and cost per stage, calibration table, run metadata with config and dataset hashes.

---

## 4. Diffing runs

```bash
stratum compare reports/run-001 reports/run-014
```

```
  METRIC                          run-001   run-014        Δ
  ta · answer_quality                52.1      62.4    +10.3
  ta · glossary_adherence            44.2      61.3    +17.1
  ta · placeholder_integrity         71.0      78.0     +7.0
  hi-Latn · script_detection         68.0      68.0      0.0
  ta · recall_at_5                   85.9      85.2     -0.7

  Newly failing: none
  Newly passing: generation.faithfulness (ta)
```

Regression gating reads this diff, not absolute thresholds — a system can be legitimately below target while still improving.
