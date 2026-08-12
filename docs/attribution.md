# Attribution: where the quality went

A single end-to-end score tells you a language is worse. It does not tell you
which stage made it worse, and that is the only thing you can act on.

This document describes how stratum attributes loss to stages, and — more
importantly — what it refuses to claim.

---

## The ladder

stratum runs your system several times over the same items. Each run repairs
one more stage using known-good input drawn from the baseline-language twin of
that item. The runs form a monotone chain:

| Pass | Repaired | How |
|---|---|---|
| `standard` | nothing | your system, as it is |
| `oracle_query` | S0 + S1 | the gold baseline query is fed to retrieval |
| `oracle_context` | + S2 | the gold chunks are fed to generation |
| `oracle_answer` | + S3 | the gold answer is fed to rendering |
| `baseline` | everything | the baseline-language run |

Each stage's loss is the gap between adjacent rungs:

```
S0+S1 = score(oracle_query)   − score(standard)
S2    = score(oracle_context) − score(oracle_query)
S3    = score(oracle_answer)  − score(oracle_context)
S4    = score(baseline)       − score(oracle_answer)
```

These telescope, so the stage losses sum to the total by construction. There is
no residual term to explain away. `report.json` carries `sums_correctly` and the
check is asserted in the test suite rather than assumed.

Pairing is preserved throughout: the same item under two passes is a matched
pair, so differences use a **paired** bootstrap. Resampling the two arms
independently would discard that pairing and inflate every interval enough to
mark real effects as noise — the wrong error for a tool whose job is separating
signal from noise.

---

## What the score must be, and why

The cascade is computed on an **outcome score** built only from properties of
the answer: refusal correctness, placeholder integrity, numeral integrity,
glossary adherence, entity preservation, and — once calibrated — judged
faithfulness and correctness.

Retrieval recall and language-detection accuracy are deliberately **excluded**,
even though they are reported as diagnostics.

The reason is not stylistic. `oracle_context` hands the system the gold chunks,
so retrieval recall becomes perfect *by construction*. Had recall contributed to
the score, repairing retrieval would inflate its own rung, and the cascade would
report a stage loss it manufactured itself. An early version of this tool did
exactly that and attributed 186% of the total loss to retrieval.

The rule: **a metric may not price a stage that an oracle pass trivially
satisfies.** Diagnostics describe how a stage behaved. The outcome score prices
what the user actually received.

---

## The four honesty constraints

These are enforced in code, not left to the reader.

**1. Unsupported passes widen a band; they never move loss to a neighbour.**

An endpoint declares its `Capabilities`. If it cannot accept a context override,
S2 cannot be isolated, and stratum reports a combined `S2 + S3` band with
`isolated: false` — rather than silently folding retrieval loss into generation.

**2. Stages that deterministic checks cannot see are reported as not measured.**

Retrieval and generation affect the user-visible answer only through its
*content*, and no deterministic check reads content. Without a calibrated judge,
those rungs are marked `not measurable` and the surrounding rungs are labelled
upper bounds. stratum does not assign them a number the method cannot support.

This is why a run with no judge shows two blank rungs. That is the honest
picture, not a bug.

**3. Negative losses are reported, never clamped.**

Repairing a stage can appear to *hurt*, through sampling noise. Clamping to zero
would hide the sampling error and inflate the adjacent stage. Such rungs are
shown as measured and flagged `negative`.

**4. Every loss carries a confidence interval, and rungs whose interval spans
zero are flagged `noise`.**

`dominant_stage` — the "fix this first" signal — is only populated from rungs
that are both positive and distinguishable from zero. On a small sample it
returns nothing at all, which is the correct answer.

---

## Requirements on your dataset

Attribution needs a baseline-language twin for every item, linked by
`parallel_id`. Items without one are scored normally but contribute nothing to
the cascade; `Dataset.validate_parallelism` warns about them at load time.

The twins must be genuine translations of the same question. If your non-English
items are machine-translated from English, the ladder measures your translation
system rather than your RAG system — and it will look deceptively good, because
the oracle passes reverse exactly the transformation that produced the items.
See [dataset-design.md](dataset-design.md).

---

## Requirements on your endpoint

```python
from stratum import CallableEndpoint, RagResponse
from stratum.endpoint import Capabilities

def my_rag(query, language, *, context_chunk_ids=None, answer_override=None):
    # context_chunk_ids: skip retrieval, use these chunks
    # answer_override:   skip generation, render this answer
    ...

endpoint = CallableEndpoint(my_rag, capabilities=Capabilities(
    accepts_query_override=True,
    accepts_context_override=True,
    accepts_answer_override=True,
))
```

Declaring a capability you do not honour is the one failure mode stratum cannot
detect. If `context_chunk_ids` is accepted but ignored, the S2 rung will read
zero and the loss will be silently absorbed elsewhere. Wire the overrides before
you declare them.

Partial support is fine and common. Start with `accepts_query_override` — it is
the cheapest to wire and it alone separates "translation broke it" from
"everything downstream broke it", which is the most useful single split.
