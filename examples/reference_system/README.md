# Reference system

A hybrid RAG system over insurance policy documents. It exists to be measured,
not to be shipped: Stratum needs something real to point at, and a tool only
ever tested against its own mock is unproven.

Not part of the `stratum-eval` package. Installing Stratum does not pull any of
this in.

## Run it

```bash
python examples/reference_system/eval/build_dataset.py

stratum run \
  --endpoint examples/reference_system/endpoint.py:endpoint \
  --dataset examples/reference_system/eval/insurance.jsonl \
  --verified "en,hi-Deva,hi-Latn" \
  --out reports/ref-001
```

No `--glossary` flag: `endpoint.py` declares a module-level `glossary`
(`render.glossary.build_glossary()`, the same `eval/glossary.json` the
renderer enforces against), and `stratum run` picks that up on its own —
see cli.py's `_load_endpoint`. Pass `--glossary` only to override it with a
different file.

No downloads. The default embedder is a deterministic character-ngram hasher, and
script detection / transliteration / translation default to no-download stubs, so
the whole pipeline runs in about a second.

For real retrieval quality:

```bash
pip install -e "examples/reference_system[models]"   # BGE-M3, ~2GB on first use
pip install -e "examples/reference_system[pdf]"      # PyMuPDF, for real policy PDFs
pip install -e "examples/reference_system[indic]"    # IndicXlit + IndicTrans2 + fastText lid
```

## Shape

| Stage | Implementation |
|---|---|
| Ingest | PyMuPDF or plain text → paragraph-first recursive chunking → content-addressed `chunk_id` |
| Query (S0+S1) | Script detection → transliteration → translation, one normalized English query out |
| Dense | BGE-M3 (or hashing fallback) → exact cosine over an in-memory matrix |
| Sparse | BM25 Okapi, written out rather than imported |
| Fusion | Reciprocal rank fusion, `k=60`, each arm independently switchable |
| Answer | Extractive span selection, IDF-weighted overlap, relevance floor → refusal |
| Render (S4) | Placeholder/numeral mask → translation (en → hi-Deva) → glossary enforcement → [hi-Latn: romanize → enforce again] → unmask |

Exact search, not ANN: at a few thousand chunks brute force is milliseconds, and
an approximate index would inject recall error into the thing being measured.

Extractive, not generative: the answer is a span that exists in the corpus, so
correctness is a set comparison rather than a judgement call. The whole cascade
becomes measurable with no model and no calibration. A generative endpoint is
added alongside later as a second experiment.

## The capability contract

The system honours all three oracle hooks:

| Hook | Effect |
|---|---|
| query override | the harness substitutes the gold query text |
| `context_chunk_ids` | retrieval is skipped, these chunks are used |
| `answer_override` | answering is skipped, this text goes to rendering |

Declaring a hook and ignoring it is the one failure Stratum cannot detect: the
rung would read zero and its loss would be absorbed silently by a neighbour.
`tests/test_reference_system.py::TestOracleHooks` verifies each one is real.

## What the first run found

Hindi scored 14.3 against English 90.5, with the entire loss attributed to input
and query processing.

That is correct. There is no translation or transliteration layer yet, so Hindi
queries are matched by BM25 against English text, retrieve nothing relevant, and
fall below the relevance floor — 30 over-refusals. The cascade located the
missing stage without being told it was missing.

Two bugs were found building this, both worth keeping in mind:

**Stopword matching defeated the refusal floor.** "What is the capital of Brazil?"
matched a sentence about maternity benefits on `the` and `of` alone. Span scoring
is now IDF-weighted, so a match must be earned on discriminating terms.

**`\w` shatters Indic words.** Python classifies Devanagari vowel signs as
combining marks, which `\w` excludes, so `प्रीमियम` tokenised as
`['प', 'र', 'म', 'यम']` — every Hindi word broken into fragments matching
nothing. The token pattern now includes the Indic block ranges explicitly.

The second one would have silently destroyed sparse retrieval for every Indic
language while looking like a modelling problem.

## What the query pipeline fixed

`query/` adds S0 (script detection) and S1 (transliteration + translation),
following the same real-model / no-download-stub split as the embedder:

| Stage | Real | Stub (default, offline) |
|---|---|---|
| Script detection | fastText lid218e | Unicode block + Hindi function-word ratio |
| Transliteration, hi-Latn → Deva | IndicXlit | syllable-table lookup |
| Translation, hi-Deva → en | IndicTrans2 | hand-built lexicon |

Re-running with the stub pipeline, nothing else changed:

| Language | Before | After |
|---|---|---|
| hi-Deva | 14.3 | 77.8 |
| hi-Latn | 50.8 | 55.6 |

hi-Deva recovered almost entirely. The lexicon covers most of the domain's
function and content words, so a word-for-word gloss is usually enough for
BM25 to land on the right chunk — attributable loss on `Input + query
processing` dropped from 76.2 points to 12.7.

hi-Latn moved less, and the reason is itself a finding worth keeping. This
dataset's hi-Latn queries are code-mixed by design — "room rent ki limit kya
hai" says "room", "rent" and "limit" in English mid-sentence, the way real
Hinglish insurance queries do. Transliterating every token blindly turned
"room"/"rent" into Devanagari that matches nothing, which scored *worse*
than doing no query processing at all (19.1, against a 50.8 do-nothing
baseline) — because the untouched English words used to at least match the
corpus verbatim, and mangling them threw that away for no gain.
`SelectiveTransliterator` (query/transliterate.py) is the fix: it only
converts tokens script detection already flagged as Hindi markers, leaving
likely-English tokens alone. That recovered the regression and edged past
the do-nothing baseline, but it is a containment measure, not a real
solution to code-switching — a model that actually knows which mid-sentence
tokens are English would do better, which is why IndicXlit is the real
implementation here rather than a bigger rule table.

## What auditing the cascade found

**`gold_answer` was the question, not an answer.** `build_dataset.py` set it
to `queries["en"]` — so `oracle_answer` fed the English *question* to
rendering as `answer_override`, and the S4 checks (numeral/placeholder/entity
survival) scored how well a question resembles an answer. That is why the
`Output rendering` rung read an identical −17.5 for both hi-Deva and
hi-Latn: it wasn't measuring rendering, it was measuring the same
category error twice. Fixed by extracting the actual gold span from the
resolved gold chunk (`gold_answer_for`, using the same overlap-scoring the
system's own answerer uses) — rendering loss dropped to −4.8 for both
languages and is now flagged `noise`, which is the honest result for a
stage that (per the Shape table above) isn't built yet.

**The dominant-stage share overclaimed completeness.** `render_cascade`
divided the dominant stage's points by `total_loss` regardless of whether
`Retrieval`/`Generation` were measurable, so it printed "100% of loss" for
a stage that — with two rungs blank — cannot actually account for the whole
gap; the missing rungs might hide more. Fixed in `stratum/attribution.py`:
when any stage is unmeasurable, the denominator is the sum of measured
positive rungs and the label reads "% of measured loss" instead. hi-Latn's
`Input + query processing` now reads 88% of measured loss, not 100% of
loss — a real change in claim, not just wording, since the two numbers
differ. Covered by `tests/test_attribution.py::test_dominant_share_is_of_measured_loss_when_stages_unmeasurable`.

**The "pre-existing disease" span-selection bug is fixed.** Both the
definition sentence and the actual waiting-period sentence were always
retrieved — the correct one usually ranked first — so this was never a
retrieval problem. `select_span`'s overlap scoring matched on exact token
equality, and "disease" (query) matching the *definition*'s singular
"disease" outscored "diseases"/"covered" (query: "cover") failing to match
the answer sentence's plural/inflected forms at all. `_stem` in
`pipeline/extractive.py` now matches on a light suffix-stripped form for
scoring purposes while still pricing each match by its real token's IDF, so
plurals and simple verb forms no longer lose to an exact-but-wrong match.
Scoped to span selection only — `tokenize()` and the BM25 index are
untouched, so retrieval ranking doesn't change. Rerunning:

| Language | Before this fix | After |
|---|---|---|
| hi-Deva | 77.8 | **84.1**, now indistinguishable from baseline |
| hi-Latn | 55.6 | **57.1** |

hi-Deva's remaining gap to English is no longer distinguishable from noise
at this sample size — the query pipeline plus this fix closes essentially
all of it. hi-Latn's gap is still real and still `Input + query processing`
dominant, consistent with the code-switching limitation described above.

**numeral_integrity for hi-Latn (33.3%, 11 failures) is not digit
corruption.** Traced every failure by hand: ASCII digits survive
`RuleBasedTransliterator` and `LexiconTranslator` unchanged in every case
tested (`45`, `500000`, `61`, `30`, `40000`, `36`, `15` — see
`TestTransliteration::test_digits_survive_untouched` and
`TestQueryPipeline::test_numerals_survive_hi_latn_normalization`). The real
cause is over-refusal: hi-Latn's marker-only transliteration still leaves
many Hindi tokens neither converted nor understood (`chahiye`, `ghante`,
`dusri`, ...), which pulls span-overlap scores below the relevance floor
more often than for hi-Deva or en — most of the 11 failures are refusals
with an empty answer, not an answer with the wrong number in it. One case
(`ins-0047`/`ins-0048`, "pre-existing disease" waiting period) is a genuine
extractive-answering bug independent of language: the span picker selects a
nearby sentence mentioning "48 months" from the definitions clause instead
of the correct "36 months" waiting-period sentence — worth its own fix, but
an S3 answering issue, not an S0/S1 one.

**`LexiconTranslator`'s dictionary is train-on-test.** Its ~110 words were
selected by scanning the Devanagari vocabulary that appears across this
dataset's own `SPEC`/`UNANSWERABLE` entries in `build_dataset.py`, then
glossed by hand. It does not memorize full queries or answers — coverage is
per-word, not per-item — but "does this word happen to be in the
dictionary" is not independent of "was this word used to build the
dictionary." The hi-Deva/hi-Latn numbers above, produced with this backend
active, are optimistic relative to what a lexicon built independently of
this eval set would score. `LexiconTranslator` now warns on construction
(`tests/test_reference_system.py::test_warns_that_its_lexicon_is_train_on_test`);
treat any hi-Deva result produced with the stub translator as a plumbing
check, not a quality claim — `IndicTrans2Translator` is the one to cite.

## What S4 rendering added

`render/` turns the S3 answer span into the query's own language — the
`Render` row the Shape table used to mark "not built yet". Same real/stub
split as everywhere else in this codebase:

| Stage | Real | Stub (default, offline) |
|---|---|---|
| Translation, en → hi-Deva | IndicTrans2 (en-indic) | hand-built lexicon, written from general insurance vocabulary, not scanned from `build_dataset.py` |
| Romanization, hi-Deva → hi-Latn | `indic_transliteration` (ITRANS scheme) | character-table Devanagari → Roman |
| Placeholder / numeral integrity | — | mask before translation, restore after (verbatim for placeholders, Indian-grouped for numerals) |
| Glossary enforcement | — | `eval/glossary.json`, the same file `endpoint.py` exposes to `stratum run` |

Two new corpus sentences (`policy_network_claims.md`'s "Claim status
notifications" section) carry genuine `{placeholder}` tokens — an SMS and an
email template — so `placeholder_integrity` has real content to measure
instead of "no placeholders in source" on every item.

**A bracket-plus-letter sentinel collided with the translator's own
tokenizer.** The first mask design used `⟦A⟧`-style tokens; `translate_en.py`'s
EN→HI tokenizer splits on letter/non-letter boundaries, so `⟦A⟧` split into
`⟦`, `A`, `⟧` — and the bare `A` matched the dictionary's `"a" → ""`
(article, dropped) entry, silently erasing the placeholder. Fixed by using a
single Unicode Private Use Area character per sentinel (`mask.py`): one
codepoint has no internal boundary for any tokenizer in this pipeline to
split on, and it isn't a Latin letter, digit, or Devanagari character, so no
table or lexicon here has an entry that could match it. Caught by
`TestMask::test_two_masking_passes_do_not_corrupt_each_other` before it ever
reached a real run.

**Glossary enforcement checked the wrong text for what's in scope.** The
first version decided which glossary terms applied by scanning the
*answer span* (`source_en`) — but stratum's own `check_glossary` scores
against the **query** (`item.query`). A user asking "is knee surgery
covered?" is in scope for "cover" regardless of whether the retrieved
sub-limit sentence happens to use that word, and it usually doesn't.
Checking only the answer made enforcement blind to most of its own
`terminology_drift` failures — measured, not assumed: fixing this dropped
hi-Latn's `terminology_drift` count from 10 to 7 and its
`glossary_adherence` score from 44.4% to 61.1% in the same run. `enforce()`
now takes the query as well as the answer (`render/glossary.py`,
`render/pipeline.py`); the remaining 7 failures are empty answers from the
already-documented hi-Latn over-refusal issue above, not an enforcement gap
— confirmed by checking that every remaining failure's output is empty.

Re-running with S4 wired in and the two new items — no `--glossary` flag
needed; see "Where the glossary lives" below:

| Language | Placeholder integrity | Glossary adherence | Quality |
|---|---|---|---|
| en | 100% | 100% | 92.4 |
| hi-Deva | 100% | 100% | 85.5 (indistinguishable from baseline) |
| hi-Latn | 100% | 61.1% (gate: 85%, still failing) | 60.9 |

Placeholder integrity is clean across all three languages — the mask/restore
mechanism holds up from an isolated string through the real query pipeline,
retriever, and render pipeline together
(`TestRenderIntegration`). Glossary adherence is clean for en/hi-Deva and
still fails the gate for hi-Latn, entirely because of over-refusal rather
than mistranslated terminology: there is no approved term to enforce in an
answer that's empty. That over-refusal is the same S1 coverage gap
documented above, not a new S4 problem — fixing it means improving
`SelectiveTransliterator`'s marker coverage or the S1 lexicon, not the
renderer.

## Where the glossary lives, and a gate bug it exposed

`eval/glossary.json` is the one glossary — `render/glossary.py` loads it for
the renderer's own term enforcement, and `endpoint.py` exposes it at module
level (`glossary = build_glossary()`) so `stratum run` picks it up on its
own (`cli.py`'s `_load_endpoint` now returns the module, not just the
endpoint, and `run` reads `module.glossary` when `--glossary` isn't passed
explicitly). `--glossary` still works, as an override — useful for scoring
against a different term list without touching the endpoint — but the
reference system no longer depends on the caller remembering it.

Before `endpoint.py` exposed its own glossary, running without the
`--glossary` flag — the easy mistake this section exists to prevent —
meant `glossary_adherence` had zero observations for every language (no
glossary loaded means the metric is never computed, `n=0`), and the gate
still printed `pass`.

The bug was in `Report.evaluate_gates` (`stratum/report.py`): the loop
initializes `passed = True` and only ever sets it `False` inside the branch
that runs when a value *was* observed and failed the threshold. A metric
that is `None` for every in-scope language never enters that branch, so
`passed` never moves off its default — the gate reports "pass" having
never actually been evaluated. `evaluate_gates` now tracks whether *any*
value was observed at all; if not, the gate is marked `skipped_reason:
"no observations for this metric — cannot evaluate"` and, by default
(`fail_on_unevaluated=True`, also exposed as `stratum run
--fail-on-unevaluated/--no-fail-on-unevaluated`), that counts as a failure
rather than a pass — the same principle `attribution.py` already applies
to cascade stages a judge-free run can't measure: an honest "didn't run" is
worse to hide than to report as a problem. `tests/test_report.py` covers
both the missing-key and present-but-`n=0` shapes a real Estimate can take,
plus that gates with real data are unaffected.
