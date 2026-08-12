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
| Render | *not built yet — week 3* |

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
