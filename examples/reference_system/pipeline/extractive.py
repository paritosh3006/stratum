"""Extractive answering: return a span from the corpus, not a generated one.

Why extractive first. If the answer is a span that exists in the corpus, then
correctness is a set comparison — exact match on span id, token-F1 against the
gold span. No judge, no model, no API, and no calibration problem.

That makes the *whole* cascade measurable, including the Retrieval and
Generation rungs that currently read "not measured". A generative endpoint is
added alongside later; it needs a judge, and it is the second experiment
rather than a prerequisite.

The system also has to know when to say nothing. An extractive system that
always returns its best span cannot be wrong in an interesting way, so a
relevance floor produces a genuine refusal — and Stratum scores refusal
behaviour in both directions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from ..ingest.chunk import split_sentences
from ..retrieval.hybrid import Hit, tokenize


@dataclass
class Span:
    span_id: str
    chunk_id: str
    text: str
    score: float


def span_id_for(chunk_id: str, text: str) -> str:
    digest = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:8]
    return f"{chunk_id}#{digest}"


def _overlap_score(
    query_tokens: set[str],
    sentence: str,
    idf: Callable[[str], float] | None = None,
) -> float:
    """IDF-weighted term overlap, length-damped.

    Unweighted overlap is unusable here. "What is the capital of Brazil?"
    matched a sentence about maternity benefits on `the` and `of` alone and
    cleared the relevance floor — the system answered a question it should
    have refused.

    Weighting by inverse document frequency makes function words contribute
    almost nothing, so a match has to be earned on terms that actually
    discriminate between chunks. Damping by sentence mass keeps the longest
    sentence from winning simply by containing more words.
    """
    sent_tokens = set(tokenize(sentence))
    if not sent_tokens or not query_tokens:
        return 0.0

    weight = idf if idf is not None else (lambda _t: 1.0)

    matched = query_tokens & sent_tokens
    if not matched:
        return 0.0

    matched_mass = sum(weight(t) for t in matched)
    query_mass = sum(weight(t) for t in query_tokens)
    sent_mass = sum(weight(t) for t in sent_tokens)

    if not query_mass or not sent_mass:
        return 0.0

    coverage = matched_mass / query_mass
    precision = matched_mass / sent_mass
    return (2 * coverage * precision) / (coverage + precision)


def select_span(
    query: str,
    hits: list[Hit],
    *,
    window: int = 2,
    min_score: float = 0.12,
    idf: Callable[[str], float] | None = None,
) -> Span | None:
    """Pick the best sentence window across the retrieved chunks.

    A window rather than a single sentence because policy text routinely
    splits a rule from its exception across adjacent sentences, and returning
    only the first half would be worse than useless.
    """
    query_tokens = set(tokenize(query))
    best: Span | None = None

    for hit in hits:
        sentences = split_sentences(hit.text)
        if not sentences:
            continue
        for i in range(len(sentences)):
            for size in range(1, window + 1):
                if i + size > len(sentences):
                    break
                candidate = " ".join(sentences[i : i + size])
                score = _overlap_score(query_tokens, candidate, idf)
                if best is None or score > best.score:
                    best = Span(
                        span_id=span_id_for(hit.chunk_id, candidate),
                        chunk_id=hit.chunk_id,
                        text=candidate,
                        score=score,
                    )

    if best is None or best.score < min_score:
        return None
    return best
