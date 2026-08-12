"""S2 — retrieval metrics.

Deterministic, no judge needed. Always reported as a delta against the
baseline language, because absolute retrieval quality is a property of
the corpus, not of the language layer.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float | None:
    """Fraction of gold chunks appearing in the top k.

    Returns None when there is no gold — so it can be excluded from the
    mean rather than counted as a zero.
    """
    if not gold:
        return None
    top = set(retrieved[:k])
    return len(top & set(gold)) / len(gold)


def mrr(retrieved: list[str], gold: list[str]) -> float | None:
    """Reciprocal rank of the first gold chunk."""
    if not gold:
        return None
    gold_set = set(gold)
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float | None:
    """Binary-relevance nDCG@k."""
    if not gold:
        return None
    gold_set = set(gold)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved[:k], start=1)
        if cid in gold_set
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(gold), k) + 1)
    )
    return dcg / ideal if ideal else None


def jaccard_at_k(a: list[str], b: list[str], k: int) -> float:
    """Overlap between two result sets — used for retrieval-equivalence (S1).

    Comparing the translated query's results against the gold baseline
    query's results is a better signal than BLEU: it measures whether the
    translation preserved *retrieval behaviour*, which is what matters.
    """
    sa, sb = set(a[:k]), set(b[:k])
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0
