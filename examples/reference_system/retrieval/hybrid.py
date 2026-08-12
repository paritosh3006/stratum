"""Hybrid retrieval: dense + sparse, fused by reciprocal rank.

Exact search, not ANN. At reference-corpus scale — a few thousand chunks —
brute-force cosine is milliseconds, and an approximate index would introduce
recall error into the very thing being measured. Swap in LanceDB or Qdrant
when the corpus outgrows memory; the interface is the same.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from ..ingest.chunk import Chunk
from .embedder import Embedder


@dataclass
class Hit:
    chunk_id: str
    score: float
    text: str
    source: str = ""   # "dense" | "sparse" | "fused"


# --------------------------------------------------------------------------
# Dense
# --------------------------------------------------------------------------

class DenseIndex:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []
        self.vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk]) -> None:
        self.chunk_ids.extend(c.chunk_id for c in chunks)
        self.texts.extend(c.text for c in chunks)
        self.vectors.extend(self.embedder.encode([c.text for c in chunks]))

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if not self.vectors:
            return []
        qv = self.embedder.encode([query])[0]
        scored = [
            (sum(a * b for a, b in zip(qv, vec)), i)
            for i, vec in enumerate(self.vectors)
        ]
        scored.sort(reverse=True)
        return [
            Hit(self.chunk_ids[i], score, self.texts[i], "dense")
            for score, i in scored[:k]
        ]


# --------------------------------------------------------------------------
# Sparse
# --------------------------------------------------------------------------

#: `\w` alone is wrong for Indic scripts. Python classifies vowel signs and
#: viramas as combining marks (Unicode category Mn/Mc), which `\w` excludes —
#: so "प्रीमियम" tokenised as ['प', 'र', 'म', 'यम'], shattering every Hindi word
#: into fragments that match nothing. The Indic block ranges are added back
#: explicitly, along with generic combining diacriticals.
_TOKEN = re.compile(r"[\w\u0300-\u036F\u0900-\u0DFF]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Unicode-aware word tokens, lowercased.

    Keeps Devanagari and Tamil words whole while splitting on punctuation —
    enough for BM25 without a script-specific segmenter dependency.
    """
    return _TOKEN.findall(text.lower())


class SparseIndex:
    """BM25 Okapi, implemented directly.

    Written out rather than pulled from a package because BM25 is forty lines,
    and the reference system should be readable end to end by someone
    auditing what Stratum measured.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []
        self.docs: list[Counter] = []
        self.lengths: list[int] = []
        self.df: Counter = Counter()
        self.avg_len: float = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            tokens = tokenize(c.text)
            tf = Counter(tokens)
            self.chunk_ids.append(c.chunk_id)
            self.texts.append(c.text)
            self.docs.append(tf)
            self.lengths.append(len(tokens))
            for term in tf:
                self.df[term] += 1
        self.avg_len = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.df.get(term, 0)
        # Floored at zero: without it, terms appearing in over half the corpus
        # get negative weight and actively push relevant chunks down.
        return max(math.log(1 + (n - df + 0.5) / (df + 0.5)), 0.0)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if not self.docs:
            return []
        q_terms = tokenize(query)
        scores: list[tuple[float, int]] = []
        for i, tf in enumerate(self.docs):
            score = 0.0
            for term in q_terms:
                freq = tf.get(term)
                if not freq:
                    continue
                norm = 1 - self.b + self.b * (self.lengths[i] / (self.avg_len or 1))
                score += self._idf(term) * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [
            Hit(self.chunk_ids[i], score, self.texts[i], "sparse")
            for score, i in scores[:k]
        ]


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_sets: list[list[Hit]],
    *,
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int = 10,
) -> list[Hit]:
    """RRF over several ranked lists.

    Rank-based rather than score-based because dense cosine and BM25 are not
    on comparable scales; normalising them would need per-corpus calibration
    that would itself become a confound.
    """
    weights = weights or [1.0] * len(result_sets)
    fused: dict[str, float] = defaultdict(float)
    texts: dict[str, str] = {}

    for hits, weight in zip(result_sets, weights):
        for rank, hit in enumerate(hits, start=1):
            fused[hit.chunk_id] += weight / (k + rank)
            texts.setdefault(hit.chunk_id, hit.text)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [Hit(cid, score, texts[cid], "fused") for cid, score in ordered[:top_k]]


class HybridRetriever:
    """Dense + sparse + RRF, with the arms individually switchable.

    Being able to run dense-only and sparse-only against the same dataset is
    what turns "we used hybrid search" into a measured claim.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        use_dense: bool = True,
        use_sparse: bool = True,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> None:
        self.dense = DenseIndex(embedder) if use_dense else None
        self.sparse = SparseIndex() if use_sparse else None
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def add(self, chunks: list[Chunk]) -> None:
        if self.dense:
            self.dense.add(chunks)
        if self.sparse:
            self.sparse.add(chunks)

    def search(self, query: str, k: int = 5, pool: int = 20) -> list[Hit]:
        sets, weights = [], []
        if self.dense:
            sets.append(self.dense.search(query, pool))
            weights.append(self.dense_weight)
        if self.sparse:
            sets.append(self.sparse.search(query, pool))
            weights.append(self.sparse_weight)

        if not sets:
            return []
        if len(sets) == 1:
            return sets[0][:k]
        return reciprocal_rank_fusion(sets, weights=weights, k=self.rrf_k, top_k=k)
