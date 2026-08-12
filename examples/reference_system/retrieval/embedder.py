"""Embedding backends.

Two implementations, deliberately:

`BGEM3Embedder` is the real one — multilingual, placing English, Devanagari
and Roman-Hindi in a single space so a Hindi query can retrieve an English
chunk without translating first. That property is the whole reason for
choosing BGE-M3, and it is one of the things worth measuring: translate-then-
retrieve versus retrieve-directly is a real architectural fork, and Stratum
can tell you which wins per language.

`HashingEmbedder` is a deterministic character-ngram encoder with no model
download. It exists so the pipeline, the index, and the tests run anywhere in
under a second. It is a poor retriever and is never the default for a real
run — but a reference system nobody can execute is not much of a reference.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence


class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


def _l2_normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class HashingEmbedder:
    """Character-ngram hashing into a fixed-width vector.

    Character ngrams rather than words: they degrade far more gracefully
    across scripts, and word tokenisation for Devanagari and Tamil would need
    its own dependency for something that is only a test fixture.
    """

    name = "hashing-ngram"

    def __init__(self, dim: int = 384, ngram: int = 3) -> None:
        self.dim = dim
        self.ngram = ngram

    def _features(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text.lower().strip())
        grams = [text[i : i + self.ngram] for i in range(max(len(text) - self.ngram + 1, 1))]
        return grams + text.split()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for feature in self._features(text):
                h = int(hashlib.md5(feature.encode("utf-8")).hexdigest()[:8], 16)
                vec[h % self.dim] += 1.0
            out.append(_l2_normalise(vec))
        return out


class BGEM3Embedder:
    """BAAI/bge-m3 via sentence-transformers. Downloads ~2GB on first use."""

    name = "bge-m3"

    def __init__(self, device: str | None = None, batch_size: int = 16) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "BGE-M3 needs sentence-transformers — "
                'pip install -e "examples/reference_system[models]"'
            ) from exc
        self._model = SentenceTransformer("BAAI/bge-m3", device=device)
        self.dim = self._model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]


def get_embedder(name: str = "hashing") -> Embedder:
    if name in {"bge", "bge-m3"}:
        return BGEM3Embedder()
    if name == "hashing":
        return HashingEmbedder()
    raise ValueError(f"unknown embedder: {name}")
