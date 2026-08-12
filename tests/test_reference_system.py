"""Tests for the reference system.

The critical one is `test_oracle_hooks_are_honoured`. Stratum cannot detect a
system that declares a capability and ignores it — the rung would read zero and
the loss would be silently absorbed by a neighbour. That contract has to be
verified here, on this side of the boundary.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from reference_system.ingest.chunk import (  # noqa: E402
    build_corpus, chunk_text, stable_chunk_id, split_sentences,
)
from reference_system.pipeline.system import ReferenceSystem, SystemConfig  # noqa: E402
from reference_system.retrieval.hybrid import (  # noqa: E402
    HybridRetriever, reciprocal_rank_fusion, Hit, tokenize,
)
from reference_system.retrieval.embedder import HashingEmbedder  # noqa: E402

CORPUS = ROOT / "examples" / "reference_system" / "corpus"


@pytest.fixture(scope="module")
def system():
    return ReferenceSystem(SystemConfig(corpus_dir=CORPUS, min_span_score=0.20))


class TestChunking:
    def test_ids_are_content_addressed(self):
        # Same text must yield the same id regardless of position, or every
        # gold reference in the dataset breaks on re-ingestion.
        assert stable_chunk_id("doc", "hello") == stable_chunk_id("doc", "hello")
        assert stable_chunk_id("doc", "hello") != stable_chunk_id("doc", "world")

    def test_id_survives_surrounding_whitespace(self):
        assert stable_chunk_id("d", " text ") == stable_chunk_id("d", "text")

    def test_readme_is_not_ingested(self):
        docs = {c.doc_id for c in build_corpus(CORPUS, target_tokens=180)}
        assert "README" not in docs

    def test_oversized_paragraph_is_split(self):
        para = " ".join(["This is a sentence about policy terms."] * 200)
        chunks = chunk_text(para, target_tokens=100)
        assert len(chunks) > 1

    def test_danda_ends_a_sentence(self):
        assert len(split_sentences("यह पहला वाक्य है। यह दूसरा है।")) == 2


class TestRetrieval:
    def test_rrf_rewards_agreement(self):
        # A chunk ranked well by both arms should beat one ranked top by only one.
        a = [Hit("shared", 1.0, "x"), Hit("only_a", 0.9, "y")]
        b = [Hit("only_b", 1.0, "z"), Hit("shared", 0.8, "x")]
        assert reciprocal_rank_fusion([a, b])[0].chunk_id == "shared"

    def test_tokenizer_keeps_devanagari_together(self):
        assert "प्रीमियम" in tokenize("प्रीमियम कितना है")

    def test_sparse_finds_exact_identifiers(self):
        # Entity recall is the reason the sparse arm exists.
        chunks = build_corpus(CORPUS, target_tokens=180)
        r = HybridRetriever(HashingEmbedder(), use_dense=False)
        r.add(chunks)
        hits = r.search("cumulative bonus claim-free", k=3)
        assert any("cumulative bonus" in h.text.lower() for h in hits)

    def test_empty_index_returns_nothing(self):
        assert HybridRetriever(HashingEmbedder()).search("anything") == []


class TestAnswering:
    def test_answers_a_covered_question(self, system):
        r = system.answer("What is the grace period for renewal premium?", "en")
        assert not r.refused and "30 days" in r.answer

    def test_refuses_outside_the_corpus(self, system):
        # Stopword overlap alone used to clear the relevance floor here.
        assert system.answer("What is the capital of Brazil?", "en").refused

    def test_returns_chunk_ids(self, system):
        r = system.answer("What is the room rent limit?", "en")
        assert r.retrieved_chunk_ids and all(
            c in system.by_id for c in r.retrieved_chunk_ids
        )


class TestOracleHooks:
    def test_context_override_is_honoured(self, system):
        gold = [system.chunks[3].chunk_id]
        r = system.answer("anything at all", "en", context_chunk_ids=gold)
        assert r.retrieved_chunk_ids == gold

    def test_answer_override_is_honoured(self, system):
        r = system.answer("q", "en", context_chunk_ids=[system.chunks[0].chunk_id],
                          answer_override="EXACT TEXT")
        assert r.answer == "EXACT TEXT" and not r.refused

    def test_declared_capabilities_match_reality(self):
        from reference_system.pipeline.system import build_endpoint
        ep = build_endpoint(SystemConfig(corpus_dir=CORPUS, min_span_score=0.20))
        caps = ep.capabilities
        # Every declared capability is exercised above; declaring one that is
        # ignored is the single failure Stratum cannot detect for itself.
        assert caps.accepts_context_override and caps.accepts_answer_override
