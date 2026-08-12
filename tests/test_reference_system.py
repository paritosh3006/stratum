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
from reference_system.query.script import (  # noqa: E402
    HeuristicScriptDetector, EN, HI_DEVA, HI_LATN,
)
from reference_system.query.transliterate import RuleBasedTransliterator  # noqa: E402
from reference_system.query.translate import LexiconTranslator  # noqa: E402
from reference_system.query.pipeline import build_query_pipeline  # noqa: E402

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

    def test_span_selection_prefers_answer_over_definition(self, system):
        # Both the "pre-existing disease" definition (48 months, singular
        # "disease") and the actual waiting-period sentence (36 months,
        # plural "diseases", "covered" not "cover") were retrieved for this
        # query; without stemmed matching, exact-token overlap favoured the
        # definition purely on "disease" matching singular-for-singular.
        r = system.answer(
            "How long must I wait for a pre-existing disease to be covered?", "en"
        )
        assert "36 months" in r.answer
        assert "48 months" not in r.answer


class TestScriptDetection:
    def test_devanagari_is_exact(self):
        d = HeuristicScriptDetector()
        assert d.detect("घुटना बदलने की सर्जरी शामिल है क्या") == HI_DEVA

    def test_plain_english(self):
        d = HeuristicScriptDetector()
        assert d.detect("What is the grace period for renewal premium?") == EN

    def test_romanized_hindi(self):
        d = HeuristicScriptDetector()
        assert d.detect("mera knee replacement surgery cover hoga kya") == HI_LATN

    def test_empty_text_is_english(self):
        assert HeuristicScriptDetector().detect("") == EN


class TestTransliteration:
    def test_produces_devanagari(self):
        out = RuleBasedTransliterator().transliterate("kitna hai")
        assert any("ऀ" <= ch <= "ॿ" for ch in out)

    def test_deterministic(self):
        t = RuleBasedTransliterator()
        assert t.transliterate("kaise kare") == t.transliterate("kaise kare")

    def test_digits_survive_untouched(self):
        # The syllable table maps only Latin letters, so numerals must never
        # be rewritten — regression guard for the numeral_integrity gap
        # investigated in examples/reference_system/README.md, which turned
        # out not to be digit corruption but is worth locking in either way.
        assert RuleBasedTransliterator().transliterate("45 saal 500000") == "45 साल 500000"


class TestTranslation:
    def test_known_words_translate(self):
        with pytest.warns(UserWarning, match="train-on-test"):
            translator = LexiconTranslator()
        out = translator.translate("घुटना बदलने की सर्जरी शामिल है क्या")
        assert "knee" in out and "surgery" in out

    def test_unknown_tokens_pass_through(self):
        # No entry for a made-up token: it must survive unchanged rather
        # than being dropped or raising.
        with pytest.warns(UserWarning, match="train-on-test"):
            translator = LexiconTranslator()
        out = translator.translate("क्या xyz123 है")
        assert "xyz123" in out

    def test_warns_that_its_lexicon_is_train_on_test(self):
        # The dictionary's vocabulary was read from this eval dataset's own
        # queries (see build_dataset.py). Anyone who runs stratum with this
        # backend active should not learn that from the README alone.
        with pytest.warns(UserWarning, match="train-on-test"):
            LexiconTranslator()


class TestQueryPipeline:
    def test_english_passes_through_untouched(self):
        pipeline = build_query_pipeline()
        result = pipeline.normalize("What is the grace period?")
        assert result.detected_script == EN
        assert result.normalized == "What is the grace period?"
        assert result.steps == []

    def test_devanagari_is_translated_only(self):
        pipeline = build_query_pipeline()
        result = pipeline.normalize("पॉलिसी में कमरे का किराया कितना देय है?")
        assert result.detected_script == HI_DEVA
        assert result.steps == ["translate:lexicon"]
        assert "room" in result.normalized and "rent" in result.normalized

    def test_roman_hindi_is_transliterated_then_translated(self):
        pipeline = build_query_pipeline()
        result = pipeline.normalize("mera knee replacement surgery cover hoga kya")
        assert result.detected_script == HI_LATN
        assert result.steps == ["transliterate:selective(rule-based)", "translate:lexicon"]

    def test_roman_hindi_leaves_english_content_words_untouched(self):
        # The regression this guards against: naively transliterating every
        # token turned "knee"/"surgery"/"cover" into unmatched Devanagari,
        # which measurably scored worse than doing nothing at all.
        pipeline = build_query_pipeline()
        result = pipeline.normalize("mera knee replacement surgery cover hoga kya")
        for word in ("knee", "replacement", "surgery", "cover"):
            assert word in result.normalized

    def test_devanagari_query_now_retrieves_the_right_chunk(self, system):
        # Before this pipeline existed, every Hindi query was matched
        # against the English index verbatim and retrieved nothing relevant.
        r = system.answer("पॉलिसी में कमरे का किराया कितना देय है?", "hi-Deva")
        assert not r.refused
        assert "room rent" in r.answer.lower()
        assert r.raw.get("query_pipeline_steps") == ["translate:lexicon"]

    def test_detected_language_reflects_actual_script(self, system):
        r = system.answer("पॉलिसी में कमरे का किराया कितना देय है?", "hi-Deva")
        assert r.detected_language == HI_DEVA

    def test_numerals_survive_hi_latn_normalization(self):
        # ASCII digits must reach the retriever unchanged. numeral_integrity
        # failures for hi-Latn turned out to trace to over-refusal, not to
        # this — see README.md's numeral integrity note — but the pipeline
        # should still guarantee it.
        pipeline = build_query_pipeline()
        result = pipeline.normalize("45 saal aur 500000 sum insured pe premium kitna hai")
        assert "45" in result.normalized.split()
        assert "500000" in result.normalized.split()


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
