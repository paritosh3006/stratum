"""The reference system, wired end to end and exposed to Stratum.

The oracle hooks are the part that matters for attribution. Each one skips a
stage using known-good input:

    context_chunk_ids  -> retrieval is skipped, these chunks are used
    answer_override    -> answering is skipped, this text goes to rendering

Declaring a capability the system does not honour is the one failure Stratum
cannot detect: the rung would read zero and the loss would be absorbed
silently by a neighbour. So the hooks are wired first and the capability flags
are set last.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from stratum import CallableEndpoint, RagResponse
from stratum.endpoint import Capabilities

from .extractive import Span, select_span
from ..ingest.chunk import Chunk, build_corpus
from ..query.pipeline import QueryPipeline, build_query_pipeline
from ..render.pipeline import RenderPipeline, build_render_pipeline
from ..retrieval.embedder import get_embedder
from ..retrieval.hybrid import Hit, HybridRetriever


@dataclass
class SystemConfig:
    corpus_dir: Path
    embedder: str = "hashing"
    use_dense: bool = True
    use_sparse: bool = True
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    top_k: int = 5
    min_span_score: float = 0.12
    #: Smaller than a typical 512 because retrieval granularity is what is
    #: being measured: oversized chunks inflate recall by making every chunk
    #: contain a bit of everything.
    target_tokens: int = 180
    #: S0 + S1. Defaults are the no-download stubs so the system runs
    #: offline; swap in "fasttext-lid218e" / "indicxlit" / "indictrans2"
    #: once `examples/reference_system[indic]` is installed.
    script_detector: str = "heuristic"
    transliterator: str = "rule-based"
    translator: str = "lexicon"
    #: S4. Same no-download-stub default; swap in "indictrans2-en-indic" /
    #: "indic-transliteration" once the indic extra is installed.
    render_translator: str = "lexicon-en-hi"
    render_romanizer: str = "table-romanizer"


class ReferenceSystem:
    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.chunks: list[Chunk] = build_corpus(
            config.corpus_dir, target_tokens=config.target_tokens
        )
        if not self.chunks:
            raise RuntimeError(f"no documents ingested from {config.corpus_dir}")

        self.by_id: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        self.query_pipeline: QueryPipeline = build_query_pipeline(
            config.script_detector, config.transliterator, config.translator
        )
        self.render_pipeline: RenderPipeline = build_render_pipeline(
            config.render_translator, config.render_romanizer
        )
        self.retriever = HybridRetriever(
            get_embedder(config.embedder),
            use_dense=config.use_dense,
            use_sparse=config.use_sparse,
            dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight,
        )
        self.retriever.add(self.chunks)

    # ------------------------------------------------------------------
    def answer(
        self,
        query: str,
        language: str,
        *,
        context_chunk_ids: list[str] | None = None,
        answer_override: str | None = None,
    ) -> RagResponse:
        # -- S0 + S1: script detection, transliteration, translation ------
        # Run unconditionally so `detected_language` reflects what the
        # system actually saw, even on oracle passes where the harness
        # already handed over an English query (script detection then
        # correctly reports "en" and normalization is a no-op).
        normalized = self.query_pipeline.normalize(query)

        # -- S2: retrieval, or the oracle bypass --------------------------
        if context_chunk_ids is not None:
            hits = [
                Hit(cid, 1.0, self.by_id[cid].text, "oracle")
                for cid in context_chunk_ids
                if cid in self.by_id
            ]
        else:
            hits = self.retriever.search(normalized.normalized, k=self.config.top_k)

        retrieved = [h.chunk_id for h in hits]

        # -- S3: answering, or the oracle bypass --------------------------
        if answer_override is not None:
            answer, refused = answer_override, False
        else:
            span: Span | None = select_span(
                normalized.normalized,
                hits,
                min_score=self.config.min_span_score,
                idf=self.retriever.sparse._idf if self.retriever.sparse else None,
            )
            if span is None:
                return RagResponse(
                    answer="",
                    retrieved_chunk_ids=retrieved,
                    detected_language=normalized.detected_script,
                    refused=True,
                    raw={
                        "reason": "no span above relevance floor",
                        "normalized_query": normalized.normalized,
                        "query_pipeline_steps": normalized.steps,
                    },
                )
            answer, refused = span.text, False

        # -- S4: rendering --------------------------------------------------
        # Runs on whichever answer text S3 produced above — the system's own
        # extracted span on standard/oracle_query/oracle_context passes, or
        # the gold answer_override on oracle_answer/baseline. That's the
        # point of the ladder: oracle_answer repairs S0..S3 and leaves S4 to
        # the system's own rendering, so this call has to happen either way.
        rendered = self.render_pipeline.render(answer, language, query=query)

        return RagResponse(
            answer=rendered.text,
            retrieved_chunk_ids=retrieved,
            detected_language=normalized.detected_script,
            refused=refused,
            raw={
                "n_hits": len(hits),
                "normalized_query": normalized.normalized,
                "query_pipeline_steps": normalized.steps,
                "render_steps": rendered.steps,
            },
        )


def build_endpoint(config: SystemConfig) -> CallableEndpoint:
    system = ReferenceSystem(config)
    return CallableEndpoint(
        system.answer,
        capabilities=Capabilities(
            accepts_query_override=True,      # the harness swaps the query text
            accepts_context_override=True,    # honoured above
            accepts_answer_override=True,     # honoured above
        ),
    )


#: Default endpoint for `stratum run --endpoint .../system.py:endpoint`
endpoint = build_endpoint(
    SystemConfig(corpus_dir=Path(__file__).resolve().parents[1] / "corpus")
)
