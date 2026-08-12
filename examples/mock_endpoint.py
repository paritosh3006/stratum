"""A deliberately broken RAG system, for testing the harness itself.

Every bug here is one seen in real multilingual pipelines:

  - Tamil translates placeholder identifiers  -> downstream KeyError
  - Tamil renders "premium" off-glossary      -> terminology drift
  - Roman-script Hindi misdetected as English -> wrong retrieval path
  - Large numerals dropped in Tamil           -> factual corruption
  - One Hindi answerable question refused     -> over-refusal

Crucially the bugs sit at *different stages*: hi-Latn breaks at input
handling, ta-Taml breaks at output rendering. A single score would rank
them similarly and tell you nothing. The cascade must separate them —
that is what this mock exists to verify.

It supports all three oracle passes, so attribution can be exercised
end to end.
"""

from __future__ import annotations

import random

from stratum import CallableEndpoint, RagResponse
from stratum.endpoint import Capabilities

random.seed(7)


def _retrieve(query: str, language: str) -> list[str]:
    q = query.lower()
    misrouted = language == "hi-Latn"   # routed to the English path

    if "knee" in q or "घुटना" in query or "முழங்கால்" in query:
        return ["c_901", "c_042"] if misrouted else ["c_042", "c_043", "c_100"]
    if "premium" in q or "प्रीमियम" in query or "பிரீமியம்" in query:
        return ["c_902", "c_903"] if misrouted else ["c_011", "c_012"]
    if "emi" in q:
        return ["c_077", "c_078"]
    if "dental" in q or "दांत" in query or "பல்" in query:
        return ["c_055"]
    return ["c_800", "c_801"]


def mock_rag(
    query: str,
    language: str,
    *,
    context_chunk_ids: list[str] | None = None,
    answer_override: str | None = None,
) -> RagResponse:
    # oracle_context: retrieval is bypassed with known-good chunks.
    chunks = context_chunk_ids if context_chunk_ids is not None else _retrieve(query, language)

    if "Brazil" in query or "பிரேசில்" in query:
        return RagResponse(
            answer="This is not covered by the policy documents.",
            retrieved_chunk_ids=[],
            detected_language=language,
            refused=True,
        )

    # oracle_answer: generation is bypassed; only rendering runs.
    answer = answer_override if answer_override is not None else query

    # -- rendering stage, where the Tamil bugs live ----------------------
    if language == "ta-Taml":
        answer = answer.replace("{amount}", "{தொகை}").replace("{date}", "{தேதி}")
        answer = answer.replace("premium", "கட்டணம்")
        answer = answer.replace("500000", "50000")
    elif language == "hi-Deva":
        answer = answer.replace("premium", "प्रीमियम")

    # -- input stage, where the Roman-Hindi bug lives --------------------
    detected = "en" if language == "hi-Latn" else language

    # Over-refusal disappears once the query is repaired, so the cascade
    # should attribute it upstream rather than to generation.
    refused = language == "hi-Deva" and "दांतों" in query

    return RagResponse(
        answer=answer,
        retrieved_chunk_ids=chunks,
        detected_language=detected,
        refused=refused,
        latency_ms={"end_to_end": random.uniform(900, 2100)},
    )


endpoint = CallableEndpoint(
    mock_rag,
    capabilities=Capabilities(
        accepts_query_override=True,
        accepts_context_override=True,
        accepts_answer_override=True,
    ),
)
