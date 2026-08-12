"""Adapters for the system under test.

stratum does not build RAG. It calls yours. Anything that can turn a query
into (answer, retrieved_chunk_ids) is a valid endpoint.

Attribution needs more than that: it needs to run your system with parts of
it bypassed. An endpoint declares which bypasses it supports, and stratum
reports honestly when a stage could not be isolated rather than guessing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class RagResponse(BaseModel):
    answer: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    detected_language: str | None = None
    refused: bool = False
    latency_ms: dict[str, float] = Field(default_factory=dict)
    raw: dict = Field(default_factory=dict)


@dataclass(frozen=True)
class Capabilities:
    """What the system under test allows stratum to bypass.

    Each flag unlocks one oracle pass. Missing flags are not fatal — the
    corresponding stage is reported as `not_isolated` rather than being
    silently folded into a neighbouring stage.
    """

    accepts_query_override: bool = False
    accepts_context_override: bool = False
    accepts_answer_override: bool = False

    @property
    def supported_passes(self) -> list[str]:
        out = ["standard"]
        if self.accepts_query_override:
            out.append("oracle_query")
        if self.accepts_context_override:
            out.append("oracle_context")
        if self.accepts_answer_override:
            out.append("oracle_answer")
        return out


@runtime_checkable
class Endpoint(Protocol):
    capabilities: Capabilities

    def query(
        self,
        text: str,
        language: str,
        *,
        context_chunk_ids: list[str] | None = None,
        answer_override: str | None = None,
    ) -> RagResponse: ...


class CallableEndpoint:
    """Wraps a plain Python function. The simplest way to get started.

    The function may accept the optional keyword arguments; if it does not,
    declare capabilities accordingly and only the standard pass will run.
    """

    def __init__(
        self,
        fn: Callable[..., RagResponse],
        capabilities: Capabilities | None = None,
    ) -> None:
        self._fn = fn
        self.capabilities = capabilities or Capabilities()

    def query(
        self,
        text: str,
        language: str,
        *,
        context_chunk_ids: list[str] | None = None,
        answer_override: str | None = None,
    ) -> RagResponse:
        started = time.perf_counter()
        kwargs: dict = {}
        if context_chunk_ids is not None:
            kwargs["context_chunk_ids"] = context_chunk_ids
        if answer_override is not None:
            kwargs["answer_override"] = answer_override

        resp = self._fn(text, language, **kwargs)
        resp.latency_ms.setdefault(
            "end_to_end", (time.perf_counter() - started) * 1000
        )
        return resp


class HttpEndpoint:
    """POSTs to a URL and maps the JSON back.

    Field names are configurable because no two RAG APIs agree on them.
    """

    def __init__(
        self,
        url: str,
        *,
        answer_field: str = "answer",
        chunks_field: str = "retrieved_chunk_ids",
        refused_field: str = "refused",
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        capabilities: Capabilities | None = None,
    ) -> None:
        self.url = url
        self.answer_field = answer_field
        self.chunks_field = chunks_field
        self.refused_field = refused_field
        self.headers = headers or {}
        self.timeout = timeout
        self.capabilities = capabilities or Capabilities()

    def query(
        self,
        text: str,
        language: str,
        *,
        context_chunk_ids: list[str] | None = None,
        answer_override: str | None = None,
    ) -> RagResponse:
        import json as _json
        import urllib.request

        body: dict = {"query": text, "language": language}
        if context_chunk_ids is not None:
            body["context_chunk_ids"] = context_chunk_ids
        if answer_override is not None:
            body["answer_override"] = answer_override

        req = urllib.request.Request(
            self.url,
            data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **self.headers},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = _json.loads(r.read())
        elapsed = (time.perf_counter() - started) * 1000

        return RagResponse(
            answer=data.get(self.answer_field, ""),
            retrieved_chunk_ids=data.get(self.chunks_field, []),
            detected_language=data.get("detected_language"),
            refused=bool(data.get(self.refused_field, False)),
            latency_ms={"end_to_end": elapsed},
            raw=data,
        )
