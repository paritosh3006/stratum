"""Dataset loading and validation.

An eval item is one question in one language, paired with the gold
answer and the chunk ids that should have been retrieved.

Items sharing a `parallel_id` are the same question in different
languages — that pairing is what makes cross-language attribution
possible, so it is worth enforcing early.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, Field

Slice = Literal[
    "parallel_core",
    "native_authored",
    "term_heavy",
    "unanswerable",
    "multi_hop",
]


class EvalItem(BaseModel):
    id: str
    language: str  # BCP-47-ish: "hi-Deva", "hi-Latn", "ta-Taml"
    slice: Slice
    query: str

    gold_answer: str | None = None
    gold_chunk_ids: list[str] = Field(default_factory=list)

    # Links the same question across languages. Required for attribution.
    parallel_id: str | None = None

    # Strings that must survive the pipeline verbatim.
    placeholders: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    numerals: list[str] = Field(default_factory=list)

    # True when the correct behaviour is to refuse.
    answerable: bool = True

    #: The language the user actually wrote the query in. Optional and
    #: independent of `language`, which stays the item's primary key for
    #: everything else (dataset grouping, S0 script_misdetection, oracle
    #: pairing). This is for use cases where query and answer language can
    #: legitimately differ — cross-lingual chat, voice — and defaults to
    #: unset so a dataset that predates it is untouched.
    query_language: str | None = None

    #: The language the *answer* is expected to come back in. Falls back to
    #: `query_language` via `effective_expected_answer_language` when unset,
    #: since same-language is the common case. When neither field is set,
    #: the output-language check simply never fires for this item — no
    #: dataset migration required.
    expected_answer_language: str | None = None

    #: Template-generated rather than human-authored. Reports built on
    #: synthetic items are scaffolding, not evidence about a real system.
    synthetic: bool = False

    @property
    def lang(self) -> str:
        """Language without the script subtag."""
        return self.language.split("-")[0]

    @property
    def effective_expected_answer_language(self) -> str | None:
        """`expected_answer_language`, defaulting to `query_language`.

        None when neither is set. Callers must treat None as "no
        expectation declared" rather than falling back to `language` —
        that field means something different (the item's own language
        variant) and conflating the two would make the check fire on every
        existing dataset the day this field shipped.
        """
        return self.expected_answer_language or self.query_language


class Dataset(BaseModel):
    items: list[EvalItem]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Dataset":
        path = Path(path)
        items: list[EvalItem] = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                items.append(EvalItem.model_validate(json.loads(line)))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{path}:{n} — invalid item: {exc}") from exc
        return cls(items=items)

    def __iter__(self) -> Iterator[EvalItem]:  # type: ignore[override]
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def languages(self) -> list[str]:
        return sorted({i.language for i in self.items})

    def by_language(self, language: str) -> list[EvalItem]:
        return [i for i in self.items if i.language == language]

    def parallel_map(self) -> dict[str, dict[str, EvalItem]]:
        """{parallel_id: {language: item}} for items that have a parallel_id."""
        out: dict[str, dict[str, EvalItem]] = defaultdict(dict)
        for item in self.items:
            if item.parallel_id:
                out[item.parallel_id][item.language] = item
        return dict(out)

    def validate_parallelism(self, baseline_language: str) -> list[str]:
        """Warnings for parallel groups missing a baseline counterpart.

        Attribution silently degrades without the baseline item, so this is
        surfaced as a warning rather than left to produce confusing numbers.
        """
        warnings: list[str] = []
        for pid, by_lang in self.parallel_map().items():
            if baseline_language not in by_lang:
                warnings.append(
                    f"parallel_id '{pid}' has no {baseline_language} item — "
                    f"attribution unavailable for {sorted(by_lang)}"
                )
        return warnings
