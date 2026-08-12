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

    #: Template-generated rather than human-authored. Reports built on
    #: synthetic items are scaffolding, not evidence about a real system.
    synthetic: bool = False

    @property
    def lang(self) -> str:
        """Language without the script subtag."""
        return self.language.split("-")[0]


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
