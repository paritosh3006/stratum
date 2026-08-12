"""S0 + S1: script detection, transliteration, translation — one normalized
English query out.

    hi-Deva  -> translate                          -> en
    hi-Latn  -> transliterate -> Deva -> translate  -> en
    en       -> pass through                        -> en

This is the stage the reference system originally shipped without. Without
it, a Hindi query went straight to an English-only sparse/hashing index,
matched nothing, fell below the relevance floor, and the entire loss landed
on the S0+S1 rung by construction — not because retrieval or generation
were bad, but because nothing upstream of them ever ran. See
examples/reference_system/README.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .script import EN, HI_LATN, ScriptDetector, get_script_detector
from .transliterate import SelectiveTransliterator, Transliterator, get_transliterator
from .translate import Translator, get_translator


@dataclass
class NormalizedQuery:
    original: str
    detected_script: str
    normalized: str
    steps: list[str] = field(default_factory=list)


class QueryPipeline:
    def __init__(
        self,
        script_detector: ScriptDetector,
        transliterator: Transliterator,
        translator: Translator,
    ) -> None:
        self.script_detector = script_detector
        self.transliterator = transliterator
        self.translator = translator

    def normalize(self, query: str) -> NormalizedQuery:
        script = self.script_detector.detect(query)

        if script == EN:
            return NormalizedQuery(query, script, query, [])

        text = query
        steps: list[str] = []
        if script == HI_LATN:
            text = self.transliterator.transliterate(text)
            steps.append(f"transliterate:{self.transliterator.name}")

        text = self.translator.translate(text, source_lang="hi")
        steps.append(f"translate:{self.translator.name}")

        return NormalizedQuery(query, script, text, steps)


def build_query_pipeline(
    script_detector: str = "heuristic",
    transliterator: str = "rule-based",
    translator: str = "lexicon",
) -> QueryPipeline:
    # Gated regardless of backend: code-mixed hi-Latn queries carry English
    # content words verbatim, and transliterating those is what caused a
    # measured regression (see query/transliterate.py) — the fix is at the
    # pipeline policy level, not specific to the rule-based table.
    gated = SelectiveTransliterator(get_transliterator(transliterator))
    return QueryPipeline(
        get_script_detector(script_detector),
        gated,
        get_translator(translator),
    )
