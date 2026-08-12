"""S4: mask -> translate (en -> hi-Deva) -> glossary enforce -> [hi-Latn:
romanize -> glossary enforce again] -> unmask.

One normalized English answer span in (from S3); one localized answer out.
`en` items pass through untouched — there is nothing to render.

Glossary enforcement runs twice for hi-Latn, not once. The Devanagari-stage
pass makes sure the *translated* answer uses approved Hindi terminology
before romanization touches it; the post-romanization pass fixes up terms
whose approved hi-Latn form is the plain English spelling ("policy", not
whatever `TableRomanizer` produces for "पॉलिसी"). Skipping the second pass
would leave every English-loanword glossary term romanized into something
that looks like Hindi but reads as neither language correctly — this
dataset's own hi-Latn queries keep exactly these words in English for a
reason (see query/transliterate.py's SelectiveTransliterator docstring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import glossary as glossary_module
from .mask import mask, unmask
from .romanize import Romanizer, get_romanizer
from .translate_en import RenderTranslator, get_render_translator

_RENDERABLE = ("hi-Deva", "hi-Latn")


@dataclass
class RenderedAnswer:
    original: str
    language: str
    text: str
    steps: list[str] = field(default_factory=list)


class RenderPipeline:
    def __init__(
        self,
        translator: RenderTranslator,
        romanizer: Romanizer,
        enforce: Callable[[str, str, str, str], str] = glossary_module.enforce,
    ) -> None:
        self.translator = translator
        self.romanizer = romanizer
        self.enforce = enforce

    def render(self, answer_en: str, language: str, query: str = "") -> RenderedAnswer:
        """`query` is what the user actually asked, in whatever language —
        needed only for deciding which glossary terms are in scope (see
        glossary.enforce's docstring); rendering itself only ever touches
        `answer_en`."""
        if language not in _RENDERABLE or not answer_en:
            return RenderedAnswer(answer_en, language, answer_en, [])

        steps: list[str] = []
        masked = mask(answer_en)

        devanagari = self.translator.translate(masked.text, target_lang="hi")
        steps.append(f"translate:{self.translator.name}")

        devanagari = self.enforce(answer_en, devanagari, "hi-Deva", query)
        steps.append("glossary:hi-Deva")

        rendered = devanagari
        if language == "hi-Latn":
            rendered = self.romanizer.romanize(devanagari)
            steps.append(f"romanize:{self.romanizer.name}")
            rendered = self.enforce(answer_en, rendered, "hi-Latn", query)
            steps.append("glossary:hi-Latn")

        final = unmask(rendered, masked.restore)
        return RenderedAnswer(answer_en, language, final, steps)


def build_render_pipeline(
    translator: str = "lexicon-en-hi",
    romanizer: str = "table-romanizer",
) -> RenderPipeline:
    return RenderPipeline(get_render_translator(translator), get_romanizer(romanizer))
