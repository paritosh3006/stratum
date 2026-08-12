"""Devanagari -> Roman transliteration for rendering hi-Latn answers — S4.

The mirror image of query/transliterate.py's Roman -> Devanagari direction:
this is the last step for hi-Latn rendering, run after translation and
glossary enforcement have already produced Devanagari text.

Character-level, following the abugida rule set directly rather than
reusing query/transliterate.py's Roman-keyed tables — those are indexed by
Roman syllable and don't cover retroflex consonants (ट ठ ड ढ ण) or
conjuncts, both common in real Devanagari prose (this corpus's own English
loanwords glossed into Hindi produce plenty of both). A consonant with no
following vowel sign keeps its inherent "a" unless followed by a virama, in
which case it takes no vowel at all (a cluster continues into the next
consonant) — schwa deletion (the silent trailing "a" native speakers drop
in speech, e.g. "प्रीमियम" said as "premium" not "premiyama") is not
handled, matching the equivalent limitation documented on the forward
direction.
"""

from __future__ import annotations

from typing import Protocol


class Romanizer(Protocol):
    name: str

    def romanize(self, text: str) -> str: ...


_CONSONANTS: dict[str, str] = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ड़": "r", "ढ़": "rh", "फ़": "f", "ज़": "z", "क़": "q", "ग़": "gh",
}
_MATRAS: dict[str, str] = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    # Candra vowel signs: not native to Sanskrit-derived spelling, added for
    # English loanwords written in Devanagari — "पॉलिसी" ("policy") needs
    # ॉ, and this corpus's own glossary is full of exactly such loanwords.
    "ॅ": "e", "ॉ": "o",
}
_VOWELS: dict[str, str] = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ऍ": "e", "ऑ": "o",
}
_TRAILING: dict[str, str] = {"ं": "n", "ँ": "n", "ः": "h"}
_VIRAMA = "्"
_NUKTA = "़"


class TableRomanizer:
    """Character-by-character Devanagari -> Roman via the tables above. No
    model, no download.

    Everything outside these tables — Latin text, digits, the render
    pipeline's own placeholder/numeral sentinels, punctuation — passes
    through unchanged, so mixed-script input (an English loanword left
    untranslated inside an otherwise Devanagari sentence) survives rather
    than being silently dropped.
    """

    name = "table-romanizer"

    def romanize(self, text: str) -> str:
        out: list[str] = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch in _CONSONANTS:
                roman = _CONSONANTS[ch]
                nxt = text[i + 1] if i + 1 < n else ""
                if nxt == _VIRAMA:
                    out.append(roman)  # no inherent vowel; cluster continues
                    i += 2
                    continue
                if nxt == _NUKTA:
                    i += 1  # nukta already folded into the consonant table
                    continue
                if nxt in _MATRAS:
                    out.append(roman + _MATRAS[nxt])
                    i += 2
                    continue
                out.append(roman + "a")  # inherent vowel
                i += 1
                continue
            if ch in _VOWELS:
                out.append(_VOWELS[ch])
                i += 1
                continue
            if ch in _TRAILING:
                out.append(_TRAILING[ch])
                i += 1
                continue
            if ch in (_VIRAMA, _NUKTA):
                i += 1  # stray mark with nothing to attach to
                continue
            out.append(ch)  # passthrough: Latin, digits, sentinels, punctuation
            i += 1
        return "".join(out)


class IndicTransliterationRomanizer:
    """`indic_transliteration`'s Devanagari -> ITRANS scheme conversion.

    Deliberately not IndicXlit here: IndicXlit (used elsewhere in this
    codebase for the Roman -> Devanagari direction, query/transliterate.py)
    is a phonetic-typing engine and is Roman-input-only — it has no
    Devanagari -> Roman direction to call. `indic_transliteration` is a
    genuinely bidirectional, rule-based scheme library, which is the
    correct real tool for this specific direction rather than a bigger
    model misapplied to a direction it doesn't support.
    """

    name = "indic-transliteration"

    def __init__(self) -> None:
        try:
            from indic_transliteration import sanscript
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Devanagari->Roman needs indic-transliteration — "
                'pip install -e "examples/reference_system[indic]"'
            ) from exc
        self._sanscript = sanscript

    def romanize(self, text: str) -> str:
        return self._sanscript.transliterate(
            text, self._sanscript.DEVANAGARI, self._sanscript.ITRANS
        )


def get_romanizer(name: str = "table-romanizer") -> Romanizer:
    if name in {"indic-transliteration", "sanscript"}:
        return IndicTransliterationRomanizer()
    if name == "table-romanizer":
        return TableRomanizer()
    raise ValueError(f"unknown romanizer: {name}")
