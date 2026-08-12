"""Roman -> Devanagari transliteration — S1a.

Only exercised for `hi-Latn` queries: script detection has already separated
those from `en`, so this stage never sees anything but Hindi typed in Latin
letters that needs to become Devanagari before translation.

The dataset's hi-Latn queries are code-mixed by design — "room rent ki
limit kya hai" says "room", "rent" and "limit" in English mid-sentence, the
way real Hinglish insurance queries do. Running every token through a Hindi
transliterator blindly does not know that: it turns "room" into nonsense
Devanagari as readily as it turns "kya" into क्या, and the corpus can match
neither the mangled English nor a translation of it. Measured on this
dataset that was a net *regression* against doing nothing — mangled
"knee"/"surgery"/"cover" lost the verbatim overlap the untouched English
words used to get, for no compensating gain.

`SelectiveTransliterator` below is the fix actually wired into the default
pipeline: it only converts tokens already flagged as Hindi markers by script
detection (script.py's `HI_LATN_MARKERS`), and leaves everything else —
almost always the English content words — untouched. `RuleBasedTransliterator`
and `IndicXlitTransliterator` remain available unwrapped for callers who
already know their input is pure Hindi.
"""

from __future__ import annotations

from typing import Protocol

from .script import HI_LATN_MARKERS


class Transliterator(Protocol):
    name: str

    def transliterate(self, text: str) -> str: ...


# Longest-match-first syllable tables. Not linguistics — a working stub that
# gets common syllables into the right script so downstream sparse/dense
# matching has something to work with. It will not round-trip cleanly and
# schwa deletion (the silent trailing "a" in most spoken Hindi words) is not
# handled at all; every consonant keeps its inherent vowel.
_CONSONANTS: list[tuple[str, str]] = [
    ("chh", "छ"), ("shh", "ष"), ("gy", "ज्ञ"), ("ksh", "क्ष"),
    ("kh", "ख"), ("gh", "घ"), ("ng", "ङ"), ("ch", "च"), ("jh", "झ"),
    ("ny", "ञ"), ("th", "थ"), ("dh", "ध"), ("ph", "फ"), ("bh", "भ"),
    ("sh", "श"),
    ("k", "क"), ("g", "ग"), ("j", "ज"), ("t", "त"), ("d", "द"),
    ("n", "न"), ("p", "प"), ("b", "ब"), ("m", "म"), ("y", "य"),
    ("r", "र"), ("l", "ल"), ("v", "व"), ("w", "व"), ("s", "स"),
    ("h", "ह"), ("f", "फ़"), ("z", "ज़"),
]
_MATRAS: list[tuple[str, str]] = [
    ("aa", "ा"), ("ee", "ी"), ("oo", "ू"), ("ai", "ै"), ("au", "ौ"),
    ("i", "ि"), ("u", "ु"), ("e", "े"), ("o", "ो"),
    # A trailing bare "a" is the same inherent vowel a consonant already
    # carries with no matra at all — matching it here (to an empty matra)
    # stops it from being re-read as a fresh syllable via _VOWELS_INITIAL,
    # which produced a spurious trailing अ on words like "hoga".
    ("a", ""),
]
_VOWELS_INITIAL: list[tuple[str, str]] = [
    ("aa", "आ"), ("ai", "ऐ"), ("au", "औ"), ("ee", "ई"), ("oo", "ऊ"),
    ("a", "अ"), ("i", "इ"), ("u", "उ"), ("e", "ए"), ("o", "ओ"),
]


class RuleBasedTransliterator:
    """Greedy syllable-table transliteration. No model, no download.

    Segments each word into consonant/vowel chunks against the tables above,
    longest match first. It gets common open syllables right and gets
    consonant clusters, aspirated pairs and code-mixed English wrong — like
    HashingEmbedder, it exists so the pipeline has an offline leg to stand
    on, not because it is a good transliterator.
    """

    name = "rule-based"

    def transliterate(self, text: str) -> str:
        return " ".join(self._word(w) for w in text.split())

    def _word(self, word: str) -> str:
        w = word.lower()
        out: list[str] = []
        i = 0
        while i < len(w):
            cons = next((c for c, _ in _CONSONANTS if w.startswith(c, i)), None)
            if cons:
                dev_c = dict(_CONSONANTS)[cons]
                i += len(cons)
                vowel = next((v for v, _ in _MATRAS if w.startswith(v, i)), None)
                if vowel:
                    out.append(dev_c + dict(_MATRAS)[vowel])
                    i += len(vowel)
                else:
                    out.append(dev_c)  # inherent vowel, no matra written
                continue
            vow = next((v for v, _ in _VOWELS_INITIAL if w.startswith(v, i)), None)
            if vow:
                out.append(dict(_VOWELS_INITIAL)[vow])
                i += len(vow)
                continue
            out.append(w[i])  # unmapped character (digits, punctuation) passes through
            i += 1
        return "".join(out)


class SelectiveTransliterator:
    """Transliterates only tokens already known to be Hindi; leaves the rest.

    Wraps another transliterator and gates it word-by-word against a
    vocabulary — by default `script.HI_LATN_MARKERS`, the same closed-class
    word list script detection uses to call a query hi-Latn in the first
    place. Declining to transliterate an unrecognised token is the
    conservative choice: it stays as plain Latin text, exactly as useless to
    Devanagari-keyed matching as it was before this pipeline existed, rather
    than being actively converted into something that matches nothing at
    all. This is what `build_query_pipeline` wires in by default.
    """

    def __init__(
        self, inner: Transliterator, vocabulary: frozenset[str] = HI_LATN_MARKERS
    ) -> None:
        self._inner = inner
        self._vocabulary = vocabulary
        self.name = f"selective({inner.name})"

    def transliterate(self, text: str) -> str:
        return " ".join(
            self._inner.transliterate(w) if w.lower() in self._vocabulary else w
            for w in text.split()
        )


class IndicXlitTransliterator:
    """AI4Bharat IndicXlit. Downloads model weights on first use."""

    name = "indicxlit"

    def __init__(self, beam_width: int = 4) -> None:
        try:
            from ai4bharat.transliteration import XlitEngine
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "IndicXlit needs ai4bharat-transliteration — "
                'pip install -e "examples/reference_system[indic]"'
            ) from exc
        self._engine = XlitEngine("hi", beam_width=beam_width, rescore=False)

    def transliterate(self, text: str) -> str:
        out = self._engine.translit_sentence(text, lang_code="hi")
        return out if isinstance(out, str) else out.get("hi", text)


def get_transliterator(name: str = "rule-based") -> Transliterator:
    if name in {"indicxlit", "ai4bharat"}:
        return IndicXlitTransliterator()
    if name == "rule-based":
        return RuleBasedTransliterator()
    raise ValueError(f"unknown transliterator: {name}")
