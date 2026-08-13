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

#: A single short vowel letter (a/i/u — never a digraph like "ai"/"ee",
#: those are already unambiguous) landing on the very last syllable of a
#: word conventionally represents the LONG vowel in real Hindi spelling,
#: not the short one: "kitna" is कितना (long आ), not कितन; "hoti" is होती
#: (long ई), not होति. The same short letter mid-word means what it says —
#: "kar" is कर, not कार — so this only overrides the *last* syllable's
#: matra/independent-vowel choice, found by checking the match reaches the
#: end of the string. Found by hand-tracing "kya"/"kitna"/"hoga"/"hoti"
#: against their real spellings after these words kept failing to match
#: the S1 lexicon post-transliteration; see query/pipeline.py's docstring
#: for the retrieval-failure chain this was breaking.
_LONG_AT_WORD_END_MATRA = {"a": "ा", "i": "ी", "u": "ू"}
_LONG_AT_WORD_END_INITIAL = {"a": "आ", "i": "ई", "u": "ऊ"}

#: Common words the syllable-table approach cannot reach even with the
#: word-final-length fix above, because they need a conjunct (two
#: consonants joined with no vowel between them via a virama) rather than
#: a vowel-length correction. "kya" is by far the highest-frequency case in
#: this dataset's queries — checked first, before the general algorithm.
_KNOWN_WORDS: dict[str, str] = {
    "kya": "क्या",       # conjunct क्य, not two open syllables
    "chahiye": "चाहिए",  # mid-word long आ the word-final rule can't reach
    "pehle": "पहले",     # first syllable is bare प, not पे
    "ghante": "घंटे",    # nasal ं before ट, not a full न
    "upar": "ऊपर",       # long ऊ, not short उ
    "walo": "वालों",     # plural oblique वालों, not a bare guess at वलो
    "shikayat": "शिकायत",  # mid-word long आ before य
    "dusri": "दूसरी",    # long ऊ, not short उ
    # Word-final/pre-consonant nasalization (anusvara ं) has no table entry
    # at all — the algorithm has no way to know "n" here means a nasal
    # mark, not a full न, so these three extremely common words come out
    # as two syllables (मेइन, हैन) or missing the mark entirely (नही
    # instead of नहीं).
    "me": "में", "mein": "में", "nahi": "नहीं", "hain": "हैं",
    "turant": "तुरंत",
}


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
        if w in _KNOWN_WORDS:
            return _KNOWN_WORDS[w]

        out: list[str] = []
        i = 0
        n = len(w)
        while i < n:
            cons = next((c for c, _ in _CONSONANTS if w.startswith(c, i)), None)
            if cons:
                dev_c = dict(_CONSONANTS)[cons]
                i += len(cons)
                vowel = next((v for v, _ in _MATRAS if w.startswith(v, i)), None)
                if vowel:
                    matra = dict(_MATRAS)[vowel]
                    if i + len(vowel) == n and vowel in _LONG_AT_WORD_END_MATRA:
                        matra = _LONG_AT_WORD_END_MATRA[vowel]
                    out.append(dev_c + matra)
                    i += len(vowel)
                else:
                    out.append(dev_c)  # inherent vowel, no matra written
                continue
            vow = next((v for v, _ in _VOWELS_INITIAL if w.startswith(v, i)), None)
            if vow:
                glyph = dict(_VOWELS_INITIAL)[vow]
                if i + len(vow) == n and vow in _LONG_AT_WORD_END_INITIAL:
                    glyph = _LONG_AT_WORD_END_INITIAL[vow]
                out.append(glyph)
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
