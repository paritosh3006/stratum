"""Script/language detection for the query pipeline — S0.

Three tags, matching the dataset's `language` field exactly:

    en       English, Latin script
    hi-Deva  Hindi, Devanagari script
    hi-Latn  Hindi, romanized ("Hinglish")

Devanagari is unambiguous: Unicode reserves U+0900-U+097F for it, so a block
check settles `hi-Deva` outright in both detectors below. The only genuinely
hard call is Latin-script text — "kitna hai" and "how much is it" use the
same alphabet, and telling them apart is a language-id problem, not a script
one.
"""

from __future__ import annotations

import re
from typing import Protocol

EN, HI_DEVA, HI_LATN = "en", "hi-Deva", "hi-Latn"

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_WORD = re.compile(r"[a-zA-Z]+")

#: Closed-class Hindi function words that survive romanization almost
#: verbatim and rarely appear in English insurance queries. Deliberately
#: restricted to function words rather than domain vocabulary — "premium",
#: "policy", "cover" are English loanwords in real Hinglish queries and
#: would not discriminate hi-Latn from en.
#:
#: Reused downstream by the transliterator (query/transliterate.py) to
#: decide which tokens in a code-mixed query are safe to convert — the same
#: property that makes a word a good hi-Latn/en discriminator here makes it
#: a word actually worth transliterating there.
HI_LATN_MARKERS = frozenset("""
kya hai hain kaise kitna kitne kitni hoga hogi hoti hota hote milta milti
milega karte kare karo karna sakte sakta sakti chahiye wala wali wale walo nahi
mera meri mere hum humein tumhe aapka aapki aapke liye pehle baad turant kab
kaun kyun kyu se ke ki ka me mein aur ya bhi to hi thi tha the yeh woh
iska uska waqt paise rupaye saal varsh kro dijiye jata jati jate
ghante hona upar din jama shikayat hui videsh dusri ko lete
""".split())


class ScriptDetector(Protocol):
    name: str

    def detect(self, text: str) -> str: ...


class HeuristicScriptDetector:
    """Unicode block for Devanagari, a function-word ratio for the rest.

    No model, no download. Devanagari detection is exact by construction.
    Roman-Hindi detection is a coarse bag-of-words vote and will misfire on
    short or heavily code-mixed queries — that is a limitation of this
    specific detector, not of the pipeline stage, and the fix is the real
    detector below rather than a bigger word list.
    """

    name = "heuristic"

    def __init__(self, min_marker_ratio: float = 0.25) -> None:
        self.min_marker_ratio = min_marker_ratio

    def detect(self, text: str) -> str:
        if _DEVANAGARI.search(text):
            return HI_DEVA
        words = [w.lower() for w in _WORD.findall(text)]
        if not words:
            return EN
        hits = sum(1 for w in words if w in HI_LATN_MARKERS)
        return HI_LATN if hits / len(words) >= self.min_marker_ratio else EN


class FastTextScriptDetector:
    """fastText lid218e language id, for the Latin-script call the heuristic
    can't make reliably.

    Downloads the language-id model (~130MB) on first use. Devanagari still
    resolves by Unicode block — a language-id model would only add noise to
    a call the script itself already answers.
    """

    name = "fasttext-lid218e"

    def __init__(self, model_path: str | None = None) -> None:
        try:
            import fasttext  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "fastText language id needs fasttext — "
                'pip install -e "examples/reference_system[indic]"'
            ) from exc
        import fasttext
        from huggingface_hub import hf_hub_download

        path = model_path or hf_hub_download(
            "facebook/fasttext-language-identification", "model.bin"
        )
        self._model = fasttext.load_model(path)

    def detect(self, text: str) -> str:
        if _DEVANAGARI.search(text):
            return HI_DEVA
        label, _ = self._model.predict(text.replace("\n", " "), k=1)
        lang = label[0].replace("__label__", "").split("_")[0]
        return HI_LATN if lang == "hi" else EN


def get_script_detector(name: str = "heuristic") -> ScriptDetector:
    if name in {"fasttext", "fasttext-lid218e"}:
        return FastTextScriptDetector()
    if name == "heuristic":
        return HeuristicScriptDetector()
    raise ValueError(f"unknown script detector: {name}")
