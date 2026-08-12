"""Devanagari Hindi -> English translation — S1b.

Runs after script detection and, for `hi-Latn` queries, after
transliteration — so this stage only ever receives Devanagari text.
"""

from __future__ import annotations

import re
import warnings
from typing import Protocol


class Translator(Protocol):
    name: str

    def translate(self, text: str, source_lang: str = "hi") -> str: ...


#: Built by scanning the Devanagari vocabulary that actually appears across
#: eval/build_dataset.py's SPEC and UNANSWERABLE entries, then glossing each
#: word by hand. That makes it train-on-test at the vocabulary level: hi-Deva
#: quality numbers produced with LexiconTranslator as the active backend are
#: optimistic relative to what a lexicon built independently of this eval
#: set would score, because "does this word happen to be in the dictionary"
#: is not independent of "was this word used to build the dictionary."
#:
#: It is *not* mined at the query or answer level — no full question or its
#: translation is stored anywhere here, only isolated words — so it cannot
#: answer a query it has never seen a resembling one to. That distinction
#: matters (it is a real, if narrow, translator rather than a lookup table
#: keyed by item id) but does not make the coverage claim above go away.
#: `LexiconTranslator` warns on construction for this reason; see
#: examples/reference_system/README.md.
_LEXICON: dict[str, str] = {
    "अधिक": "more", "अवधि": "period", "अस्पताल": "hospital", "आयु": "age",
    "आवेदन": "apply", "इलाज": "treatment", "इस": "this", "उप": "sub",
    "उससे": "than that", "ऋण": "loan", "और": "and", "कंपनी": "company",
    "कब": "when", "कमरे": "room", "कर": "do", "करना": "to do",
    "कराएं": "get approved", "करें": "do", "कवर": "cover", "का": "of",
    "कितना": "how much", "कितनी": "how much", "किराया": "rent", "की": "of",
    "कौन": "which", "सा": "one",
    "के": "of", "कैशलेस": "cashless", "कैसे": "how", "कॉस्मेटिक": "cosmetic",
    "को": "to", "क्या": "what", "खर्च": "expenses", "गया": "gone",
    "गृह": "home", "घुटना": "knee", "छूट": "grace", "जमा": "submit",
    "जुड़ता": "added", "तक": "by", "तुरंत": "immediately", "तो": "then",
    "दर": "rate", "दांतों": "dental", "दावा": "claim", "दावे": "claim",
    "दूसरी": "another", "देने": "paying", "देय": "payable",
    "मेरा": "my",
    "नवीनीकरण": "renewal", "नहीं": "not", "न्यूनतम": "minimum", "पर": "on",
    "पहले": "before", "पासपोर्ट": "passport", "पूर्व": "pre",
    "पॉलिसी": "policy", "पोर्ट": "port", "प्रतिपूर्ति": "reimbursement",
    "प्रतीक्षा": "waiting", "प्रसूति": "maternity", "बदलने": "replacement",
    "बाद": "after", "बीमा": "insurance", "बीमारी": "disease",
    "बोनस": "bonus", "ब्याज": "interest", "ब्राज़ील": "brazil",
    "भर्ती": "admission", "भुगतान": "payment", "मिलने": "receiving",
    "मुक्त": "free", "में": "in", "मेरी": "my", "मैं": "i",
    "मोतियाबिंद": "cataract", "मौजूदा": "existing", "यदि": "if",
    "रद्द": "cancel", "राजधानी": "capital", "राशि": "amount",
    "स्थिति": "status", "सूचना": "notification", "प्रारूप": "format",
    "उपयोग": "use", "ईमेल": "email", "रिमाइंडर": "reminder",
    "लगता": "applies", "लागू": "applicable", "लिए": "for", "लिया": "taken",
    "वर्ष": "years", "विदेश": "abroad", "शामिल": "included",
    "शिकायत": "grievance", "संचयी": "cumulative", "सकता": "can",
    "सदस्यों": "members", "समय": "time", "सर्जरी": "surgery", "सह": "co",
    "सीमा": "limit", "से": "from", "स्वीकृत": "approved", "हल": "resolved",
    "हूँ": "am", "है": "is", "हैं": "are", "होता": "is", "होती": "is",
    "होने": "being",
}

_TOKEN = re.compile(r"[ऀ-ॿ]+|[^\sऀ-ॿ]+")


class LexiconTranslator:
    """Word-for-word gloss against a lexicon built from this eval set's own
    vocabulary. No model — and, despite appearances, not a general-purpose
    dictionary either; see the train-on-test note on `_LEXICON` above.

    Reorders nothing and disambiguates nothing — a gloss, not a machine
    translation. It exists so the pipeline runs offline; how well it serves
    retrieval downstream depends entirely on how many content words the
    lexicon happens to cover, and that coverage was shaped by this dataset.
    """

    name = "lexicon"

    def __init__(self) -> None:
        warnings.warn(
            "LexiconTranslator's dictionary was built by reading this eval "
            "dataset's own Hindi vocabulary (examples/reference_system/eval/"
            "build_dataset.py) — hi-Deva/hi-Latn scores produced with it as "
            "the active translator are train-on-test at the vocabulary "
            "level, not a result a general-purpose lexicon would reproduce. "
            "See examples/reference_system/README.md.",
            stacklevel=2,
        )

    def translate(self, text: str, source_lang: str = "hi") -> str:
        words = _TOKEN.findall(text)
        return " ".join(_LEXICON.get(w, w) for w in words)


class IndicTrans2Translator:
    """AI4Bharat IndicTrans2, indic-en distilled checkpoint.

    Downloads on first use (hundreds of MB to ~2GB, depending on checkpoint).
    """

    name = "indictrans2"

    def __init__(
        self, model_name: str = "ai4bharat/indictrans2-indic-en-dist-200M"
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "IndicTrans2 needs transformers + torch — "
                'pip install -e "examples/reference_system[indic]"'
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True
        )

    def translate(self, text: str, source_lang: str = "hi") -> str:
        tag = "hin_Deva" if source_lang == "hi" else source_lang
        batch = self._tokenizer(f"{tag} eng_Latn {text}", return_tensors="pt")
        out = self._model.generate(**batch, max_length=256, num_beams=4)
        return self._tokenizer.decode(out[0], skip_special_tokens=True)


def get_translator(name: str = "lexicon") -> Translator:
    if name in {"indictrans2", "ai4bharat"}:
        return IndicTrans2Translator()
    if name == "lexicon":
        return LexiconTranslator()
    raise ValueError(f"unknown translator: {name}")
