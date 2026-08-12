"""English -> Hindi (Devanagari) translation for rendering — S4.

Runs on the S3 answer span (an English sentence from the corpus), after
placeholders and numerals have been masked (mask.py) and before glossary
enforcement (glossary.py) forces approved terms over whatever this produced
for them.
"""

from __future__ import annotations

import re
from typing import Protocol


class RenderTranslator(Protocol):
    name: str

    def translate(self, text: str, target_lang: str = "hi") -> str: ...


#: Built by hand from general insurance-policy vocabulary while reading this
#: corpus's documents — not by scanning eval/build_dataset.py the way S1's
#: LexiconTranslator's dictionary was (see query/translate.py). That
#: distinction matters less than it sounds: a dictionary sized to translate
#: *this corpus's own prose* is still shaped by this corpus, just at the
#: source-document level rather than the eval-query level. Treat render
#: quality numbers produced with this backend with the same caution as S1's
#: — a lexicon built for a different insurance corpus would have different,
#: probably worse, coverage here.
_EN_HI: dict[str, str] = {
    "a": "", "an": "", "the": "",
    "is": "है", "are": "हैं", "was": "था", "were": "थे", "be": "हो", "been": "रहा",
    "and": "और", "or": "या", "not": "नहीं", "no": "नहीं", "any": "कोई",
    "this": "यह", "that": "वह", "these": "ये", "those": "वे",
    "policy": "पॉलिसी", "policies": "पॉलिसियाँ", "premium": "प्रीमियम",
    "claim": "दावा", "claims": "दावे", "cover": "कवर", "covers": "कवर",
    "covered": "कवर",
    "coverage": "कवरेज", "insured": "बीमित", "insurance": "बीमा",
    "insurer": "बीमाकर्ता", "company": "कंपनी", "policyholder": "पॉलिसीधारक",
    "member": "सदस्य", "members": "सदस्य", "sum": "राशि", "amount": "राशि",
    "waiting": "प्रतीक्षा", "period": "अवधि", "periods": "अवधियाँ",
    "grace": "छूट", "renewal": "नवीनीकरण", "renewable": "नवीकरणीय",
    "cancel": "रद्द", "cancellation": "रद्दीकरण", "lapse": "समाप्त",
    "lapses": "समाप्त होती है",
    "hospital": "अस्पताल", "hospitals": "अस्पताल",
    "hospitalisation": "अस्पताल में भर्ती", "hospitalization": "अस्पताल में भर्ती",
    "admission": "भर्ती", "admitted": "भर्ती", "discharge": "छुट्टी",
    "network": "नेटवर्क", "cashless": "कैशलेस", "reimbursement": "प्रतिपूर्ति",
    "treatment": "इलाज", "surgery": "सर्जरी", "surgical": "शल्य",
    "procedure": "प्रक्रिया", "procedures": "प्रक्रियाएं",
    "condition": "स्थिति", "ailment": "बीमारी", "injury": "चोट",
    "disease": "बीमारी", "diseases": "बीमारियाँ", "illness": "बीमारी",
    "existing": "मौजूदा", "pre-existing": "पूर्व-मौजूदा",
    "diagnosed": "निदान किया गया", "physician": "चिकित्सक", "doctor": "डॉक्टर",
    "maternity": "प्रसूति", "delivery": "प्रसव", "deliveries": "प्रसव",
    "cataract": "मोतियाबिंद", "dental": "दांतों", "cosmetic": "कॉस्मेटिक",
    "knee": "घुटना", "hip": "कूल्हा", "replacement": "बदलने",
    "hernia": "हर्निया", "repair": "मरम्मत", "benign": "सौम्य",
    "room": "कमरा", "rent": "किराया",
    "co-payment": "सह-भुगतान", "copayment": "सह-भुगतान",
    "cumulative": "संचयी", "bonus": "बोनस", "claim-free": "दावा-मुक्त",
    "excluded": "बहिष्कृत", "exclusion": "बहिष्करण", "exclusions": "बहिष्करण",
    "expenses": "खर्च", "payable": "देय", "paid": "भुगतान किया गया",
    "receipt": "प्राप्ति", "received": "प्राप्त हुआ",
    "sub-limit": "उप-सीमा", "sub-limits": "उप-सीमाएं", "limit": "सीमा",
    "limits": "सीमाएं", "minimum": "न्यूनतम", "maximum": "अधिकतम",
    "percent": "प्रतिशत", "per": "प्रति", "day": "दिन", "days": "दिन",
    "month": "महीना", "months": "महीने", "year": "वर्ष", "years": "वर्ष",
    "hour": "घंटा", "hours": "घंटे", "age": "आयु", "aged": "आयु के",
    "commencement": "प्रारंभ", "continuous": "निरंतर", "continuity": "निरंतरता",
    "first": "पहली", "prior": "पूर्व", "within": "के भीतर",
    "before": "पहले", "after": "बाद", "from": "से", "of": "का", "to": "को",
    "for": "के लिए", "with": "साथ", "up": "तक", "including": "सहित",
    "means": "मतलब है", "applies": "लागू होता है", "apply": "लागू करें",
    "applicable": "लागू", "subject": "अधीन", "specified": "निर्दिष्ट",
    "including,": "सहित,", "issued": "जारी",
    "grievance": "शिकायत", "grievances": "शिकायतें",
    "resolved": "हल", "resolve": "हल करें", "unresolved": "अनसुलझा",
    "ombudsman": "लोकपाल", "jurisdiction": "अधिकार क्षेत्र",
    "abroad": "विदेश", "port": "पोर्ट", "another": "दूसरी", "insurer's": "बीमाकर्ता का",
    "approved": "स्वीकृत", "third": "तीसरी", "party": "पक्ष",
    "written": "लिखित", "notice": "सूचना", "submitted": "जमा किया गया",
    "form": "फॉर्म", "prospectus": "प्रॉस्पेक्टस",
    "your": "आपका", "has": "है", "have": "है", "allowed": "अनुमति है",
    "payment": "भुगतान", "annual": "वार्षिक", "additional": "अतिरिक्त",
    "tax": "कर", "goods": "वस्तुएं", "services": "सेवाएं", "rate": "दर",
    "date": "तारीख", "value": "मूल्य", "status": "स्थिति",
    "sms": "एसएमएस", "email": "ईमेल", "notification": "सूचना",
    "reference": "संदर्भ", "number": "संख्या", "id": "आईडी",
    "identifier": "पहचानकर्ता", "sent": "भेजा गया", "format": "प्रारूप",
    "e-card": "ई-कार्ड", "card": "कार्ड",
}

_TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*|[^\sA-Za-z]+")


class EnHiLexiconTranslator:
    """Word-for-word gloss, English -> Devanagari. No model.

    Reorders nothing and disambiguates nothing — a gloss, not a machine
    translation. Every unknown word (proper nouns, anything outside this
    dictionary) passes through untouched in Latin script, which is a
    plausible-looking but ungrammatical result, not a refusal to render.
    """

    name = "lexicon-en-hi"

    def translate(self, text: str, target_lang: str = "hi") -> str:
        words = _TOKEN.findall(text)
        out = []
        for w in words:
            gloss = _EN_HI.get(w.lower())
            if gloss is None:
                out.append(w)
            elif gloss:
                out.append(gloss)
            # gloss == "" (articles): dropped silently
        return " ".join(out)


class IndicTrans2EnIndicTranslator:
    """AI4Bharat IndicTrans2, en-indic distilled checkpoint.

    Downloads on first use (hundreds of MB to ~2GB, depending on checkpoint).
    """

    name = "indictrans2-en-indic"

    def __init__(
        self, model_name: str = "ai4bharat/indictrans2-en-indic-dist-200M"
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

    def translate(self, text: str, target_lang: str = "hi") -> str:
        tag = "hin_Deva" if target_lang == "hi" else target_lang
        batch = self._tokenizer(f"eng_Latn {tag} {text}", return_tensors="pt")
        out = self._model.generate(**batch, max_length=256, num_beams=4)
        return self._tokenizer.decode(out[0], skip_special_tokens=True)


def get_render_translator(name: str = "lexicon-en-hi") -> RenderTranslator:
    if name in {"indictrans2", "indictrans2-en-indic", "ai4bharat"}:
        return IndicTrans2EnIndicTranslator()
    if name == "lexicon-en-hi":
        return EnHiLexiconTranslator()
    raise ValueError(f"unknown render translator: {name}")
