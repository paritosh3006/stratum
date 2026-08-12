"""Generate the synthetic demo dataset.

This is NOT the evaluation dataset. It is scaffolding: enough parallel items
that the cascade has the statistical power to demonstrate itself, built from
templates so it can be regenerated deterministically.

The real dataset is human-authored and human-verified, per docs/dataset-design.md.
Synthetic items are marked `synthetic: true` so no report built on them can be
mistaken for evidence about a real system.
"""

from __future__ import annotations

import json
from pathlib import Path

# (english, hindi-devanagari, hindi-roman, tamil)
COVERAGE = [
    ("Is {proc} covered under my policy?",
     "क्या मेरी पॉलिसी में {proc_hi} शामिल है?",
     "mera {proc_rom} cover hoga kya",
     "{proc_ta} என் பாலிசியில் அடங்குமா?"),
]

PROCEDURES = [
    ("knee replacement surgery", "घुटना बदलने की सर्जरी", "knee operation", "முழங்கால் மாற்று அறுவை சிகிச்சை", "c_042"),
    ("cataract surgery", "मोतियाबिंद की सर्जरी", "cataract operation", "கண்புரை அறுவை சிகிச்சை", "c_045"),
    ("dental treatment", "दांतों का इलाज", "dental treatment", "பல் சிகிச்சை", "c_055"),
    ("maternity care", "प्रसूति देखभाल", "maternity care", "மகப்பேறு பராமரிப்பு", "c_061"),
    ("physiotherapy", "फिजियोथेरेपी", "physiotherapy", "பிசியோதெரபி", "c_067"),
    ("cardiac bypass", "हृदय बाईपास", "heart bypass", "இதய பைபாஸ்", "c_071"),
    ("chemotherapy", "कीमोथेरेपी", "chemotherapy", "கீமோதெரபி", "c_074"),
    ("cosmetic surgery", "कॉस्मेटिक सर्जरी", "cosmetic surgery", "அழகு அறுவை சிகிச்சை", "c_079"),
]

PREMIUM_CASES = [
    (28, 300000, "c_011"), (35, 500000, "c_011"), (45, 500000, "c_012"),
    (52, 1000000, "c_013"), (60, 750000, "c_014"), (41, 250000, "c_011"),
]

TERM_CASES = [
    ("What is the premium payable this year?",
     "इस वर्ष देय premium कितना है?",
     "premium kitna dena padega",
     "இந்த ஆண்டு செலுத்த வேண்டிய premium என்ன?", "c_011"),
    ("How is the premium calculated for my policy?",
     "मेरी पॉलिसी के लिए premium की गणना कैसे होती है?",
     "premium kaise calculate hota hai",
     "என் பாலிசிக்கான premium எப்படி கணக்கிடப்படுகிறது?", "c_012"),
    ("Can I pay the premium in instalments?",
     "क्या मैं premium किश्तों में दे सकता हूँ?",
     "premium instalment me de sakte hai kya",
     "premium தவணைகளில் செலுத்த முடியுமா?", "c_015"),
]

PLACEHOLDER_CASES = [
    ("Your EMI of {amount} is due on {date} - explain the premium grace period",
     "{amount} की EMI {date} को देय है - premium की छूट अवधि बताएं",
     "{amount} ki EMI {date} ko due hai - premium grace period batao",
     "{amount} EMI {date} அன்று செலுத்த வேண்டும் - premium சலுகைக் காலத்தை விளக்கவும்", "c_077"),
    ("Reminder: policy {policy_no} expires on {date}. What are my renewal options?",
     "सूचना: पॉलिसी {policy_no} {date} को समाप्त हो रही है। नवीनीकरण के विकल्प क्या हैं?",
     "policy {policy_no} {date} ko expire ho rahi hai, renewal options kya hai",
     "பாலிசி {policy_no} {date} அன்று காலாவதியாகிறது. புதுப்பித்தல் விருப்பங்கள் என்ன?", "c_081"),
]

ENTITY_CASES = [
    ("POL-88213", "c_055"), ("POL-45019", "c_056"), ("POL-77302", "c_057"),
]

UNANSWERABLE = [
    ("What is the capital of Brazil?", "ब्राज़ील की राजधानी क्या है?",
     "brazil ki capital kya hai", "பிரேசிலின் தலைநகரம் என்ன?"),
    ("Who won the cricket world cup?", "क्रिकेट विश्व कप कौन जीता?",
     "cricket world cup kaun jeeta", "கிரிக்கெட் உலகக் கோப்பையை யார் வென்றார்?"),
    ("How do I apply for a passport?", "पासपोर्ट के लिए आवेदन कैसे करें?",
     "passport ke liye apply kaise kare", "கடவுச்சீட்டுக்கு எப்படி விண்ணப்பிப்பது?"),
]

LANGS = ["en", "hi-Deva", "hi-Latn", "ta-Taml"]


def build() -> list[dict]:
    items: list[dict] = []
    n = 0

    def emit(pid: str, slice_: str, queries: dict[str, str], chunks: list[str],
             *, answerable=True, placeholders=None, numerals=None, entities=None):
        nonlocal n
        for lang in LANGS:
            n += 1
            items.append({
                "id": f"syn-{n:04d}",
                "language": lang,
                "slice": slice_,
                "parallel_id": pid,
                "query": queries[lang],
                "gold_answer": queries["en"],
                "gold_chunk_ids": chunks,
                "answerable": answerable,
                "placeholders": placeholders or [],
                "numerals": numerals or [],
                "entities": entities or [],
                "synthetic": True,
            })

    # coverage questions
    tmpl = COVERAGE[0]
    for i, (en, hi, rom, ta, chunk) in enumerate(PROCEDURES):
        emit(f"cov{i:02d}", "parallel_core", {
            "en": tmpl[0].format(proc=en),
            "hi-Deva": tmpl[1].format(proc_hi=hi),
            "hi-Latn": tmpl[2].format(proc_rom=rom),
            "ta-Taml": tmpl[3].format(proc_ta=ta),
        }, [chunk])

    # premium + numerals
    for i, (age, sum_insured, chunk) in enumerate(PREMIUM_CASES):
        emit(f"prm{i:02d}", "term_heavy", {
            "en": f"What is the premium for a {age} year old with sum insured of {sum_insured}?",
            "hi-Deva": f"{age} वर्ष की आयु और {sum_insured} बीमा राशि पर premium कितना है?",
            "hi-Latn": f"{age} saal ki age aur {sum_insured} sum insured pe premium kitna hai",
            "ta-Taml": f"{age} வயது மற்றும் {sum_insured} காப்பீட்டுத் தொகைக்கான premium என்ன?",
        }, [chunk], numerals=[str(age), str(sum_insured)])

    # terminology
    for i, (en, hi, rom, ta, chunk) in enumerate(TERM_CASES):
        emit(f"trm{i:02d}", "term_heavy",
             {"en": en, "hi-Deva": hi, "hi-Latn": rom, "ta-Taml": ta}, [chunk])

    # placeholders
    for i, (en, hi, rom, ta, chunk) in enumerate(PLACEHOLDER_CASES):
        ph = ["{amount}", "{date}"] if "{amount}" in en else ["{policy_no}", "{date}"]
        emit(f"plc{i:02d}", "term_heavy",
             {"en": en, "hi-Deva": hi, "hi-Latn": rom, "ta-Taml": ta}, [chunk],
             placeholders=ph)

    # entities
    for i, (pol, chunk) in enumerate(ENTITY_CASES):
        emit(f"ent{i:02d}", "parallel_core", {
            "en": f"Does policy {pol} cover dental treatment?",
            "hi-Deva": f"क्या पॉलिसी {pol} में दांतों का इलाज शामिल है?",
            "hi-Latn": f"policy {pol} me dental treatment cover hai kya",
            "ta-Taml": f"{pol} பாலிசி பல் சிகிச்சையை உள்ளடக்குகிறதா?",
        }, [chunk], entities=[pol])

    # unanswerable
    for i, (en, hi, rom, ta) in enumerate(UNANSWERABLE):
        emit(f"una{i:02d}", "unanswerable",
             {"en": en, "hi-Deva": hi, "hi-Latn": rom, "ta-Taml": ta}, [],
             answerable=False)

    return items


if __name__ == "__main__":
    items = build()
    out = Path(__file__).parent / "demo.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"{len(items)} items · {len(items) // len(LANGS)} parallel groups")
    print(dict(Counter(i["slice"] for i in items if i["language"] == "en")))
