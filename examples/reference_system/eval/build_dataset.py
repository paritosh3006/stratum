"""Build an eval dataset whose gold chunk ids come from the real corpus.

Gold references cannot be written by hand: the ids are content hashes, so they
only exist once the corpus has been ingested. This script states each question
with a *locator* — distinctive phrases that identify the passage answering it —
resolves those to chunk ids, and fails loudly if a locator matches zero or
several chunks.

Failing loudly matters. A locator that silently matches nothing produces an item
with no gold, which scores as "not applicable" rather than as a miss, and
quietly removes the hardest questions from the evaluation.

The Hindi and Roman-Hindi strings here are written by hand, not machine
translated. Machine-translating the English set would make the oracle passes
reverse exactly the transformation that produced the items, and the system
would look far better than it is.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from reference_system.ingest.chunk import build_corpus  # noqa: E402
from reference_system.pipeline.extractive import select_span  # noqa: E402
from reference_system.retrieval.hybrid import Hit  # noqa: E402

CORPUS = ROOT / "corpus"

# (parallel_id, slice, locator, {lang: query}, extras)
SPEC: list[tuple] = [
    ("q01", "parallel_core", "Specified surgical procedures carry a waiting period", {
        "en": "Is knee replacement surgery covered under my policy?",
        "hi-Deva": "क्या मेरी पॉलिसी में घुटना बदलने की सर्जरी शामिल है?",
        "hi-Latn": "mera knee replacement surgery cover hoga kya",
    }, {}),
    ("q02", "parallel_core", "Maternity expenses are payable after", {
        "en": "What is the waiting period for maternity expenses?",
        "hi-Deva": "प्रसूति खर्च के लिए प्रतीक्षा अवधि कितनी है?",
        "hi-Latn": "maternity ke liye waiting period kitna hai",
    }, {}),
    ("q03", "term_heavy", "aged 45 with a sum insured of 500000", {
        "en": "What is the premium for a member aged 45 with sum insured of 500000?",
        "hi-Deva": "45 वर्ष की आयु और 500000 बीमा राशि पर premium कितना है?",
        "hi-Latn": "45 saal aur 500000 sum insured pe premium kitna hai",
    }, {"numerals": ["45", "500000"]}),
    ("q04", "term_heavy", "grace period of 30 days is allowed", {
        "en": "What is the grace period for paying the renewal premium?",
        "hi-Deva": "नवीनीकरण premium देने के लिए छूट अवधि कितनी है?",
        "hi-Latn": "renewal premium ke liye grace period kitna hai",
    }, {}),
    ("q05", "parallel_core", "Dental treatment is excluded unless", {
        "en": "Is dental treatment covered under this policy?",
        "hi-Deva": "क्या इस पॉलिसी में दांतों का इलाज शामिल है?",
        "hi-Latn": "dental treatment cover hota hai kya",
    }, {}),
    ("q06", "parallel_core", "Cosmetic surgery is excluded unless", {
        "en": "Is cosmetic surgery covered?",
        "hi-Deva": "क्या कॉस्मेटिक सर्जरी शामिल है?",
        "hi-Latn": "cosmetic surgery cover hoti hai kya",
    }, {}),
    ("q07", "parallel_core", "minimum of 24 consecutive hours", {
        "en": "What is the minimum hospitalisation duration for a claim?",
        "hi-Deva": "दावे के लिए न्यूनतम अस्पताल में भर्ती अवधि क्या है?",
        "hi-Latn": "claim ke liye minimum hospitalisation kitne ghante ka hona chahiye",
    }, {"numerals": ["24"]}),
    ("q08", "term_heavy", "Room rent is payable up to 1 percent", {
        "en": "What is the room rent limit under the policy?",
        "hi-Deva": "पॉलिसी में कमरे का किराया कितना देय है?",
        "hi-Latn": "room rent ki limit kya hai policy me",
    }, {}),
    ("q09", "parallel_core", "co-payment of 20 percent applies", {
        "en": "Does a co-payment apply for members aged 61 and above?",
        "hi-Deva": "61 वर्ष और उससे अधिक आयु के सदस्यों पर सह-भुगतान लागू होता है?",
        "hi-Latn": "61 saal se upar walo pe copayment lagta hai kya",
    }, {"numerals": ["61", "20"]}),
    ("q10", "term_heavy", "cumulative bonus of 10 percent", {
        "en": "How much cumulative bonus is added for a claim-free year?",
        "hi-Deva": "दावा-मुक्त वर्ष के लिए कितना संचयी बोनस जुड़ता है?",
        "hi-Latn": "claim free year pe kitna cumulative bonus milta hai",
    }, {"numerals": ["10"]}),
    ("q11", "parallel_core", "pre-authorisation from the third party", {
        "en": "How do I get cashless treatment approved before admission?",
        "hi-Deva": "भर्ती से पहले कैशलेस इलाज कैसे स्वीकृत कराएं?",
        "hi-Latn": "admission se pehle cashless approval kaise lete hai",
    }, {}),
    ("q12", "parallel_core", "submitted within 30 days of discharge", {
        "en": "By when must a reimbursement claim be submitted?",
        "hi-Deva": "प्रतिपूर्ति दावा कब तक जमा करना होता है?",
        "hi-Latn": "reimbursement claim kitne din me jama karna hota hai",
    }, {"numerals": ["30"]}),
    ("q13", "term_heavy", "Cataract surgery is subject to a sub-limit", {
        "en": "What is the sub-limit for cataract surgery?",
        "hi-Deva": "मोतियाबिंद सर्जरी की उप-सीमा क्या है?",
        "hi-Latn": "cataract surgery ki sub limit kitni hai",
    }, {"numerals": ["40000"]}),
    ("q14", "parallel_core", "cancel the policy within 15 days of receipt", {
        "en": "Can I cancel the policy soon after receiving it?",
        "hi-Deva": "क्या मैं पॉलिसी मिलने के तुरंत बाद रद्द कर सकता हूँ?",
        "hi-Latn": "policy milne ke baad turant cancel kar sakte hai kya",
    }, {"numerals": ["15"]}),
    ("q15", "parallel_core", "approach the Insurance Ombudsman", {
        "en": "What can I do if my grievance is not resolved?",
        "hi-Deva": "यदि मेरी शिकायत हल नहीं होती तो मैं क्या कर सकता हूँ?",
        "hi-Latn": "shikayat solve nahi hui to kya kar sakte hai",
    }, {}),
    ("q16", "multi_hop", "Pre-existing diseases are covered after", {
        "en": "How long must I wait for a pre-existing disease to be covered?",
        "hi-Deva": "पूर्व-मौजूदा बीमारी के कवर होने में कितना समय लगता है?",
        "hi-Latn": "pre existing disease kitne time baad cover hoti hai",
    }, {"numerals": ["36"]}),
    ("q17", "parallel_core", "Treatment taken outside India", {
        "en": "Is treatment taken abroad covered?",
        "hi-Deva": "क्या विदेश में लिया गया इलाज कवर होता है?",
        "hi-Latn": "videsh me treatment cover hota hai kya",
    }, {}),
    ("q18", "term_heavy", "port this policy to another insurer", {
        "en": "How do I port this policy to another insurer?",
        "hi-Deva": "इस पॉलिसी को दूसरी बीमा कंपनी में कैसे पोर्ट करें?",
        "hi-Latn": "policy ko dusri company me port kaise kare",
    }, {"numerals": ["45"]}),
    ("q19", "term_heavy", "Claim status updates are sent by SMS using the template", {
        "en": "What format is used for claim status SMS notifications?",
        "hi-Deva": "दावा स्थिति SMS सूचना के लिए कौन सा प्रारूप उपयोग होता है?",
        "hi-Latn": "claim status SMS notification ke liye kaun sa format use hota hai",
    }, {"placeholders": ["{claim_id}", "{status}", "{amount}"]}),
    ("q20", "term_heavy", "Policy renewal reminders are sent by email using the template", {
        "en": "What format is used for policy renewal email reminders?",
        "hi-Deva": "पॉलिसी नवीनीकरण ईमेल रिमाइंडर के लिए कौन सा प्रारूप उपयोग होता है?",
        "hi-Latn": "policy renewal email reminder ke liye kaun sa format use hota hai",
    }, {"placeholders": ["{policy_number}", "{renewal_date}"]}),
]

UNANSWERABLE: list[tuple] = [
    ("u01", {
        "en": "What is the capital of Brazil?",
        "hi-Deva": "ब्राज़ील की राजधानी क्या है?",
        "hi-Latn": "brazil ki rajdhani kya hai",
    }),
    ("u02", {
        "en": "How do I apply for a passport?",
        "hi-Deva": "पासपोर्ट के लिए आवेदन कैसे करें?",
        "hi-Latn": "passport ke liye apply kaise kare",
    }),
    ("u03", {
        "en": "What is the interest rate on a home loan?",
        "hi-Deva": "गृह ऋण पर ब्याज दर क्या है?",
        "hi-Latn": "home loan pe interest rate kya hai",
    }),
]

LANGS = ["en", "hi-Deva", "hi-Latn"]


def _flat(text: str) -> str:
    """Collapse whitespace before matching.

    Source documents are hard-wrapped, so a locator spanning a line break
    would otherwise fail to match text that is plainly present.
    """
    return re.sub(r"\s+", " ", text.lower()).strip()


def resolve(locator: str, chunks) -> str:
    needle = _flat(locator)
    matches = [c.chunk_id for c in chunks if needle in _flat(c.text)]
    if not matches:
        raise SystemExit(f"locator matched nothing: {locator!r}")
    if len(matches) > 1:
        raise SystemExit(f"locator is ambiguous ({len(matches)} chunks): {locator!r}")
    return matches[0]


def gold_answer_for(query_en: str, chunk) -> str:
    """Extract the gold answer span from the resolved gold chunk.

    Not the question restated: `oracle_answer` feeds this straight to
    rendering as `answer_override`, so if it were the English query text
    (as it used to be), that pass would score the system on how well it
    echoes a question back rather than on rendering an actual answer — and
    every language would land on the same rendering-loss number regardless
    of what rendering actually does, which is exactly what happened before
    this fix (both hi-Deva and hi-Latn read an identical -17.5).

    Reuses the same overlap-scoring the system's own answerer uses
    (`min_score=0.0` because the chunk is already known-good — the relevance
    floor exists to reject wrong chunks, not to second-guess a gold one).
    """
    span = select_span(query_en, [Hit(chunk.chunk_id, 1.0, chunk.text)], min_score=0.0)
    return span.text if span else query_en


def main() -> None:
    chunks = build_corpus(CORPUS, target_tokens=180)
    by_id = {c.chunk_id: c for c in chunks}
    print(f"corpus: {len(chunks)} chunks")

    items: list[dict] = []
    n = 0

    for pid, slice_, locator, queries, extras in SPEC:
        gold = resolve(locator, chunks)
        gold_answer = gold_answer_for(queries["en"], by_id[gold])
        for lang in LANGS:
            n += 1
            items.append({
                "id": f"ins-{n:04d}", "language": lang, "slice": slice_,
                "parallel_id": pid, "query": queries[lang],
                "gold_answer": gold_answer, "gold_chunk_ids": [gold],
                "answerable": True,
                "numerals": extras.get("numerals", []),
                "entities": extras.get("entities", []),
                "placeholders": extras.get("placeholders", []),
            })

    for pid, queries in UNANSWERABLE:
        for lang in LANGS:
            n += 1
            items.append({
                "id": f"ins-{n:04d}", "language": lang, "slice": "unanswerable",
                "parallel_id": pid, "query": queries[lang],
                "gold_answer": "", "gold_chunk_ids": [], "answerable": False,
                "numerals": [], "entities": [], "placeholders": [],
            })

    out = ROOT / "eval" / "insurance.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"{len(items)} items · {len(SPEC) + len(UNANSWERABLE)} parallel groups → {out}")


if __name__ == "__main__":
    main()
