"""Output-language expectation — reusable across any stage that emits text.

A wrong output language is not always an S0 bug. In a voice pipeline an
ALD (automatic language detection) misclassification at the door produces
a correct-but-wrong-language answer all the way through the pipeline; in
a RAG system a translation-rendering bug at S4 produces the identical
symptom. `check_output_language` measures exactly one thing — does the
detected language of some text match what was expected — and takes no
position on which stage that belongs to or what kind of system produced
the text. The caller decides the stage when it turns a failing check into
a report `Failure`; nothing here hardcodes one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..language.base import LanguageGuess

Outcome = Literal["pass", "fail", "inconclusive", "not_applicable"]


def _split_tag(tag: str) -> tuple[str, str | None]:
    """"hi-Deva" -> ("hi", "Deva"); "en" -> ("en", None)."""
    lang, _, script = tag.partition("-")
    return lang, script or None


@dataclass
class LanguageCheckResult:
    outcome: Outcome
    detected: str | None
    expected: str | None
    detail: str = ""

    @property
    def is_measured(self) -> bool:
        """Whether this check produced a definite pass/fail.

        `inconclusive` and `not_applicable` are excluded from scoring —
        the same reasoning `s2_retrieval.recall_at_k` uses when it returns
        None rather than 0.0 for an item with no gold chunks: an
        unmeasured item must not silently count as a failure.
        """
        return self.outcome in ("pass", "fail")


def check_output_language(
    detected: LanguageGuess | None, expected: str | None
) -> LanguageCheckResult:
    """Compare a detector's guess about some text against what was expected.

    `expected` is stratum's BCP-47-ish tag ("hi-Deva", "en"). A detector
    that can only name a script (`ScriptRangeDetector`) still lets this
    catch an unambiguous mismatch — a Devanagari answer can never satisfy
    an `en` expectation — but returns `inconclusive` rather than `pass`
    when the script alone cannot rule the expected language *out*, since
    several languages can share one script and a script match is not a
    language match. Only a detector willing to name an actual language
    (`detected.language` set) can produce a `pass`.
    """
    if expected is None:
        return LanguageCheckResult(
            "not_applicable", None, None, "no output-language expectation declared"
        )
    if detected is None:
        return LanguageCheckResult(
            "inconclusive", None, expected, "no language detector configured"
        )

    expected_lang, expected_script = _split_tag(expected)

    if detected.language is not None:
        ok = detected.language == expected_lang
        detail = (
            f"{expected_lang} confirmed" if ok
            else f"detected {detected.language}, expected {expected_lang}"
        )
        return LanguageCheckResult("pass" if ok else "fail", detected.language, expected, detail)

    if detected.script is None:
        return LanguageCheckResult(
            "inconclusive", None, expected, "answer language could not be detected"
        )

    if expected_script is not None and detected.script != expected_script:
        return LanguageCheckResult(
            "fail", f"script:{detected.script}", expected,
            f"script {detected.script} is incompatible with expected {expected}",
        )

    return LanguageCheckResult(
        "inconclusive", f"script:{detected.script}", expected,
        f"script {detected.script} is consistent with {expected} but shared by "
        f"other languages — use a language-identifying detector for a definitive check",
    )
