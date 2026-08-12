"""S4 — output rendering metrics.

This is where multilingual RAG usually breaks, and where nothing needs an
LLM judge. Every check here is mechanical, deterministic, and cheap.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

PLACEHOLDER_RE = re.compile(r"\{[^{}\s]+\}|\{\{[^{}]+\}\}|%\([^)]+\)s|%[sd]|\$\{[^}]+\}")

# Digit families that show up in Indic output. Comparing numerals across
# scripts requires folding these to ASCII first, otherwise a correct
# translation reads as a mismatch.
_DIGIT_FOLD = {
    ord(c): str(i % 10)
    for block_start in (0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6,
                        0x0C66, 0x0CE6, 0x0D66, 0x0660, 0x06F0)
    for i, c in enumerate(chr(cp) for cp in range(block_start, block_start + 10))
}

NUMERIC_RE = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass
class CheckResult:
    passed: bool
    detail: str = ""
    missing: list[str] = field(default_factory=list)
    corrupted: list[str] = field(default_factory=list)


def fold_digits(text: str) -> str:
    """Map Devanagari/Tamil/Arabic-Indic digits to ASCII."""
    return unicodedata.normalize("NFC", text).translate(_DIGIT_FOLD)


# --------------------------------------------------------------------------
# Placeholder integrity
# --------------------------------------------------------------------------

def check_placeholders(
    source: str, output: str, declared: list[str] | None = None
) -> CheckResult:
    """Every placeholder in the source must survive verbatim in the output.

    This is a bug class, not a score: a translated placeholder identifier
    breaks downstream string formatting at runtime. Treated as pass/fail
    per item, and gated at 100% rather than averaged into a soft number.
    """
    expected = declared if declared is not None else PLACEHOLDER_RE.findall(source)
    if not expected:
        return CheckResult(True, "no placeholders in source")

    missing = [p for p in expected if p not in output]
    if not missing:
        return CheckResult(True, f"{len(expected)} preserved")

    # Distinguish "dropped" from "translated" — the fix differs.
    found_in_output = set(PLACEHOLDER_RE.findall(output))
    corrupted = [
        p for p in missing
        if found_in_output and p not in found_in_output and len(found_in_output) > 0
    ]
    kind = "translated/corrupted" if corrupted else "dropped"
    return CheckResult(
        False,
        f"{len(missing)}/{len(expected)} {kind}",
        missing=missing,
        corrupted=corrupted,
    )


# --------------------------------------------------------------------------
# Glossary adherence
# --------------------------------------------------------------------------

class Glossary:
    """Approved renderings for domain terms, per language.

    {"premium": {"hi-Deva": ["प्रीमियम"], "ta-Taml": ["பிரீமியம்"]}}

    `forbidden` catches the specific failure of a term drifting between
    several plausible-but-inconsistent renderings within one answer.
    """

    def __init__(
        self,
        terms: dict[str, dict[str, list[str]]],
        forbidden: dict[str, dict[str, list[str]]] | None = None,
    ) -> None:
        self.terms = terms
        self.forbidden = forbidden or {}

    @classmethod
    def from_dict(cls, raw: dict) -> "Glossary":
        return cls(raw.get("terms", {}), raw.get("forbidden", {}))


def check_glossary(
    source: str, output: str, glossary: Glossary, language: str
) -> CheckResult:
    """Domain terms present in the source must appear in approved form."""
    src_lower = source.lower()
    in_scope = [t for t in glossary.terms if t.lower() in src_lower]
    if not in_scope:
        return CheckResult(True, "no glossary terms in scope")

    violations: list[str] = []
    for term in in_scope:
        approved = glossary.terms[term].get(language, [])
        if not approved:
            continue
        if not any(a in output for a in approved):
            violations.append(term)
            continue
        # Approved form present, but check no unapproved variant also appears.
        banned = glossary.forbidden.get(term, {}).get(language, [])
        if any(b in output for b in banned):
            violations.append(f"{term} (drift)")

    if violations:
        return CheckResult(
            False,
            f"{len(violations)}/{len(in_scope)} terms off-glossary",
            missing=violations,
        )
    return CheckResult(True, f"{len(in_scope)} terms correct")


# --------------------------------------------------------------------------
# Numeral integrity
# --------------------------------------------------------------------------

def check_numerals(source: str, output: str, declared: list[str] | None = None) -> CheckResult:
    """Numbers must survive translation, allowing for script differences.

    Digit-grouping conventions differ (1,50,000 vs 150,000), so separators
    are stripped before comparison — a formatting change is not an error,
    a changed value is.
    """
    def norm(text: str) -> list[str]:
        return [m.replace(",", "").replace(".", "") for m in NUMERIC_RE.findall(fold_digits(text))]

    expected = declared if declared is not None else norm(source)
    if not expected:
        return CheckResult(True, "no numerals in source")

    expected = [e.replace(",", "").replace(".", "") for e in expected]
    got = norm(output)
    missing = [e for e in expected if e not in got]

    if missing:
        return CheckResult(
            False, f"{len(missing)}/{len(expected)} numerals lost or altered", missing=missing
        )
    return CheckResult(True, f"{len(expected)} preserved")


# --------------------------------------------------------------------------
# Entity preservation (S1, but same machinery)
# --------------------------------------------------------------------------

def check_entities(output: str, entities: list[str]) -> CheckResult:
    """Named entities that must appear verbatim (policy numbers, act names)."""
    if not entities:
        return CheckResult(True, "no entities declared")
    missing = [e for e in entities if e not in output]
    if missing:
        return CheckResult(False, f"{len(missing)}/{len(entities)} entities lost", missing=missing)
    return CheckResult(True, f"{len(entities)} preserved")
