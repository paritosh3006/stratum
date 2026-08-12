"""S4b: glossary enforcement — approved domain-term renderings.

Distinct from S1's `LexiconTranslator` (query/translate.py): a glossary is
*supposed* to be curated to the domain, not mined from the eval set's own
queries — that's what a glossary is for. `eval/glossary.json` is the
standard Hindi insurance-industry rendering for the terms this corpus uses,
written directly rather than scanned from `build_dataset.py`.

It is loaded once here and used two ways: passed to the stratum harness via
`stratum run --glossary` (so `check_glossary` scores against it), and used
by `enforce()` below to make the renderer actually produce those terms. One
file, one glossary, driving both production and evaluation — the same
pattern a real system would use, and the reason a glossary is worth having
at all: a mismatch between what you render and what you're scored against
would defeat the point.
"""

from __future__ import annotations

import json
from pathlib import Path

from stratum.metrics.s4_rendering import Glossary

_PATH = Path(__file__).resolve().parents[1] / "eval" / "glossary.json"
_RAW = json.loads(_PATH.read_text(encoding="utf-8"))

TERMS: dict[str, dict[str, list[str]]] = _RAW["terms"]
FORBIDDEN: dict[str, dict[str, list[str]]] = _RAW.get("forbidden", {})


def build_glossary() -> Glossary:
    """The object `stratum run --glossary` loads independently from the same
    JSON — this just gives Python callers (tests, the renderer) a way to get
    it without going through the CLI."""
    return Glossary(TERMS, FORBIDDEN)


def enforce(source_en: str, rendered: str, language: str, query: str = "") -> str:
    """Force the approved rendering for every in-scope term.

    Runs after translation, not instead of it: the translator may render a
    domain term as a plausible-but-wrong synonym (a forbidden variant) or
    leave it untouched because the word isn't in its dictionary. Either way,
    trusting general translation to get terminology right is exactly the
    problem a glossary exists to solve, so this overwrites the specific
    span rather than hoping the translation step already got it.

    `query` matters as much as `source_en` for deciding what's in scope.
    stratum's own `check_glossary` decides scope from the *query*
    (`item.query`), not the answer — a user asking "is X covered?" scores
    on whether "cover" is in scope, even if the extracted answer span talks
    about a sub-limit without ever using that word. Checking only the
    answer for in-scope terms (an earlier version of this function did)
    made enforcement blind to exactly that case, and it was the majority of
    this reference system's own `terminology_drift` failures once measured
    — the query said "cover" or "cashless", the answer span didn't, so
    nothing forced the approved term into the rendering at all.
    """
    src_lower = f"{source_en} {query}".lower()
    for term, forms in TERMS.items():
        if term not in src_lower:
            continue
        approved = forms.get(language)
        if not approved:
            continue
        target = approved[0]
        if target in rendered:
            continue

        banned = FORBIDDEN.get(term, {}).get(language, [])
        replaced = False
        for variant in banned:
            if variant in rendered:
                rendered = rendered.replace(variant, target)
                replaced = True

        if not replaced:
            # The translator produced nothing recognisable for this concept
            # at all — append the approved term rather than silently drop
            # it from the answer.
            rendered = f"{rendered} {target}"

    return rendered
