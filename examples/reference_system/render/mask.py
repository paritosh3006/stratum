"""S4a + S4c: placeholder mask/restore and numeral formatting.

Render can corrupt a placeholder identifier or a number exactly the way
translation can — a word-for-word gloss has no way to know `{claim_id}` is
not a word, and a transliteration table has no way to know `500000` is not
a syllable. The fix is the same for both: hide anything that must survive
verbatim (or be reformatted deliberately, for numerals) behind a sentinel
no dictionary lookup or transliteration table has any reason to touch, run
translation/transliteration on the masked text, then substitute the real
text back in afterwards.

Uses stratum's own `PLACEHOLDER_RE`/`NUMERIC_RE` rather than redefining
them, so masking always agrees with what `check_placeholders`/
`check_numerals` actually look for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from stratum.metrics.s4_rendering import NUMERIC_RE, PLACEHOLDER_RE

#: One Unicode Private Use Area codepoint per masked span — not brackets
#: around a letter/digit counter. A bracket-plus-letter sentinel looked safe
#: in isolation but wasn't: render/translate_en.py's EN->HI tokenizer splits
#: on letter/non-letter boundaries, so `⟦A⟧` split into `⟦`, `A`, `⟧`, and
#: the bare `A` collided with the dictionary's "a" -> "" (article, dropped)
#: entry — silently erasing the sentinel's content. A single opaque
#: character has no internal boundary for any tokenizer in this pipeline to
#: split on, and it isn't a Latin letter, a digit, or Devanagari, so no
#: table or lexicon here has any entry that could match it.
_SENTINEL_BASE = 0xE000  # start of the Unicode Private Use Area


def _sentinel(n: int) -> str:
    return chr(_SENTINEL_BASE + n)


@dataclass
class Masked:
    text: str
    restore: dict[str, str] = field(default_factory=dict)


def indian_grouping(digits: str) -> str:
    """Reformat a bare digit string using the Indian numbering convention:
    the last three digits, then groups of two (5,00,000 — not 500,000).

    stratum's `check_numerals` strips separators before comparing values
    (see its docstring: "a formatting change is not an error, a changed
    value is"), so regrouping digits during render is exactly the kind of
    change that check is designed to treat as a non-error.
    """
    if len(digits) <= 3:
        return digits
    last3, rest = digits[-3:], digits[:-3]
    groups: list[str] = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return ",".join(groups + [last3])


def _format_numeral(raw: str) -> str:
    """Strip any separators already present, then apply Indian grouping.
    A decimal remainder (rare in this corpus) is kept as-is, ungrouped."""
    whole, _, frac = raw.partition(".")
    whole = whole.replace(",", "")
    grouped = indian_grouping(whole)
    return f"{grouped}.{frac}" if frac else grouped


def _masking_pass(
    text: str,
    pattern,
    value_for: Callable[[str], str],
    restore: dict[str, str],
    counter: int,
) -> tuple[str, int]:
    """Replace every regex match with a fresh sentinel, recording what to
    restore it to. Returns the masked text and the next free counter value
    so a second pass (numerals, after placeholders) doesn't reuse tokens."""

    def repl(m) -> str:
        nonlocal counter
        token = _sentinel(counter)
        restore[token] = value_for(m.group(0))
        counter += 1
        return token

    return pattern.sub(repl, text), counter


def mask(text: str) -> Masked:
    """Mask placeholders (verbatim) and numerals (for Indian-format
    restoration). Run translation/transliteration on `.text`, then call
    `unmask` with `.restore` once that's done."""
    restore: dict[str, str] = {}
    masked, n = _masking_pass(text, PLACEHOLDER_RE, lambda s: s, restore, 0)
    masked, _ = _masking_pass(masked, NUMERIC_RE, _format_numeral, restore, n)
    return Masked(masked, restore)


def unmask(text: str, restore: dict[str, str]) -> str:
    for token, original in restore.items():
        text = text.replace(token, original)
    return text
