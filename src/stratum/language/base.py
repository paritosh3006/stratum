"""Language detection backends.

Output-language expectation is a metric like any other in stratum:
declared per item, measured deterministically where possible, and honest
about what it cannot measure. What's unusual here is that "measuring"
means running a detector over free text, and detectors vary enormously in
what they can actually tell apart.

Devanagari is shared by Hindi, Marathi, Bodo, Nepali and Konkani. Bengali
script by Bengali and Assamese. Perso-Arabic by Urdu, Kashmiri and Sindhi.
A detector built on Unicode block ranges alone can name the script with
certainty, and that's enough to rule out a language written in a
*different* script — but it cannot tell Hindi from Bodo, both Devanagari.
Claiming it could would turn a script check into a language claim it has
no basis for, which is exactly the production bug (`wrong_output_language`
between hi-Deva and brx) that this module exists to catch, not repeat.

`stratum.metrics.language` depends on the `LanguageDetector` protocol
below, never on a specific backend, so a new detector is a new class in
`backends.py`, not a new branch in the evaluator or the harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LanguageGuess:
    """What a detector could tell about one piece of text.

    `language` is the BCP-47-ish language subtag stratum datasets use
    ("hi", "brx", "en") — the detector's actual claim about which language
    this is. None when the detector cannot or, by design, will not name
    one (a script-only detector never sets this).

    `script` is the ISO-15924-ish script tag ("Deva", "Beng", "Latn") when
    identifiable. It is independent of `language`: a detector can be
    certain of script while refusing to guess language, which is precisely
    the honest middle ground a shared-script family requires.
    """

    language: str | None
    script: str | None = None
    confidence: float = 1.0
    detector_id: str = ""


class LanguageDetector(Protocol):
    """Any backend that can look at text and say what language it's in."""

    detector_id: str

    def detect(self, text: str) -> LanguageGuess: ...
