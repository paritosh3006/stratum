"""Detector backends: script ranges (always available, no model), IndicLID
(real language id for script-confusable languages, optional dependency),
and a deterministic stub for tests.

Same split as `judges/backends.py` and the reference system's
`retrieval/embedder.py`: a real backend that needs something installed,
and a no-download stand-in so the plumbing has an offline leg to stand on.
"""

from __future__ import annotations

import re
from typing import Callable

from .base import LanguageDetector, LanguageGuess

#: Unicode block ranges mapped to an ISO-15924 script tag, in the order
#: they're tested. First match wins — the same "no post-hoc guessing on
#: mixed-script text" stance as the reference system's script detector.
#: The comment against each range names the languages stratum's example
#: datasets use that share it; it documents the ambiguity, it is not a
#: claim this detector can resolve it.
_SCRIPT_RANGES: list[tuple[str, re.Pattern[str], str]] = [
    ("Deva", re.compile(r"[ऀ-ॿ]"), "shared by hi, mr, brx, ne, kok"),
    ("Beng", re.compile(r"[ঀ-৿]"), "shared by bn, as"),
    ("Arab", re.compile(r"[؀-ۿ]"), "shared by ur, ks, sd"),
    ("Guru", re.compile(r"[਀-੿]"), "pa"),
    ("Gujr", re.compile(r"[઀-૿]"), "gu"),
    ("Taml", re.compile(r"[஀-௿]"), "ta"),
    ("Telu", re.compile(r"[ఀ-౿]"), "te"),
    ("Knda", re.compile(r"[ಀ-೿]"), "kn"),
    ("Mlym", re.compile(r"[ഀ-ൿ]"), "ml"),
    ("Latn", re.compile(r"[A-Za-z]"), "shared by en and every romanized language"),
]


class ScriptRangeDetector:
    """Unicode block ranges. Identifies script, not language — by design.

    Generalises examples/reference_system's Devanagari-only heuristic to
    every script range stratum's datasets use, and moves it behind the
    `LanguageDetector` protocol so core depends on the interface rather
    than one script's worth of regex.

    `LanguageGuess.language` is always None here. Returning a language
    guess derived only from script would misreport a Bodo answer as Hindi
    — exactly the bug class this exists to stop core from making silently.
    Reach for `IndicLIDLanguageDetector` (or another real language-id
    backend) when script-confusable languages need to be told apart.
    """

    detector_id = "script-range"

    def detect(self, text: str) -> LanguageGuess:
        for script, pattern, _shared_by in _SCRIPT_RANGES:
            if pattern.search(text):
                return LanguageGuess(language=None, script=script, detector_id=self.detector_id)
        return LanguageGuess(language=None, script=None, detector_id=self.detector_id)


class IndicLIDLanguageDetector:
    """Real language identification for script-confusable Indic languages
    (Hindi vs Bodo vs Marathi vs Nepali vs Konkani, all Devanagari) — the
    call `ScriptRangeDetector` refuses to make.

    IndicLID (AI4Bharat) ships as a research codebase — a fastText script
    classifier feeding a transformer language-id head, distributed as
    checkpoints rather than a pip-installable package with a stable
    inference API. Vendoring a specific checkpoint's loading code into
    stratum core would hardcode one lab's model format into a library that
    otherwise has zero ML dependencies at install time, and would still go
    stale the moment that repo's format changes.

    What's fixed here is the contract instead: construct with a
    `predict_fn(text) -> (language_code, confidence)` that wraps whatever
    inference session the caller has already loaded (IndicLID, a
    fine-tuned fastText model, a hosted API — this class doesn't care),
    and `detect()` adapts its output to `LanguageGuess`. A basic
    `pip install stratum-eval` never imports torch or downloads a model
    because of this class; it only fails, loudly, if constructed without
    a `predict_fn`.
    """

    detector_id = "indiclid"

    def __init__(self, predict_fn: Callable[[str], tuple[str, float]] | None = None) -> None:
        if predict_fn is None:
            raise RuntimeError(
                "IndicLIDLanguageDetector needs predict_fn(text) -> "
                "(language_code, confidence) — stratum core does not vendor "
                "the IndicLID model itself. Load IndicLID (or another "
                "language-id model) yourself and pass its prediction call "
                "in as predict_fn; see this class's docstring for why."
            )
        self._predict = predict_fn

    def detect(self, text: str) -> LanguageGuess:
        language, confidence = self._predict(text)
        return LanguageGuess(
            language=language, script=None, confidence=confidence,
            detector_id=self.detector_id,
        )


class StubLanguageDetector:
    """Deterministic, offline, no downloads, no model — the language
    equivalent of `HashingEmbedder` and `StubJudge`. It exists so the
    output-language plumbing (`EvalItem` fields, the evaluator, stage
    attribution) has an offline leg to test against, not to produce a real
    language-id number.

    `overrides` lets a test pin the exact guess for a given exact answer
    text — e.g. to simulate the production bug this feature was built for
    (a Hindi query answered in Bodo) without a real Devanagari-capable
    language-id model in the test suite. Text not in `overrides` falls
    back to `default`, so a whole suite of "everything comes back in
    language X" tests needs no per-item wiring either.
    """

    detector_id = "stub-language"

    def __init__(self, overrides: dict[str, str] | None = None, default: str | None = None) -> None:
        self.overrides = overrides or {}
        self.default = default

    def detect(self, text: str) -> LanguageGuess:
        return LanguageGuess(
            language=self.overrides.get(text, self.default), detector_id=self.detector_id
        )


def get_language_detector(name: str = "script") -> LanguageDetector:
    if name in {"script", "script-range"}:
        return ScriptRangeDetector()
    if name == "stub":
        return StubLanguageDetector()
    if name == "indiclid":
        raise ValueError(
            "indiclid needs a predict_fn — construct IndicLIDLanguageDetector "
            "directly rather than through get_language_detector(name)"
        )
    raise ValueError(f"unknown language detector: {name}")
