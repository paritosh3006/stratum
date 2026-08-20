"""stratum — per-stage evaluation for multilingual RAG."""

from .attribution import Cascade, build_cascade, render_cascade
from .dataset import Dataset, EvalItem
from .endpoint import Capabilities, CallableEndpoint, HttpEndpoint, RagResponse
from .harness import Harness
from .judges import Calibration, CalibrationRegistry
from .language import LanguageDetector, LanguageGuess, ScriptRangeDetector
from .report import Gate, Report
from .stats import Estimate, bootstrap_difference, bootstrap_mean

__version__ = "0.1.0"
__all__ = [
    "Dataset", "EvalItem", "Capabilities", "CallableEndpoint", "HttpEndpoint",
    "RagResponse", "Harness", "Report", "Gate", "Cascade", "build_cascade",
    "render_cascade", "Estimate", "bootstrap_mean", "bootstrap_difference",
    "Calibration", "CalibrationRegistry",
    "LanguageDetector", "LanguageGuess", "ScriptRangeDetector",
]
