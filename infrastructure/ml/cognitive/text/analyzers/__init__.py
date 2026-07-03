"""Text sub-analyzers: sentiment, urgency, readability, structure.

Each module exports a single public function and its result dataclass.
All functions are pure: no I/O, no side effects, no global state.
"""

from .keyword_config import URGENCY_KEYWORDS_EN, URGENCY_KEYWORDS_ES
from .readability import ReadabilityResult, compute_readability
from .sentiment import SentimentResult, compute_sentiment
from .text_structure import StructureResult, compute_text_structure
from .urgency import UrgencyResult, compute_urgency

__all__ = [
    "compute_sentiment",
    "SentimentResult",
    "compute_urgency",
    "UrgencyResult",
    "compute_readability",
    "ReadabilityResult",
    "compute_text_structure",
    "StructureResult",
    "URGENCY_KEYWORDS_ES",
    "URGENCY_KEYWORDS_EN",
]
