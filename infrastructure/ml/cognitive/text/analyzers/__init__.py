"""Text analyzers for cognitive text processing.

This package contains pure functions for analyzing text sentiment,
urgency, readability, and structural properties.
"""

from .text_sentiment import SentimentResult, compute_sentiment
from .text_urgency import UrgencyResult, compute_urgency
from .text_readability import ReadabilityResult, compute_readability
from .text_structural import TextStructuralResult, compute_text_structure
from .text_pattern import compute_text_patterns
from .keyword_config import (
    URGENCY_KEYWORDS_ES,
    URGENCY_KEYWORDS_EN,
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    DOMAIN_KEYWORDS,
    ACTION_TEMPLATES,
)

__all__ = [
    "SentimentResult",
    "compute_sentiment",
    "UrgencyResult", 
    "compute_urgency",
    "ReadabilityResult",
    "compute_readability",
    "TextStructuralResult",
    "compute_text_structure",
    "compute_text_patterns",
    "URGENCY_KEYWORDS_ES",
    "URGENCY_KEYWORDS_EN",
    "POSITIVE_WORDS",
    "NEGATIVE_WORDS",
    "DOMAIN_KEYWORDS",
    "ACTION_TEMPLATES",
]
