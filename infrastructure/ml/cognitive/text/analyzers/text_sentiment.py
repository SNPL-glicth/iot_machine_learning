"""Sentiment scoring for text documents.

Pure function — no side effects, no I/O.
Uses keyword matching against configurable word lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .keyword_config import POSITIVE_WORDS, NEGATIVE_WORDS


@dataclass(frozen=True)
class SentimentResult:
    """Result of sentiment analysis.

    Attributes:
        score: Sentiment score in [-1.0, 1.0].
        label: ``"positive"`` | ``"neutral"`` | ``"negative"``.
        positive_count: Number of positive keyword matches.
        negative_count: Number of negative keyword matches.
    """

    score: float
    label: str
    positive_count: int
    negative_count: int


def compute_sentiment(text: str) -> SentimentResult:
    """Compute sentiment from keyword matching.

    Args:
        text: Raw text content.

    Returns:
        ``SentimentResult`` with score, label, and counts.
    """
    text_lower = text.lower()

    pos_count = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in text_lower)

    total = pos_count + neg_count
    score = (pos_count - neg_count) / max(total, 1)

    if score < -0.2:
        label = "negative"
    elif score > 0.2:
        label = "positive"
    else:
        label = "neutral"

    return SentimentResult(
        score=round(score, 3),
        label=label,
        positive_count=pos_count,
        negative_count=neg_count,
    )
