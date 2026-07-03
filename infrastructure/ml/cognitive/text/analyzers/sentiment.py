"""Lexicon-based sentiment analysis (Spanish-first, English fallback).

Pure function: no I/O, no global state, no external dependencies.
Guarantees: never raises, never returns None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .keyword_config import ALL_NEGATIVE, ALL_POSITIVE

_WORD_PATTERN = re.compile(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+")


@dataclass(frozen=True)
class SentimentResult:
    """Immutable result of sentiment analysis.

    Attributes:
        score: Normalised sentiment score in [-1.0, 1.0].
               Negative = negative sentiment, positive = positive sentiment.
        label: One of "positive", "negative", "neutral".
        positive_count: Number of positive keyword matches.
        negative_count: Number of negative keyword matches.
    """

    score: float
    label: str
    positive_count: int
    negative_count: int


def _tokenise(text: str) -> List[str]:
    """Split text into lowercase alphabetic tokens."""
    return [m.group().lower() for m in _WORD_PATTERN.finditer(text)]


def compute_sentiment(text: str) -> SentimentResult:
    """Score the sentiment polarity of *text*.

    Uses a combined Spanish + English keyword lexicon.
    Empty or whitespace-only text returns neutral result with score 0.0.

    Args:
        text: Input string to analyse.

    Returns:
        SentimentResult — never None, never raises.
    """
    if not text or not text.strip():
        return SentimentResult(score=0.0, label="neutral", positive_count=0, negative_count=0)

    tokens = _tokenise(text)
    if not tokens:
        return SentimentResult(score=0.0, label="neutral", positive_count=0, negative_count=0)

    positive_count = sum(1 for t in tokens if t in ALL_POSITIVE)
    negative_count = sum(1 for t in tokens if t in ALL_NEGATIVE)
    total_matches = positive_count + negative_count

    if total_matches == 0:
        return SentimentResult(score=0.0, label="neutral", positive_count=0, negative_count=0)

    score = (positive_count - negative_count) / total_matches

    if score > _POSITIVE_THRESHOLD:
        label = "positive"
    elif score < _NEGATIVE_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return SentimentResult(
        score=round(score, 4),
        label=label,
        positive_count=positive_count,
        negative_count=negative_count,
    )


_POSITIVE_THRESHOLD: float = 0.15
_NEGATIVE_THRESHOLD: float = -0.15
