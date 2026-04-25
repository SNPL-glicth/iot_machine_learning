"""Sentiment scoring for text documents.

Loads keywords from external JSON. Handles negation and intensification.
Pure function — no side effects except config JSON read at import time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


@dataclass(frozen=True)
class SentimentResult:
    """Result of sentiment analysis.

    Attributes:
        score: Sentiment score in [-1.0, 1.0].
        label: ``"positive"`` | ``"neutral"`` | ``"negative"``.
        positive_count: Weighted positive keyword matches.
        negative_count: Weighted negative keyword matches.
    """

    score: float
    label: str
    positive_count: float
    negative_count: float


def _load_sentiment_config() -> Dict:
    cfg_path = Path(__file__).parent / "data" / "sentiment_keywords.json"
    with cfg_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


_CONFIG = _load_sentiment_config()

_POSITIVE_ES: Set[str] = set(_CONFIG["positive"]["es"])
_POSITIVE_EN: Set[str] = set(_CONFIG["positive"]["en"])
_NEGATIVE_ES: Set[str] = set(_CONFIG["negative"]["es"])
_NEGATIVE_EN: Set[str] = set(_CONFIG["negative"]["en"])
_INTENSIFIERS: Set[str] = set(
    _CONFIG["intensifiers"]["es"] + _CONFIG["intensifiers"]["en"]
)
_NEGATORS: Set[str] = set(
    _CONFIG["negators"]["es"] + _CONFIG["negators"]["en"]
)


def _word_hits(word: str, keywords: Set[str]) -> float:
    """Count how many keywords appear as substrings in *word*."""
    return sum(1 for kw in keywords if kw in word)


def compute_sentiment(text: str) -> SentimentResult:
    """Compute sentiment from keyword matching with negation/intensification.

    Args:
        text: Raw text content.

    Returns:
        ``SentimentResult`` with score, label, and counts.
    """
    words = text.lower().split()
    pos_count = 0.0
    neg_count = 0.0

    for i, w in enumerate(words):
        base = 1.0
        if i > 0 and words[i - 1] in _INTENSIFIERS:
            base = 1.5
        if i > 0 and words[i - 1] in _NEGATORS:
            base = -base

        pos_hits = _word_hits(w, _POSITIVE_ES) + _word_hits(w, _POSITIVE_EN)
        neg_hits = _word_hits(w, _NEGATIVE_ES) + _word_hits(w, _NEGATIVE_EN)

        if base < 0:
            pos_count += abs(base) * neg_hits
            neg_count += abs(base) * pos_hits
        else:
            pos_count += base * pos_hits
            neg_count += base * neg_hits

    total = pos_count + neg_count
    score = (pos_count - neg_count) / max(total, 1)
    score = max(-1.0, min(1.0, score))

    if score < -0.2:
        label = "negative"
    elif score > 0.2:
        label = "positive"
    else:
        label = "neutral"

    return SentimentResult(
        score=round(score, 3),
        label=label,
        positive_count=round(pos_count, 2),
        negative_count=round(neg_count, 2),
    )
