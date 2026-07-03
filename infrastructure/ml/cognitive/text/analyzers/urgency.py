"""Lexicon-based urgency / criticality analysis (Spanish-first, English fallback).

Pure function: no I/O, no global state, no external dependencies.
Guarantees: never raises, never returns None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .keyword_config import ALL_URGENCY

_WORD_PATTERN = re.compile(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+")

_HIGH_URGENCY_SCORE: float = 0.7
_MEDIUM_URGENCY_SCORE: float = 0.4
_CRITICAL_SEVERITY_HITS: int = 3


@dataclass(frozen=True)
class UrgencyResult:
    """Immutable result of urgency / criticality analysis.

    Attributes:
        score: Urgency score in [0.0, 1.0]. Higher = more urgent.
        severity: One of "critical", "warning", "info".
        total_hits: Number of urgency keyword matches.
        hits: List of matched urgency keywords (lowercase).
    """

    score: float
    severity: str
    total_hits: int
    hits: List[str]


def _tokenise(text: str) -> List[str]:
    """Split text into lowercase alphabetic tokens."""
    return [m.group().lower() for m in _WORD_PATTERN.finditer(text)]


def compute_urgency(text: str) -> UrgencyResult:
    """Score the urgency level of *text*.

    Uses a combined Spanish + English urgency keyword lexicon.
    Empty or whitespace-only text returns info-level result.

    Args:
        text: Input string to analyse.

    Returns:
        UrgencyResult — never None, never raises.
    """
    if not text or not text.strip():
        return UrgencyResult(score=0.0, severity="info", total_hits=0, hits=[])

    tokens = _tokenise(text)
    if not tokens:
        return UrgencyResult(score=0.0, severity="info", total_hits=0, hits=[])

    matched: List[str] = [t for t in tokens if t in ALL_URGENCY]
    total_hits = len(matched)

    if total_hits == 0:
        return UrgencyResult(score=0.0, severity="info", total_hits=0, hits=[])

    score = min(1.0, total_hits / len(tokens) * 10.0)

    if score >= _HIGH_URGENCY_SCORE or total_hits >= _CRITICAL_SEVERITY_HITS:
        severity = "critical"
    elif score >= _MEDIUM_URGENCY_SCORE:
        severity = "warning"
    else:
        severity = "info"

    return UrgencyResult(
        score=round(score, 4),
        severity=severity,
        total_hits=total_hits,
        hits=matched[:10],
    )
