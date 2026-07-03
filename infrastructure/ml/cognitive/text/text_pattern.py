"""Text pattern detection — escalation, spikes, shifts, degradation.

Analyses sentence-length sequences to detect structural patterns:
narrative escalation, critical spikes, sentiment shifts, context changes.

Pure function: no I/O, no global state.
Guarantees: never raises, never returns None.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

_MIN_PATTERN_WINDOW: int = 3
_Z_SCORE_THRESHOLD: float = 1.5
_SHIFT_THRESHOLD_MULTIPLIER: float = 1.5
_MIN_SEGMENTS_FOR_SHIFT: int = 4


@dataclass(frozen=True)
class PatternResult:
    """Immutable result of text pattern analysis.

    Attributes:
        n_patterns: Number of pattern types detected (0-4).
        change_points: Sentence indices where structural shifts occur.
        spikes: Sentence indices identified as outlier spikes in length.
        available: Whether the input had enough data for analysis.
        summary: Human-readable dict of detected pattern signals.
    """

    n_patterns: int
    change_points: List[int]
    spikes: List[int]
    available: bool
    summary: Dict[str, Any] = field(default_factory=dict)


def _sentence_lengths(sentences: List[str]) -> np.ndarray:
    """Convert sentences to a numpy array of word counts."""
    return np.array([len(s.split()) for s in sentences], dtype=np.float64)


def _detect_spike_indices(lengths: np.ndarray) -> List[int]:
    """Find sentence indices whose length deviates > Z_SCORE_THRESHOLD from mean."""
    if len(lengths) < _MIN_PATTERN_WINDOW:
        return []
    mean = np.mean(lengths)
    std = np.std(lengths)
    if std < 0.5:
        return []
    z_scores = np.abs((lengths - mean) / std)
    return [int(i) for i in range(len(z_scores)) if z_scores[i] > _Z_SCORE_THRESHOLD]


def _detect_change_points(lengths: np.ndarray) -> List[int]:
    """Detect structural regime shifts using segment-mean comparison.

    Compares the mean of each contiguous half-window against the overall mean.
    Returns indices where a shift is detected.
    """
    n = len(lengths)
    if n < _MIN_SEGMENTS_FOR_SHIFT:
        return []
    global_mean = np.mean(lengths)
    points: List[int] = []
    window = max(2, n // 4)
    for i in range(window, n - window):
        left_mean = np.mean(lengths[i - window : i])
        right_mean = np.mean(lengths[i : i + window])
        if abs(left_mean - right_mean) > _SHIFT_THRESHOLD_MULTIPLIER * np.std(lengths):
            points.append(i)
    return points


def _build_summary(
    lengths: np.ndarray,
    spikes: List[int],
    change_points: List[int],
) -> Dict[str, Any]:
    """Build a human-readable summary dict of detected patterns."""
    n = len(lengths)
    summary: Dict[str, Any] = {
        "n_sentences": n,
        "mean_length": float(round(np.mean(lengths), 2)) if n > 0 else 0.0,
        "std_length": float(round(np.std(lengths), 2)) if n > 1 else 0.0,
    }

    if spikes:
        summary["has_abrupt_spike"] = True
        summary["spike_count"] = len(spikes)
        summary["spike_indices"] = spikes
    else:
        summary["has_abrupt_spike"] = False

    if change_points:
        has_escalation = any(
            i > 0 and i < n - 1
            and lengths[i] > np.mean(lengths[:i])
            and lengths[i] > np.mean(lengths[i:])
            for i in change_points
        )
        summary["has_escalation"] = has_escalation
        summary["n_change_points"] = len(change_points)
    else:
        summary["has_escalation"] = False
        summary["n_change_points"] = 0

    return summary


def detect_text_patterns(sentences: List[str]) -> PatternResult:
    """Analyse sentence-level patterns in *sentences*.

    Detects:
    - **Spikes**: sentences that are unusually long/short (z-score outlier).
    - **Change points**: structural regime shifts in sentence length.

    Args:
        sentences: List of sentence strings. May be empty.

    Returns:
        PatternResult — never None, never raises.
        Empty input returns available=False.
    """
    if not sentences or len(sentences) < _MIN_PATTERN_WINDOW:
        return PatternResult(
            n_patterns=0,
            change_points=[],
            spikes=[],
            available=False,
            summary={
                "n_sentences": len(sentences) if sentences else 0,
                "has_abrupt_spike": False,
                "has_escalation": False,
                "n_change_points": 0,
            },
        )

    lengths = _sentence_lengths(sentences)
    spikes = _detect_spike_indices(lengths)
    change_points = _detect_change_points(lengths)
    summary = _build_summary(lengths, spikes, change_points)

    n_patterns = 0
    if spikes:
        n_patterns += 1
    if change_points:
        n_patterns += 1

    return PatternResult(
        n_patterns=n_patterns,
        change_points=change_points,
        spikes=sorted(spikes),
        available=True,
        summary=summary,
    )
