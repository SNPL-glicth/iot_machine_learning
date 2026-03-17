"""Change-point and spike detection on text sentence signals.

Treats sentence lengths (word count per sentence) as a numeric signal
and applies lightweight statistical methods to detect:
- **Change-points**: where narrative style shifts abruptly (mean-shift).
- **Spikes**: unusually long or short sentences (z-score outliers).

Self-contained — no IoT infrastructure dependencies.  Uses only stdlib
math.

Graceful-fail: returns empty result on error.

Single entry point: ``detect_text_patterns(sentences)``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_MIN_SENTENCES = 5
_SPIKE_Z_THRESHOLD = 2.0
_CHANGE_POINT_RATIO = 1.5


@dataclass(frozen=True)
class TextPatternResult:
    """Patterns detected in the narrative signal.

    Attributes:
        change_points: Indices where narrative style shifts.
        spikes: Indices of unusually long/short sentences.
        n_patterns: Total patterns detected.
        available: Whether pattern analysis could be computed.
        summary: Human-readable summary of findings.
        raw: Full pattern results for downstream use.
    """

    change_points: List[int] = field(default_factory=list)
    spikes: List[int] = field(default_factory=list)
    n_patterns: int = 0
    available: bool = False
    summary: str = ""
    raw: List[Dict[str, Any]] = field(default_factory=list)


def detect_text_patterns(sentences: List[str]) -> TextPatternResult:
    """Detect structural patterns in the sentence-length signal.

    Args:
        sentences: List of sentence strings.

    Returns:
        ``TextPatternResult`` with change-points and spikes.
    """
    if len(sentences) < _MIN_SENTENCES:
        return TextPatternResult()

    try:
        lengths = [float(len(s.split())) for s in sentences]

        change_points = _detect_change_points(lengths)
        spikes = _detect_spikes(lengths)

        n_patterns = len(change_points) + len(spikes)
        raw: List[Dict[str, Any]] = []

        for idx in change_points:
            raw.append({
                "type": "change_point",
                "index": idx,
                "description": f"Narrative shift at sentence {idx}",
            })
        for idx in spikes:
            raw.append({
                "type": "spike",
                "index": idx,
                "length": int(lengths[idx]),
                "description": f"Outlier sentence at position {idx} ({int(lengths[idx])} words)",
            })

        summary = _build_summary(n_patterns, change_points, spikes)

        return TextPatternResult(
            change_points=change_points,
            spikes=spikes,
            n_patterns=n_patterns,
            available=True,
            summary=summary,
            raw=raw,
        )
    except Exception as exc:
        logger.warning("[TEXT_PATTERN] Pattern detection failed: %s", exc)
        return TextPatternResult()


# ── Internal detectors ───────────────────────────────────────────


def _detect_change_points(lengths: List[float]) -> List[int]:
    """Detect mean-shift change-points using a sliding window.

    Compares the mean of a left window to the mean of a right window.
    If the ratio exceeds ``_CHANGE_POINT_RATIO``, marks as change-point.
    """
    n = len(lengths)
    if n < 6:
        return []

    window = max(3, n // 5)
    change_points: List[int] = []

    for i in range(window, n - window):
        left_mean = sum(lengths[i - window:i]) / window
        right_mean = sum(lengths[i:i + window]) / window

        if left_mean == 0 and right_mean == 0:
            continue

        denom = max(left_mean, right_mean, 1e-9)
        numer = min(left_mean, right_mean, denom)
        ratio = denom / numer

        if ratio >= _CHANGE_POINT_RATIO:
            # Avoid marking consecutive indices
            if not change_points or (i - change_points[-1]) >= window:
                change_points.append(i)

    return change_points


def _detect_spikes(lengths: List[float]) -> List[int]:
    """Detect outlier sentences using z-score."""
    n = len(lengths)
    mean = sum(lengths) / n
    variance = sum((x - mean) ** 2 for x in lengths) / n
    std = math.sqrt(variance) if variance > 0 else 0.0

    if std < 1e-9:
        return []

    spikes: List[int] = []
    for i, length in enumerate(lengths):
        z = abs(length - mean) / std
        if z >= _SPIKE_Z_THRESHOLD:
            spikes.append(i)

    return spikes


def _build_summary(
    n_patterns: int,
    change_points: List[int],
    spikes: List[int],
) -> str:
    """Build human-readable pattern summary."""
    if n_patterns == 0:
        return "Narrative structure is consistent throughout the document."

    parts: List[str] = []

    if change_points:
        parts.append(
            f"{len(change_points)} narrative shift(s) detected "
            f"(at sentence {', '.join(str(i) for i in change_points[:5])})"
        )

    if spikes:
        parts.append(
            f"{len(spikes)} unusually long/short sentence(s) "
            f"(at position {', '.join(str(i) for i in spikes[:5])})"
        )

    return ". ".join(parts) + "." if parts else ""
