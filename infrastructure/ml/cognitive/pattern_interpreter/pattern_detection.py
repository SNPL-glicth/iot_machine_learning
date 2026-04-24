"""Pattern detection functions for text data.

Each function evaluates a specific pattern type from computed signal summary.
"""

from __future__ import annotations

from typing import Any, Dict, List


def detect_narrative_escalation(pattern_summary: Dict[str, Any], urgency_score: float) -> bool:
    """Detect progressive escalation from minor to critical issues.

    Requires TRUE progression — not just high overall urgency.
    Needs BOTH:
    1. Detected progression across text segments (progression_detected=True)
    2. Escalation keywords spread across multiple segments (>= 2)
    """
    progression_detected = pattern_summary.get("progression_detected", False)
    escalation_spread = pattern_summary.get("escalation_keywords_spread", 0)
    return progression_detected and escalation_spread >= 2


def detect_critical_spike(spikes: List[Any], pattern_summary: Dict[str, Any]) -> bool:
    """Detect abrupt urgency spike at specific point.

    Uses real segment-based spike detection OR structural outliers.
    """
    has_abrupt_spike = pattern_summary.get("has_abrupt_spike", False)
    if has_abrupt_spike:
        return True

    if spikes:
        max_spike_magnitude = max((spike.get("magnitude", 0) for spike in spikes), default=0)
        return max_spike_magnitude > 2.0

    return False


def detect_sustained_degradation(pattern_summary: Dict[str, Any], urgency_score: float) -> bool:
    """Detect consistently high urgency without improvement.

    Requires sustained high urgency across MULTIPLE segments,
    not just a high overall document score.
    """
    high_segments = pattern_summary.get("high_urgency_segments", 0)
    improvement_segments = pattern_summary.get("improvement_segments", 0)
    max_segment_urgency = pattern_summary.get("max_segment_urgency", 0.0)

    return (
        high_segments >= 3 and
        improvement_segments == 0 and
        max_segment_urgency >= 0.7
    )


def detect_regime_shift(change_points: List[Any], pattern_summary: Dict[str, Any]) -> bool:
    """Detect topic or tonality change in document.

    Uses real structural change points OR sentiment shift across segments.
    """
    has_structural_changes = len(change_points) > 0
    sentiment_shift = pattern_summary.get("sentiment_shift_detected", False)
    n_change_points = pattern_summary.get("n_change_points", 0)

    return (
        has_structural_changes or
        sentiment_shift or
        n_change_points > 0
    )


def detect_stable_operations(pattern_summary: Dict[str, Any], urgency_score: float) -> bool:
    """Detect stable operations without significant changes.

    Requires truly stable conditions: no progression, no spikes, no shifts,
    and low urgency across all segments.
    """
    if pattern_summary.get("progression_detected", False):
        return False
    if pattern_summary.get("has_abrupt_spike", False):
        return False
    if pattern_summary.get("sentiment_shift_detected", False):
        return False

    high_segments = pattern_summary.get("high_urgency_segments", 0)
    if high_segments > 0:
        return False

    return (
        pattern_summary.get("n_change_points", 0) == 0 and
        pattern_summary.get("n_spikes", 0) == 0
    )
