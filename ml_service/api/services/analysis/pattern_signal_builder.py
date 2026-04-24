"""Build real pattern signals from text segment analysis.

Splits text into segments, computes per-segment urgency/sentiment,
and detects progression, spikes, shifts, and degradation from real
indicators instead of deriving everything from the overall urgency_score.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def build_real_pattern_signals(
    full_text: str,
    urgency: Any,
    sentiment: Any,
    structural: Any,
    patterns: Any,
) -> Dict[str, Any]:
    """Build real pattern signals from text segment analysis.

    Returns:
        pattern_summary dict with real detected signals.
    """
    segments = _split_segments(full_text)
    n_segments = len(segments)
    if n_segments == 0:
        return _empty_signal()

    segment_urgencies, segment_sentiments, high_urgency_segments, improvement_segments, escalation_keywords_spread = _compute_segment_urgencies(segments)

    progression_detected = _detect_progression(segment_urgencies, n_segments)
    has_abrupt_spike = _detect_spikes(segment_urgencies, n_segments)
    sentiment_shift_detected = _detect_sentiment_shift(segment_sentiments, n_segments)

    n_change_points = len(patterns.change_points) if patterns and hasattr(patterns, "change_points") else 0
    n_spikes = len(patterns.spikes) if patterns and hasattr(patterns, "spikes") else 0

    logger.info(
        f"[PATTERN_SIGNALS] segments={n_segments}, progression={progression_detected}, "
        f"spike={has_abrupt_spike}, shift={sentiment_shift_detected}, "
        f"high_segments={high_urgency_segments}, change_points={n_change_points}"
    )

    return {
        "progression_detected": progression_detected,
        "escalation_keywords_spread": escalation_keywords_spread,
        "max_segment_urgency": max(segment_urgencies) if segment_urgencies else 0.0,
        "min_segment_urgency": min(segment_urgencies) if segment_urgencies else 0.0,
        "high_urgency_segments": high_urgency_segments,
        "sentiment_shift_detected": sentiment_shift_detected,
        "n_change_points": n_change_points,
        "n_spikes": n_spikes,
        "has_abrupt_spike": has_abrupt_spike,
        "improvement_segments": improvement_segments,
        "segment_sentiments": segment_sentiments,
        "segment_urgencies": segment_urgencies,
    }


def _split_segments(full_text: str) -> List[str]:
    """Split text into segments (paragraphs or sentences)."""
    segments = [s.strip() for s in full_text.split("\n\n") if s.strip()]
    if len(segments) < 2:
        segments = [s.strip() for s in full_text.split(".") if len(s.strip()) > 10]
    return segments


def _empty_signal() -> Dict[str, Any]:
    """Return empty signal when no segments found."""
    return {
        "progression_detected": False,
        "escalation_keywords_spread": 0,
        "max_segment_urgency": 0.0,
        "min_segment_urgency": 0.0,
        "high_urgency_segments": 0,
        "sentiment_shift_detected": False,
        "n_change_points": 0,
        "n_spikes": 0,
        "has_abrupt_spike": False,
        "improvement_segments": 0,
        "segment_sentiments": [],
        "segment_urgencies": [],
    }


def _compute_segment_urgencies(segments: List[str]) -> Tuple[List[float], List[str], int, int, int]:
    """Compute per-segment urgency, sentiment, and keyword counts.

    Returns:
        (segment_urgencies, segment_sentiments, high_urgency_segments,
         improvement_segments, escalation_keywords_spread)
    """
    from iot_machine_learning.infrastructure.ml.cognitive.text.analyzers import compute_urgency, compute_sentiment
    from iot_machine_learning.infrastructure.ml.cognitive.text.analyzers.keyword_config import URGENCY_KEYWORDS_ES, URGENCY_KEYWORDS_EN

    all_urgency_kws = list(dict.fromkeys(URGENCY_KEYWORDS_ES + URGENCY_KEYWORDS_EN))

    segment_urgencies: List[float] = []
    segment_sentiments: List[str] = []
    high_urgency_segments = 0
    improvement_segments = 0
    escalation_keywords_spread = 0

    for seg in segments:
        seg_lower = seg.lower()
        seg_urgency = compute_urgency(seg)
        seg_sentiment = compute_sentiment(seg)

        score = seg_urgency.score if seg_urgency else 0.0
        label = seg_sentiment.label if seg_sentiment else "neutral"

        segment_urgencies.append(score)
        segment_sentiments.append(label)

        if score >= 0.5:
            high_urgency_segments += 1
        if label in ("positive", "neutral") and score < 0.3:
            improvement_segments += 1
        if any(kw in seg_lower for kw in all_urgency_kws):
            escalation_keywords_spread += 1

    return segment_urgencies, segment_sentiments, high_urgency_segments, improvement_segments, escalation_keywords_spread


def _detect_progression(segment_urgencies: List[float], n_segments: int) -> bool:
    """Detect progression: urgency increases across segments."""
    if n_segments < 3:
        return False
    increasing_count = 0
    for i in range(1, n_segments):
        if segment_urgencies[i] > segment_urgencies[i - 1] * 1.1:
            increasing_count += 1
    return increasing_count > (n_segments - 1) // 2


def _detect_spikes(segment_urgencies: List[float], n_segments: int) -> bool:
    """Detect abrupt spike: any segment with urgency >> mean of others."""
    if n_segments < 2 or not segment_urgencies:
        return False
    mean_urgency = sum(segment_urgencies) / n_segments
    for score in segment_urgencies:
        if mean_urgency > 0.05 and score > mean_urgency * 3.0 and score >= 0.6:
            return True
    return False


def _detect_sentiment_shift(segment_sentiments: List[str], n_segments: int) -> bool:
    """Detect sentiment shift: starts neutral/positive, ends negative."""
    if n_segments < 3:
        return False
    first_label = segment_sentiments[0] if segment_sentiments else "neutral"
    last_label = segment_sentiments[-1] if segment_sentiments else "neutral"
    return first_label in ("positive", "neutral") and last_label == "negative"
