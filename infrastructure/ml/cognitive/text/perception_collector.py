"""TextPerceptionCollector — maps pre-computed text scores to EnginePerception.

Each text sub-analyzer (sentiment, urgency, readability, structural,
pattern) is treated as a cognitive "engine" that produces an
``EnginePerception`` — the same type used by ``MetaCognitiveOrchestrator``.

This enables reuse of ``InhibitionGate``, ``WeightedFusion``, and
``PlasticityTracker`` without modification.

No imports from ml_service — receives pre-computed scores via
``TextAnalysisInput``.

Single entry point: ``TextPerceptionCollector.collect()``.
"""

from __future__ import annotations

from typing import Dict, List

from ..analysis.types import EnginePerception
from .types import TextAnalysisInput

# Default base weights for text sub-analyzers
DEFAULT_TEXT_WEIGHTS: Dict[str, float] = {
    "text_sentiment": 0.20,
    "text_urgency": 0.30,
    "text_readability": 0.15,
    "text_structural": 0.15,
    "text_pattern": 0.20,
}


class TextPerceptionCollector:
    """Maps pre-computed text analysis scores to EnginePerception[].

    Stateless — safe to reuse across documents.
    """

    def collect(self, inp: TextAnalysisInput) -> List[EnginePerception]:
        """Build EnginePerception list from pre-computed analysis scores.

        Args:
            inp: Pre-computed text analysis scores.

        Returns:
            List of ``EnginePerception`` (one per sub-analyzer).
        """
        perceptions: List[EnginePerception] = []
        perceptions.append(_sentiment_perception(inp))
        perceptions.append(_urgency_perception(inp))
        perceptions.append(_readability_perception(inp))
        perceptions.append(_structural_perception(inp))
        perceptions.append(_pattern_perception(inp))
        return perceptions


def _sentiment_perception(inp: TextAnalysisInput) -> EnginePerception:
    """Map sentiment scores → EnginePerception."""
    # Normalize score from [-1, 1] to [0, 1]
    normalized = (inp.sentiment_score + 1.0) / 2.0
    total_hits = inp.sentiment_positive_count + inp.sentiment_negative_count
    hit_ratio = total_hits / max(1, total_hits + 5)

    if inp.sentiment_score > 0.1:
        trend = "up"
    elif inp.sentiment_score < -0.1:
        trend = "down"
    else:
        trend = "stable"

    return EnginePerception(
        engine_name="text_sentiment",
        predicted_value=round(normalized, 4),
        confidence=round(min(1.0, 0.5 + hit_ratio), 4),
        trend=trend,
        stability=round(1.0 - min(1.0, hit_ratio + 0.3), 4),
        local_fit_error=round(max(0.0, 0.5 - hit_ratio), 4),
        metadata={"label": inp.sentiment_label, "raw_score": inp.sentiment_score},
    )


def _urgency_perception(inp: TextAnalysisInput) -> EnginePerception:
    """Map urgency scores → EnginePerception."""
    hit_density = inp.urgency_total_hits / max(1, inp.urgency_total_hits + 10)

    if inp.urgency_severity in ("critical", "warning"):
        trend = "up"
    else:
        trend = "stable"

    return EnginePerception(
        engine_name="text_urgency",
        predicted_value=round(inp.urgency_score, 4),
        confidence=round(min(1.0, 0.5 + hit_density), 4),
        trend=trend,
        stability=round(abs(inp.urgency_score - 0.5) * 2, 4),
        local_fit_error=round(max(0.0, 0.3 - hit_density), 4),
        metadata={
            "severity": inp.urgency_severity,
            "total_hits": inp.urgency_total_hits,
        },
    )


def _readability_perception(inp: TextAnalysisInput) -> EnginePerception:
    """Map readability scores → EnginePerception."""
    # Normalize avg_sentence_length: ideal range 15-25 words
    ideal_center = 20.0
    deviation = abs(inp.readability_avg_sentence_length - ideal_center) / ideal_center
    readability_score = max(0.0, 1.0 - deviation)

    return EnginePerception(
        engine_name="text_readability",
        predicted_value=round(readability_score, 4),
        confidence=round(min(1.0, 0.6 + (inp.readability_n_sentences / 50.0)), 4),
        trend="stable",
        stability=round(min(1.0, deviation), 4),
        local_fit_error=round(deviation * 0.5, 4),
        metadata={
            "avg_sentence_length": inp.readability_avg_sentence_length,
            "n_sentences": inp.readability_n_sentences,
            "vocabulary_richness": inp.readability_vocabulary_richness,
        },
    )


def _structural_perception(inp: TextAnalysisInput) -> EnginePerception:
    """Map structural analysis scores → EnginePerception."""
    if not inp.structural_available:
        return EnginePerception(
            engine_name="text_structural",
            predicted_value=0.5,
            confidence=0.3,
            trend="stable",
            stability=0.0,
            local_fit_error=0.5,
            metadata={"available": False},
        )

    trend_map = {"increasing": "up", "decreasing": "down"}
    trend = trend_map.get(inp.structural_trend, "stable")

    return EnginePerception(
        engine_name="text_structural",
        predicted_value=round(inp.structural_stability, 4),
        confidence=round(min(1.0, 0.6 + inp.structural_stability * 0.3), 4),
        trend=trend,
        stability=round(inp.structural_noise, 4),
        local_fit_error=round(inp.structural_noise * 0.4, 4),
        metadata={
            "regime": inp.structural_regime,
            "trend": inp.structural_trend,
            "noise": inp.structural_noise,
            "available": True,
        },
    )


def _pattern_perception(inp: TextAnalysisInput) -> EnginePerception:
    """Map pattern detection scores → EnginePerception."""
    if not inp.pattern_available and inp.readability_sentences:
        from .analyzers.text_pattern import compute_text_patterns

        return compute_text_patterns(inp.readability_sentences)

    if not inp.pattern_available:
        return EnginePerception(
            engine_name="text_pattern",
            predicted_value=0.5,
            confidence=0.3,
            trend="stable",
            stability=0.0,
            local_fit_error=0.5,
            metadata={"available": False},
        )

    # Fewer patterns = more consistent = higher score
    pattern_density = min(1.0, inp.pattern_n_patterns / 10.0)
    consistency_score = 1.0 - pattern_density

    trend = "stable"
    if inp.pattern_change_points:
        trend = "up"

    return EnginePerception(
        engine_name="text_pattern",
        predicted_value=round(consistency_score, 4),
        confidence=round(min(1.0, 0.6 + (1.0 - pattern_density) * 0.3), 4),
        trend=trend,
        stability=round(pattern_density, 4),
        local_fit_error=round(pattern_density * 0.3, 4),
        metadata={
            "n_patterns": inp.pattern_n_patterns,
            "n_change_points": len(inp.pattern_change_points),
            "n_spikes": len(inp.pattern_spikes),
            "available": True,
        },
    )
