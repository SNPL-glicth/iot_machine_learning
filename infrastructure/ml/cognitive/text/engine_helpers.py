"""Helper functions for TextCognitiveEngine.

Extracted utility functions to keep the main engine under 180 lines.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .types import TextAnalysisInput


def compute_confidence(inp: TextAnalysisInput, has_recall: bool) -> float:
    """Compute overall confidence from text metrics."""
    confidence = 0.75
    if inp.word_count > 100:
        confidence = 0.80
    if inp.word_count > 500:
        confidence = 0.85
    if inp.structural_available:
        confidence += 0.05
    if has_recall:
        confidence = min(0.95, confidence + 0.05)
    if inp.pattern_available:
        confidence = min(0.95, confidence + 0.02)
    return confidence


def build_analysis_dict(
    inp: TextAnalysisInput,
    signal: Any,
    perceptions: List[Any],
    final_weights: Dict[str, float],
    impact_result: Any = None,
) -> Dict[str, Any]:
    """Build backward-compatible analysis dict."""
    d: Dict[str, Any] = {
        "sentiment": inp.sentiment_label,
        "sentiment_score": inp.sentiment_score,
        "urgency_score": inp.urgency_score,
        "urgency_hits": inp.urgency_hits,
        "readability": {
            "avg_sentence_length": inp.readability_avg_sentence_length,
            "n_sentences": inp.readability_n_sentences,
            "vocabulary_richness": inp.readability_vocabulary_richness,
            "embedded_numeric_values": inp.readability_embedded_numeric_count,
        },
        "structural": {
            "sentence_length_regime": inp.structural_regime,
            "sentence_length_trend": inp.structural_trend,
            "sentence_length_stability": inp.structural_stability,
            "sentence_length_noise": inp.structural_noise,
        } if inp.structural_available else {},
        "patterns": {
            "n_patterns": inp.pattern_n_patterns,
            "change_points": inp.pattern_change_points,
            "spikes": inp.pattern_spikes,
            "summary": inp.pattern_summary,
        } if inp.pattern_available else {},
        "cognitive": {
            "engine_weights": {
                k: round(v, 4) for k, v in final_weights.items()
            },
            "engine_perceptions": [
                p.to_dict() for p in perceptions
            ],
            "signal_profile": signal.to_dict(),
        },
    }
    if impact_result is not None:
        d["impact"] = impact_result.to_dict()
    return d


def build_basic_conclusion(
    domain: str,
    severity: Any,
    inp: TextAnalysisInput,
    impact_result: Any = None,
) -> str:
    """Build a basic conclusion from severity and domain.

    The caller (ml_service text_analyzer) may replace this with the
    richer output of ``build_semantic_conclusion()``.
    """
    parts: List[str] = []

    # Domain line
    domain_label = domain.capitalize() if domain != "general" else "General"
    parts.append(f"Domain: {domain_label}")

    # Severity line
    parts.append(f"Severity: {severity.severity}")

    # Impact signals
    if impact_result is not None and impact_result.summary:
        parts.append(impact_result.summary)

    # Key signals
    if inp.urgency_severity in ("critical", "warning"):
        parts.append(f"Urgency: {inp.urgency_severity} (score: {inp.urgency_score:.2f})")
    if inp.sentiment_label != "neutral":
        parts.append(f"Sentiment: {inp.sentiment_label} (score: {inp.sentiment_score:.2f})")

    # Action
    if severity.action_required:
        parts.append(f"Recommended actions: {severity.recommended_action}")
    else:
        parts.append("No immediate action required.")

    return "\n".join(parts)
