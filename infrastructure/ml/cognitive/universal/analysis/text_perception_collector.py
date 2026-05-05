"""Text perception collection logic."""
from __future__ import annotations
import logging
from typing import Any, Dict, List

from ...analysis.types import EnginePerception

logger = logging.getLogger(__name__)

_ATTENTION_AVAILABLE = False
try:
    from ...neural.attention import AttentionContextCollector
    from iot_machine_learning.infrastructure.ml.cognitive.text.analyzers.keyword_config import ATTENTION_CONFIG
    _ATTENTION_AVAILABLE = True
except Exception:
    pass


def collect_text_perceptions(scores: Dict[str, Any]) -> List[EnginePerception]:
    """Build text perceptions from pre-computed scores."""
    print(f"[DEBUG] _collect_text received scores: {list(scores.keys())}")
    print(f"[DEBUG] urgency_score: {scores.get('urgency_score', 'N/A')}, urgency_severity: {scores.get('urgency_severity', 'N/A')}")

    perceptions: List[EnginePerception] = []

    sentiment_score = scores.get("sentiment_score", 0.0)
    sentiment_label = scores.get("sentiment_label", "neutral")
    normalized = max(0.0, min(1.0, (sentiment_score + 1.0) / 2.0))

    perceptions.append(EnginePerception(
        engine_name="text_sentiment",
        predicted_value=round(normalized, 4),
        confidence=0.7,
        trend="up" if sentiment_score > 0.1 else "down" if sentiment_score < -0.1 else "stable",
        stability=0.3,
        local_fit_error=0.2,
        metadata={"label": sentiment_label, "raw_score": sentiment_score},
    ))

    urgency_score = max(0.0, min(1.0, scores.get("urgency_score", 0.0)))
    urgency_severity = scores.get("urgency_severity", "info")

    perceptions.append(EnginePerception(
        engine_name="text_urgency",
        predicted_value=round(urgency_score, 4),
        confidence=0.7,
        trend="up" if urgency_severity in ("critical", "warning") else "stable",
        stability=0.3,
        local_fit_error=0.2,
        metadata={"severity": urgency_severity},
    ))

    readability_avg = scores.get("readability_avg_sentence_length", 20.0)
    ideal = 20.0
    deviation = abs(readability_avg - ideal) / ideal
    readability_score = max(0.0, 1.0 - deviation)

    perceptions.append(EnginePerception(
        engine_name="text_readability",
        predicted_value=round(readability_score, 4),
        confidence=0.6,
        trend="stable",
        stability=round(min(1.0, deviation), 4),
        local_fit_error=round(deviation * 0.5, 4),
        metadata={"avg_sentence_length": readability_avg},
    ))

    if _ATTENTION_AVAILABLE and scores.get("enable_attention", False):
        try:
            raw_text = scores.get("raw_text", "")
            if raw_text and len(raw_text) > 50:
                vocab = {kw: i for i, kw in enumerate(
                    [w for kws in ATTENTION_CONFIG.get("TEMPORAL_KEYWORDS", [])][:ATTENTION_CONFIG.get("D_MODEL", 64)]
                )}
                if vocab:
                    collector = AttentionContextCollector(
                        vocab=vocab,
                        n_heads=ATTENTION_CONFIG.get("N_HEADS", 4),
                        d_model=ATTENTION_CONFIG.get("D_MODEL", 64),
                    )
                    ctx = collector.collect_context(raw_text, budget_ms=ATTENTION_CONFIG.get("BUDGET_MS", 100.0))
                    if ctx and ctx.confidence >= ATTENTION_CONFIG.get("CONFIDENCE_THRESHOLD", 0.5):
                        perceptions.append(EnginePerception(
                            engine_name="text_attention",
                            predicted_value=round(ctx.confidence, 4),
                            confidence=round(ctx.confidence, 4),
                            trend="stable",
                            stability=0.5,
                            local_fit_error=0.2,
                            metadata={
                                "attended_sentences": ctx.attended_sentences[:3],
                                "temporal_markers": ctx.temporal_markers,
                                "negation_context": ctx.negation_context,
                                "multi_domain_scores": ctx.multi_domain_scores,
                            },
                        ))
        except Exception as e:
            logger.debug(f"attention_context_failed: {e}")

    semantic_enrichment = scores.get("semantic_enrichment")
    if semantic_enrichment and isinstance(semantic_enrichment, dict):
        try:
            entity_count = semantic_enrichment.get("entity_count", 0)
            critical_count = len(semantic_enrichment.get("critical_entities", []))
            equipment_metrics = semantic_enrichment.get("equipment_metric_pairs", [])

            richness = min(1.0, (
                (entity_count / 10) * 0.3 +
                (critical_count / 5) * 0.4 +
                (len(equipment_metrics) / 3) * 0.3
            ))

            perceptions.append(EnginePerception(
                engine_name="semantic_entities",
                predicted_value=round(richness, 4),
                confidence=semantic_enrichment.get("enrichment_confidence", 0.7),
                trend="up" if critical_count > 0 else "stable",
                stability=0.4 if equipment_metrics else 0.6,
                local_fit_error=0.2,
                metadata={
                    "entity_count": entity_count,
                    "critical_count": critical_count,
                    "equipment_metric_pairs": equipment_metrics,
                    "entity_types": list(set(
                        e.get("entity_type") for e in
                        semantic_enrichment.get("entities", [])
                    )),
                    "domain_detected": semantic_enrichment.get("domain_detected", "general"),
                },
            ))

            if critical_count > 0:
                critical_confidence = min(0.95, 0.6 + (critical_count * 0.1))
                perceptions.append(EnginePerception(
                    engine_name="semantic_critical_alert",
                    predicted_value=min(1.0, critical_count * 0.25),
                    confidence=critical_confidence,
                    trend="up",
                    stability=0.2,
                    local_fit_error=0.3,
                    metadata={
                        "critical_entities": semantic_enrichment.get("critical_entities", [])[:3],
                        "alert_reason": "critical_semantic_entities_detected",
                        "n_critical": critical_count,
                    },
                ))

        except Exception as e:
            logger.debug(f"semantic_entity_perception_failed: {e}")

    return perceptions
