"""Build pre-computed scores for text content in the universal bridge."""

from __future__ import annotations

import logging
from typing import Any, Dict

from .pattern_signal_builder import build_real_pattern_signals

logger = logging.getLogger(__name__)


def build_text_pre_computed_scores(raw_data: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build pre_computed_scores for text content.

    Computes sentiment, urgency, readability, entities, and real pattern signals.
    Returns empty dict on failure (never None).
    """
    try:
        from iot_machine_learning.infrastructure.ml.cognitive.text.analyzers import (
            compute_sentiment, compute_urgency, compute_readability, compute_text_structure,
        )
        from iot_machine_learning.infrastructure.ml.cognitive.text.text_pattern import detect_text_patterns

        full_text = raw_data if isinstance(raw_data, str) else str(raw_data)
        word_count = len(full_text.split()) if full_text else 0
        paragraph_count = len(full_text.split("\n\n")) if full_text else 0

        logger.info(f"[UNIVERSAL_BRIDGE] Building scores directly for text length: {len(full_text)}")

        sentiment = compute_sentiment(full_text)
        urgency = compute_urgency(full_text)
        readability = compute_readability(full_text, word_count)
        structural = compute_text_structure(readability.sentences if readability else [])
        patterns = detect_text_patterns(readability.sentences if readability else [])

        entities = _extract_entities(full_text, payload)

        urgency_score = urgency.score if urgency else 0.0
        urgency_severity = urgency.severity if urgency else "info"

        real_pattern_signals = build_real_pattern_signals(
            full_text=full_text,
            urgency=urgency,
            sentiment=sentiment,
            structural=structural,
            patterns=patterns,
        )

        pre_computed_scores: Dict[str, Any] = {
            "sentiment_score": sentiment.score if sentiment else 0.0,
            "sentiment_label": sentiment.label if sentiment else "neutral",
            "urgency_score": urgency_score,
            "urgency_severity": urgency_severity,
            "word_count": word_count,
            "paragraph_count": paragraph_count,
            "entities": entities,
            "patterns": {
                "pattern_summary": real_pattern_signals,
                "change_points": [
                    {"index": idx, "description": f"Narrative shift at sentence {idx}"}
                    for idx in (patterns.change_points if patterns and hasattr(patterns, "change_points") else [])
                ],
                "spikes": [
                    {"index": idx, "magnitude": 2.5, "description": f"Outlier at position {idx}"}
                    for idx in (patterns.spikes if patterns and hasattr(patterns, "spikes") else [])
                ],
            },
        }

        logger.info(
            f"[UNIVERSAL_BRIDGE] Direct scores built: urgency={pre_computed_scores['urgency_score']:.2f}, "
            f"sentiment={pre_computed_scores['sentiment_label']}, patterns={real_pattern_signals}"
        )
        return pre_computed_scores

    except Exception as e:
        logger.warning(f"[UNIVERSAL_BRIDGE] Failed to build direct scores: {e}")
        import traceback
        logger.debug(f"[UNIVERSAL_BRIDGE] Traceback: {traceback.format_exc()}")
        return {}


def _extract_entities(full_text: str, payload: Dict[str, Any]) -> list:
    """Extract entities using hybrid embeddings or regex fallback."""
    try:
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        from iot_machine_learning.infrastructure.ml.cognitive.text.embeddings import HybridEntityDetector

        flags = FeatureFlags()
        if getattr(flags, "ML_ENABLE_HYBRID_EMBEDDINGS", False):
            detector = HybridEntityDetector(
                domain_hint=payload.get("domain", "general"),
                magnitude_threshold=getattr(flags, "ML_HYBRID_ENTITY_THRESHOLD", 0.3),
            )
            entity_result = detector.extract_entities(full_text)
            entities = entity_result.to_list()
            logger.info(f"[UNIVERSAL_BRIDGE] Hybrid embeddings extracted {len(entities)} entities")
            return entities
    except Exception as e:
        logger.debug(f"hybrid_embedding_failed: {e}, falling back to regex")

    return _extract_entities_regex(full_text)


def _extract_entities_regex(text: str) -> list:
    """Fallback regex entity extraction."""
    import re
    result = []
    result.extend(re.findall(r"\b\d+\s*°[CF]\b", text))
    result.extend(re.findall(r"\b(NODE|TMP|SERVER|ROUTER|SWITCH)-\w+\b", text))
    result.extend(re.findall(r"\b(COMP|VLV|MOT|PUMP|CMP|BLR|GEN|TX|HV)[-]?[A-Z0-9]+\b", text, re.IGNORECASE))
    result.extend(re.findall(r"\$[\d,]+(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})+\s*(?:USD|EUR|USD\$|\$)\b", text))
    result.extend(re.findall(r"\b\d+%\b", text))
    result.extend(re.findall(r"\bSLA\s+\d+\.?\d*%?\b", text))
    return result
