"""Universal perception collector — dispatcher to type-specific analyzers.

CRITICAL: Reuses existing infrastructure/ml/ components for numeric analysis.
Does NOT reinvent analyzers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...analysis.types import EnginePerception
from .types import InputType

logger = logging.getLogger(__name__)

from ....patterns.change_point_detector import CUSUMDetector
from ....patterns.regime_detector import RegimeDetector
from ....anomaly.core.detector import VotingAnomalyDetector
from ....anomaly.core.config import AnomalyDetectorConfig
from iot_machine_learning.domain.validators.structural_analysis import compute_structural_analysis
from iot_machine_learning.domain.entities.iot.sensor_reading import SensorReading, SensorWindow

# --- Attention Integration (optional) ---
_ATTENTION_AVAILABLE = False
try:
    from ...neural.attention import AttentionContextCollector
    from ....text.analyzers.keyword_config import ATTENTION_CONFIG
    _ATTENTION_AVAILABLE = True
except Exception:
    pass


class UniversalPerceptionCollector:
    """Maps any input type to List[EnginePerception].

    Dispatches to appropriate sub-analyzers based on InputType.
    
    TEXT: Reuses text_sentiment, text_urgency, text_readability, text_structural, text_pattern
    NUMERIC: Wraps existing infrastructure/ml/ components as EnginePerceptions
    TABULAR: Analyzes first numeric column
    MIXED: Hybrid approach
    """

    def collect(
        self,
        raw_data: Any,
        input_type: InputType,
        metadata: Dict[str, Any],
        pre_computed_scores: Optional[Dict[str, Any]] = None,
    ) -> List[EnginePerception]:
        """Build EnginePerception list from input.

        Args:
            raw_data: Original input
            input_type: Detected InputType
            metadata: From input_detector
            pre_computed_scores: Scores from ml_service analyzers (if available)

        Returns:
            List of EnginePerception (one per sub-analyzer)
        """
        if input_type == InputType.TEXT:
            # Merge pre_computed_scores with metadata (includes semantic_enrichment)
            merged_scores = {**(pre_computed_scores or {})}
            # Add semantic_enrichment from metadata if present
            if metadata and "semantic_enrichment" in metadata:
                merged_scores["semantic_enrichment"] = metadata["semantic_enrichment"]
            return self._collect_text(merged_scores)
        
        if input_type == InputType.NUMERIC:
            return self._collect_numeric(raw_data, None, metadata)
        
        if input_type == InputType.TABULAR:
            return self._collect_tabular(raw_data, metadata)
        
        if input_type == InputType.MIXED:
            return self._collect_mixed(raw_data, metadata)
        
        return []

    def _collect_text(
        self,
        scores: Dict[str, Any],
    ) -> List[EnginePerception]:
        """Delegate to text perception logic (from text/perception_collector.py)."""
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
        
        # --- Attention Context Enhancement (optional) ---
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
                            # Add attention perception with context
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
        
        # --- Semantic Entity Extraction ---
        # Generate EnginePerception from structured entities if available
        semantic_enrichment = scores.get("semantic_enrichment")
        if semantic_enrichment and isinstance(semantic_enrichment, dict):
            try:
                entity_count = semantic_enrichment.get("entity_count", 0)
                critical_count = len(semantic_enrichment.get("critical_entities", []))
                equipment_metrics = semantic_enrichment.get("equipment_metric_pairs", [])
                
                # Compute semantic richness score (0-1)
                # Based on: entity density, critical entities, equipment-metric pairs
                richness = min(1.0, (
                    (entity_count / 10) * 0.3 +  # Max 0.3 for 10+ entities
                    (critical_count / 5) * 0.4 +  # Max 0.4 for 5+ critical
                    (len(equipment_metrics) / 3) * 0.3  # Max 0.3 for 3+ pairs
                ))
                
                # Create semantic entity perception
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
                
                # Add critical entity alert perception if critical entities exist
                if critical_count > 0:
                    critical_confidence = min(0.95, 0.6 + (critical_count * 0.1))
                    perceptions.append(EnginePerception(
                        engine_name="semantic_critical_alert",
                        predicted_value=min(1.0, critical_count * 0.25),  # Scale with count
                        confidence=critical_confidence,
                        trend="up",
                        stability=0.2,  # Unstable = alert condition
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

    def _collect_numeric(
        self,
        values: list,
        timestamps: Optional[list],
        metadata: Dict[str, Any],
    ) -> List[EnginePerception]:
        """Build perceptions for numeric series.

        REUSES existing infrastructure/ml/ components:
            - numeric_drift: CUSUMDetector
            - numeric_regime: RegimeDetector
            - numeric_anomaly: VotingAnomalyDetector
            - numeric_structural: compute_structural_analysis()
        """
        perceptions: List[EnginePerception] = []
        n = len(values)
        
        if n < 3:
            return perceptions
        
        if timestamps is None:
            timestamps = [float(i) for i in range(n)]
        
        try:
            structural = compute_structural_analysis(values, timestamps)
            perceptions.append(EnginePerception(
                engine_name="numeric_structural",
                predicted_value=round(structural.stability, 4),
                confidence=0.8,
                trend="up" if structural.slope > 0.1 else "down" if structural.slope < -0.1 else "stable",
                stability=round(structural.stability, 4),
                local_fit_error=round(structural.noise_ratio, 4),
                metadata={
                    "regime": structural.regime.value if hasattr(structural.regime, 'value') else str(structural.regime),
                    "slope": structural.slope,
                    "curvature": structural.curvature,
                },
            ))
        except Exception as e:
            logger.debug(f"structural_analysis failed: {e}")
        
        # numeric_spikes: neutral placeholder (DeltaSpikeClassifier removed)
        perceptions.append(EnginePerception(
            engine_name="numeric_spikes",
            predicted_value=0.0,
            confidence=0.0,
            trend="stable",
            stability=1.0,
            local_fit_error=0.0,
            metadata={"reason": "delta_spike_classifier_removed"},
        ))
        
        if n >= 20:
            try:
                detector = CUSUMDetector()
                changes = detector.detect(values)
                n_changes = len(changes) if changes else 0
                drift_score = min(1.0, n_changes / 5.0)
                
                perceptions.append(EnginePerception(
                    engine_name="numeric_drift",
                    predicted_value=round(1.0 - drift_score, 4),
                    confidence=0.75,
                    trend="up" if n_changes > 0 else "stable",
                    stability=round(1.0 - drift_score, 4),
                    local_fit_error=round(drift_score, 4),
                    metadata={"n_change_points": n_changes},
                ))
            except Exception as e:
                logger.debug(f"cusum failed: {e}")
            
            try:
                regime_det = RegimeDetector()
                regime_result = regime_det.detect(values)
                n_regimes = len(set(regime_result.labels)) if hasattr(regime_result, 'labels') else 1
                regime_consistency = 1.0 / n_regimes
                
                perceptions.append(EnginePerception(
                    engine_name="numeric_regime",
                    predicted_value=round(regime_consistency, 4),
                    confidence=0.7,
                    trend="stable",
                    stability=round(regime_consistency, 4),
                    local_fit_error=round(1.0 - regime_consistency, 4),
                    metadata={"n_regimes": n_regimes},
                ))
            except Exception as e:
                logger.debug(f"regime_detector failed: {e}")
            
            try:
                config = AnomalyDetectorConfig(min_training_points=20)
                anom_detector = VotingAnomalyDetector(config=config)
                anom_detector.train(values, timestamps=timestamps)
                
                readings = [
                    SensorReading(sensor_id=0, value=v, timestamp=t)
                    for v, t in zip(values, timestamps)
                ]
                window = SensorWindow(sensor_id=0, readings=readings)
                result = anom_detector.detect(window)
                
                perceptions.append(EnginePerception(
                    engine_name="numeric_anomaly",
                    predicted_value=round(1.0 - result.score, 4),
                    confidence=round(result.confidence, 4),
                    trend="up" if result.is_anomaly else "stable",
                    stability=round(1.0 - result.score, 4),
                    local_fit_error=round(result.score, 4),
                    metadata={
                        "is_anomaly": result.is_anomaly,
                        "severity": result.severity.value,
                    },
                ))
            except Exception as e:
                logger.debug(f"voting_anomaly failed: {e}")
        
        return perceptions

    def _collect_tabular(
        self,
        data: dict,
        metadata: Dict[str, Any],
    ) -> List[EnginePerception]:
        """Build perceptions for tabular data."""
        numeric_columns = metadata.get("numeric_columns", [])
        
        if not numeric_columns:
            return []
        
        first_col = numeric_columns[0]
        values = data.get(first_col, [])
        
        return self._collect_numeric(values, None, metadata)

    def _collect_mixed(
        self,
        raw_data: Any,
        metadata: Dict[str, Any],
    ) -> List[EnginePerception]:
        """Hybrid collection for mixed-type data."""
        return []
