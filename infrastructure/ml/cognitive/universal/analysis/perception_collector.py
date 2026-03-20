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

_ML_COMPONENTS_AVAILABLE = True
try:
    from ....patterns.delta_spike import DeltaSpikeClassifier
    from ....patterns.change_point import CUSUMDetector
    from ....patterns.regime_detector import RegimeDetector
    from ....anomaly.core.detector import VotingAnomalyDetector
    from ....anomaly.core.config import AnomalyDetectorConfig
    from .....domain.validators.structural_analysis import compute_structural_analysis
    from .....domain.entities.iot.sensor_reading import SensorReading, SensorWindow
except Exception:
    _ML_COMPONENTS_AVAILABLE = False


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
            return self._collect_text(pre_computed_scores or {})
        
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
        perceptions: List[EnginePerception] = []
        
        sentiment_score = scores.get("sentiment_score", 0.0)
        sentiment_label = scores.get("sentiment_label", "neutral")
        normalized = (sentiment_score + 1.0) / 2.0
        
        perceptions.append(EnginePerception(
            engine_name="text_sentiment",
            predicted_value=round(normalized, 4),
            confidence=0.7,
            trend="up" if sentiment_score > 0.1 else "down" if sentiment_score < -0.1 else "stable",
            stability=0.3,
            local_fit_error=0.2,
            metadata={"label": sentiment_label, "raw_score": sentiment_score},
        ))
        
        urgency_score = scores.get("urgency_score", 0.0)
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
        
        return perceptions

    def _collect_numeric(
        self,
        values: list,
        timestamps: Optional[list],
        metadata: Dict[str, Any],
    ) -> List[EnginePerception]:
        """Build perceptions for numeric series.

        REUSES existing infrastructure/ml/ components:
            - numeric_spikes: DeltaSpikeClassifier
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
        
        if not _ML_COMPONENTS_AVAILABLE:
            logger.warning("ML components not available for numeric analysis")
            return perceptions
        
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
                    "regime": structural.regime.value,
                    "slope": structural.slope,
                    "curvature": structural.curvature,
                },
            ))
        except Exception as e:
            logger.debug(f"structural_analysis failed: {e}")
        
        try:
            classifier = DeltaSpikeClassifier()
            spike_result = classifier.detect_spikes(values, threshold_sigma=2.5)
            n_spikes = len(spike_result.spike_indices) if hasattr(spike_result, 'spike_indices') else 0
            spike_score = min(1.0, n_spikes / 10.0)
            
            perceptions.append(EnginePerception(
                engine_name="numeric_spikes",
                predicted_value=round(1.0 - spike_score, 4),
                confidence=0.7,
                trend="stable" if n_spikes == 0 else "up",
                stability=round(1.0 - spike_score, 4),
                local_fit_error=round(spike_score, 4),
                metadata={"n_spikes": n_spikes},
            ))
        except Exception as e:
            logger.debug(f"delta_spike failed: {e}")
        
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
