"""
DriftDetectionEngine for detecting operational and cognitive drift.

Implements statistical drift detection, regime drift, feature distribution drift, anomaly frequency drift, and basic embedding drift.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import statistics

from domain.entities.observability import DriftResult


class DriftDetectionEngine:
    """Engine for drift detection."""
    
    def __init__(
        self,
        statistical_drift_threshold: float = 0.3,
        regime_drift_threshold: float = 0.4,
        feature_drift_threshold: float = 0.35,
        anomaly_drift_threshold: float = 0.3,
        embedding_drift_threshold: float = 0.25,
    ):
        """
        Initialize drift detection engine.
        
        Args:
            statistical_drift_threshold: Threshold for statistical drift
            regime_drift_threshold: Threshold for regime drift
            feature_drift_threshold: Threshold for feature drift
            anomaly_drift_threshold: Threshold for anomaly frequency drift
            embedding_drift_threshold: Threshold for embedding drift
        """
        self._statistical_drift_threshold = statistical_drift_threshold
        self._regime_drift_threshold = regime_drift_threshold
        self._feature_drift_threshold = feature_drift_threshold
        self._anomaly_drift_threshold = anomaly_drift_threshold
        self._embedding_drift_threshold = embedding_drift_threshold
        
        # Baseline distributions
        self._baseline_regime_distribution: Dict[str, float] = {}
        self._baseline_feature_means: Dict[str, float] = {}
        self._baseline_anomaly_frequency: float = 0.0
        self._baseline_embedding_mean: float = 0.0
    
    def set_baselines(
        self,
        regime_distribution: Dict[str, float],
        feature_means: Dict[str, float],
        anomaly_frequency: float,
        embedding_mean: float,
    ) -> None:
        """Set baseline distributions for drift detection."""
        self._baseline_regime_distribution = regime_distribution
        self._baseline_feature_means = feature_means
        self._baseline_anomaly_frequency = anomaly_frequency
        self._baseline_embedding_mean = embedding_mean
    
    def detect_drift(
        self,
        current_regime_distribution: Dict[str, float],
        current_feature_means: Dict[str, float],
        current_anomaly_frequency: float,
        current_embedding_mean: float,
        sensor_id: Optional[int] = None,
        regime: Optional[str] = None,
        temporal_window: Optional[Tuple[float, float]] = None,
    ) -> DriftResult:
        """Detect drift across multiple dimensions."""
        # Calculate individual drift scores
        statistical_drift_score = self._detect_statistical_drift(current_feature_means)
        regime_drift_score = self._detect_regime_drift(current_regime_distribution)
        feature_drift_score = self._detect_feature_drift(current_feature_means)
        anomaly_frequency_drift_score = self._detect_anomaly_frequency_drift(current_anomaly_frequency)
        embedding_drift_score = self._detect_embedding_drift(current_embedding_mean)
        
        # Determine if drift is detected
        drift_detected = any([
            statistical_drift_score > self._statistical_drift_threshold,
            regime_drift_score > self._regime_drift_threshold,
            feature_drift_score > self._feature_drift_threshold,
            anomaly_frequency_drift_score > self._anomaly_drift_threshold,
            embedding_drift_score > self._embedding_drift_threshold,
        ])
        
        # Determine drift type and magnitude
        drift_type, drift_magnitude = self._determine_drift_type_and_magnitude(
            statistical_drift_score,
            regime_drift_score,
            feature_drift_score,
            anomaly_frequency_drift_score,
            embedding_drift_score,
        )
        
        return DriftResult(
            timestamp=time.time(),
            drift_detected=drift_detected,
            drift_type=drift_type,
            drift_magnitude=drift_magnitude,
            drift_sensor_id=sensor_id,
            drift_regime=regime,
            drift_temporal_window=temporal_window,
            statistical_drift_score=statistical_drift_score,
            regime_drift_score=regime_drift_score,
            feature_drift_score=feature_drift_score,
            anomaly_frequency_drift_score=anomaly_frequency_drift_score,
            embedding_drift_score=embedding_drift_score,
        )
    
    def _detect_statistical_drift(self, current_feature_means: Dict[str, float]) -> float:
        """Detect statistical drift in feature means."""
        if not self._baseline_feature_means:
            return 0.0
        
        drift_scores = []
        for feature, current_mean in current_feature_means.items():
            baseline_mean = self._baseline_feature_means.get(feature, 0.0)
            if baseline_mean != 0:
                drift_score = abs(current_mean - baseline_mean) / abs(baseline_mean)
                drift_scores.append(drift_score)
        
        return statistics.mean(drift_scores) if drift_scores else 0.0
    
    def _detect_regime_drift(self, current_regime_distribution: Dict[str, float]) -> float:
        """Detect drift in regime distribution."""
        if not self._baseline_regime_distribution:
            return 0.0
        
        drift_scores = []
        all_regimes = set(self._baseline_regime_distribution.keys()) | set(current_regime_distribution.keys())
        
        for regime in all_regimes:
            baseline = self._baseline_regime_distribution.get(regime, 0.0)
            current = current_regime_distribution.get(regime, 0.0)
            
            if baseline != 0:
                drift_score = abs(current - baseline) / baseline
                drift_scores.append(drift_score)
        
        return statistics.mean(drift_scores) if drift_scores else 0.0
    
    def _detect_feature_drift(self, current_feature_means: Dict[str, float]) -> float:
        """Detect drift in feature distributions."""
        # Similar to statistical drift but with different interpretation
        return self._detect_statistical_drift(current_feature_means)
    
    def _detect_anomaly_frequency_drift(self, current_anomaly_frequency: float) -> float:
        """Detect drift in anomaly frequency."""
        if self._baseline_anomaly_frequency == 0:
            return 0.0
        
        return abs(current_anomaly_frequency - self._baseline_anomaly_frequency) / self._baseline_anomaly_frequency
    
    def _detect_embedding_drift(self, current_embedding_mean: float) -> float:
        """Detect drift in embedding distributions."""
        if self._baseline_embedding_mean == 0:
            return 0.0
        
        return abs(current_embedding_mean - self._baseline_embedding_mean) / self._baseline_embedding_mean
    
    def _determine_drift_type_and_magnitude(
        self,
        statistical_drift: float,
        regime_drift: float,
        feature_drift: float,
        anomaly_drift: float,
        embedding_drift: float,
    ) -> Tuple[str, float]:
        """Determine drift type and magnitude."""
        drift_scores = {
            "statistical": statistical_drift,
            "regime": regime_drift,
            "feature": feature_drift,
            "anomaly_frequency": anomaly_drift,
            "embedding": embedding_drift,
        }
        
        # Find the highest drift score
        max_drift_type = max(drift_scores, key=drift_scores.get)
        max_drift_magnitude = drift_scores[max_drift_type]
        
        return max_drift_type, max_drift_magnitude
