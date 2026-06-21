"""
CausalCorrelationEngine for detecting operational correlations and causal relationships.

Implements lagged correlations, temporal correlation analysis, basic Granger causality,
rolling correlation analysis, and propagation likelihood estimation.
"""

from typing import Dict, List, Optional, Set, Tuple
import time
from collections import defaultdict

from domain.entities.causal import CausalCorrelation
from .utils.correlation_calculator import CorrelationCalculator
from .utils.granger_causality import GrangerCausalityDetector

_LAG_STEP_SECONDS = 30
_MAX_LAG_CACHE_SIZE = 10000


class CausalCorrelationEngine:
    """Engine for detecting operational causal correlations."""

    def __init__(
        self,
        min_correlation_threshold: float = 0.5,
        min_lag_seconds: float = 1.0,
        max_lag_seconds: float = 3600.0,
        min_confidence: float = 0.6,
        min_samples: int = 10,
        correlation_threshold: float = 0.3,
    ):
        self._min_correlation_threshold = min_correlation_threshold
        self._min_lag_seconds = min_lag_seconds
        self._max_lag_seconds = max_lag_seconds
        self._min_confidence = min_confidence
        self._min_samples = min_samples
        self._correlation_threshold = correlation_threshold

        self._sensor_data: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self._dirty_sensors: Set[int] = set()
        self._correlation_cache: Dict[tuple, CausalCorrelation] = {}
    
    def add_sensor_reading(
        self,
        sensor_id: int,
        value: float,
        timestamp: float,
    ) -> None:
        """Add sensor reading for correlation analysis."""
        self._sensor_data[sensor_id].append((timestamp, value))
        self._dirty_sensors.add(sensor_id)
        
        if len(self._sensor_data[sensor_id]) > 1000:
            self._sensor_data[sensor_id] = self._sensor_data[sensor_id][-1000:]
    
    def detect_correlations(
        self,
        sensor_id: int,
        target_sensor_ids: Optional[List[int]] = None,
    ) -> List[CausalCorrelation]:
        """Detect correlations between sensors."""
        if target_sensor_ids is None:
            target_sensor_ids = list(self._sensor_data.keys())
        
        if sensor_id not in self._sensor_data or len(self._sensor_data[sensor_id]) < self._min_samples:
            return []
        
        source_data = self._sensor_data[sensor_id]
        correlations = []
        
        for target_id in target_sensor_ids:
            if target_id == sensor_id:
                continue
            
            target_data = self._sensor_data.get(target_id, [])
            if len(target_data) < self._min_samples:
                continue
            
            cache_key = (sensor_id, target_id)
            if sensor_id not in self._dirty_sensors and cache_key in self._correlation_cache:
                correlation = self._correlation_cache[cache_key]
                if correlation is not None:
                    correlations.append(correlation)
                continue
            
            correlation = self._compute_lagged_correlation(sensor_id, target_id, source_data, target_data)

            if correlation is not None:
                if abs(correlation.correlation_coefficient) >= self._correlation_threshold:
                    self._correlation_cache[cache_key] = correlation
                    correlations.append(correlation)
            else:
                self._correlation_cache.pop(cache_key, None)
        
        self._dirty_sensors.discard(sensor_id)
        
        self._cleanup_cache()
        
        return correlations
    
    def _compute_lagged_correlation(
        self,
        source_id: int,
        target_id: int,
        source_data: List[Tuple[float, float]],
        target_data: List[Tuple[float, float]],
    ) -> Optional[CausalCorrelation]:
        """Compute lagged correlation between sensors."""
        best_correlation = 0.0
        best_lag = 0.0
        
        max_lag = min(int(self._max_lag_seconds), len(source_data) * 2)
        lag_step = max(_LAG_STEP_SECONDS, max_lag // 50)

        for lag_seconds in range(1, max_lag + 1, lag_step):
            correlation = CorrelationCalculator.compute_correlation_at_lag(
                source_data, target_data, lag_seconds
            )

            if correlation > best_correlation:
                best_correlation = correlation
                best_lag = lag_seconds

        if best_correlation < self._correlation_threshold:
            return None
        
        return CausalCorrelation(
            source_sensor_id=source_id,
            target_sensor_id=target_id,
            correlation_coefficient=best_correlation,
            lag_seconds=best_lag,
            confidence=self._calculate_confidence(best_correlation, len(source_data)),
            propagation_likelihood=self._calculate_propagation_likelihood(best_correlation, best_lag),
            timestamp=time.time(),
            metadata={
                "data_points": len(source_data),
                "lag_range_tested": max_lag,
            },
        )
    
    def _calculate_confidence(self, correlation: float, data_points: int) -> float:
        """Calculate confidence based on correlation and data points."""
        correlation_factor = min(1.0, abs(correlation))
        data_factor = min(1.0, data_points / 100.0)
        
        return (correlation_factor * 0.7 + data_factor * 0.3)
    
    def _calculate_propagation_likelihood(self, correlation: float, lag: float) -> float:
        """Calculate propagation likelihood."""
        correlation_factor = min(1.0, abs(correlation))
        lag_factor = max(0.0, 1.0 - (lag / self._max_lag_seconds))
        
        return (correlation_factor * 0.8 + lag_factor * 0.2)
    
    def _cleanup_cache(self) -> None:
        if len(self._correlation_cache) > _MAX_LAG_CACHE_SIZE:
            cutoff = time.time() - 3600
            stale_keys = [
                k for k, v in self._correlation_cache.items()
                if v.timestamp < cutoff
            ]
            for k in stale_keys:
                self._correlation_cache.pop(k, None)

    def detect_granger_causality(
        self,
        source_id: int,
        target_id: int,
        max_lag: int = 10,
    ) -> Optional[float]:
        """Detect basic Granger causality (simplified)."""
        source_data = self._sensor_data.get(source_id, [])
        target_data = self._sensor_data.get(target_id, [])
        
        if len(source_data) < max_lag + 10 or len(target_data) < max_lag + 10:
            return None
        
        return GrangerCausalityDetector.detect_granger_causality(
            source_data, target_data, max_lag
        )
    
    def reset(self) -> None:
        """Reset all sensor data."""
        self._sensor_data.clear()
        self._dirty_sensors.clear()
        self._correlation_cache.clear()
