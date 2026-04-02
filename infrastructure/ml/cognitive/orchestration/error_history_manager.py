"""Error History Manager — Data Leakage Fix (CRIT-1).

Encapsulates error tracking with series_id as primary namespace.
Prevents cross-sensor contamination in InhibitionGate decisions.

Thread Safety:
    Uses RLock to protect shared dictionary from race conditions.
    All public methods are thread-safe.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from threading import RLock
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorHistoryManager:
    """Manages prediction error history per series_id and engine.
    
    Structure: {series_id: {engine_name: deque[errors]}}
    
    This class ensures that prediction errors from Sensor A never
    affect the inhibition decisions for Sensor B, even when both use
    the same engine (e.g., "taylor").
    
    Attributes:
        max_history: Maximum errors to keep per engine (default: 50)
        max_series: Maximum series to track before LRU eviction (default: 10000)
    
    Examples:
        >>> manager = ErrorHistoryManager(max_history=50)
        >>> manager.record_error("sensor_1", "taylor", 5.0)
        >>> manager.record_error("sensor_2", "taylor", 0.1)
        >>> # sensor_1 and sensor_2 have isolated error histories
        >>> errors_sensor_1 = manager.get_errors("sensor_1", "taylor")
        >>> errors_sensor_2 = manager.get_errors("sensor_2", "taylor")
    """
    
    def __init__(
        self,
        max_history: int = 50,
        max_series: int = 10000,
    ) -> None:
        """Initialize error history manager.
        
        Args:
            max_history: Maximum errors per engine per series
            max_series: Maximum series to track (LRU eviction)
        """
        self.max_history = max_history
        self.max_series = max_series
        
        # Structure: {series_id: {engine_name: deque[errors]}}
        self._errors: Dict[str, Dict[str, Deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=max_history))
        )
        
        # LRU tracking for series eviction
        self._series_access_time: Dict[str, float] = {}
        
        # Thread safety
        self._lock = RLock()
    
    def record_error(
        self,
        series_id: str,
        engine_name: str,
        error: float,
    ) -> None:
        """Record a prediction error (thread-safe).
        
        Args:
            series_id: Series identifier
            engine_name: Name of the prediction engine
            error: Absolute prediction error
        """
        if error < 0:
            raise ValueError(f"error must be >= 0, got {error}")
        
        import time
        
        with self._lock:
            now = time.time()
            
            # Record error
            self._errors[series_id][engine_name].append(error)
            self._series_access_time[series_id] = now
            
            # LRU eviction if over limit
            if len(self._errors) > self.max_series:
                oldest_series = min(
                    self._series_access_time,
                    key=self._series_access_time.get
                )
                del self._errors[oldest_series]
                del self._series_access_time[oldest_series]
                logger.debug(
                    "error_history_series_evicted",
                    extra={"series_id": oldest_series, "reason": "lru_eviction"},
                )
        
        logger.debug(
            "error_recorded",
            extra={
                "series_id": series_id,
                "engine_name": engine_name,
                "error": error,
            },
        )
    
    def get_errors(
        self,
        series_id: str,
        engine_name: str,
    ) -> List[float]:
        """Get error history for a specific series and engine (thread-safe).
        
        Args:
            series_id: Series identifier
            engine_name: Name of the prediction engine
        
        Returns:
            List of recent errors (copy, safe to modify)
        """
        with self._lock:
            if series_id not in self._errors:
                return []
            if engine_name not in self._errors[series_id]:
                return []
            
            # Return copy to prevent external modification
            return list(self._errors[series_id][engine_name])
    
    def get_error_dict_for_inhibition(
        self,
        series_id: str,
        engine_names: List[str],
    ) -> Dict[str, List[float]]:
        """Get error dict formatted for InhibitionGate.compute().
        
        This method returns errors in the format expected by
        InhibitionGate, but scoped to a specific series_id.
        
        Args:
            series_id: Series identifier
            engine_names: List of engine names to include
        
        Returns:
            Dict mapping engine_name -> list of recent errors
        """
        with self._lock:
            if series_id not in self._errors:
                return {name: [] for name in engine_names}
            
            return {
                name: list(self._errors[series_id].get(name, deque()))
                for name in engine_names
            }
    
    def get_all_errors_for_series(
        self,
        series_id: str,
    ) -> Dict[str, List[float]]:
        """Get all error histories for a series.
        
        Args:
            series_id: Series identifier
        
        Returns:
            Dict mapping engine_name -> list of errors
        """
        with self._lock:
            if series_id not in self._errors:
                return {}
            
            return {
                name: list(errors)
                for name, errors in self._errors[series_id].items()
            }
    
    def reset(
        self,
        series_id: Optional[str] = None,
        engine_name: Optional[str] = None,
    ) -> None:
        """Reset error history (thread-safe).
        
        Args:
            series_id: If provided, reset only this series
            engine_name: If provided, reset only this engine (requires series_id)
        """
        with self._lock:
            if series_id is None:
                self._errors.clear()
                self._series_access_time.clear()
                logger.info("error_history_reset_all")
            elif engine_name is None:
                if series_id in self._errors:
                    del self._errors[series_id]
                    if series_id in self._series_access_time:
                        del self._series_access_time[series_id]
                    logger.info("error_history_reset_series", extra={"series_id": series_id})
            else:
                if series_id in self._errors and engine_name in self._errors[series_id]:
                    del self._errors[series_id][engine_name]
                    logger.info(
                        "error_history_reset_engine",
                        extra={"series_id": series_id, "engine_name": engine_name},
                    )
    
    def record_errors_from_perceptions(
        self,
        series_id: str,
        perceptions: List,
        actual_value: float,
    ) -> None:
        """Record errors for all perceptions at once.
        
        Convenience method that computes errors from perceptions
        and records them all atomically.
        
        Args:
            series_id: Series identifier
            perceptions: List of EnginePerception (must have predicted_value)
            actual_value: True observed value
        """
        with self._lock:
            for p in perceptions:
                error = abs(p.predicted_value - actual_value)
                self._errors[series_id][p.engine_name].append(error)
            
            import time
            self._series_access_time[series_id] = time.time()


def create_error_history_manager(max_history: int = 50) -> ErrorHistoryManager:
    """Factory function for creating ErrorHistoryManager instances.
    
    Args:
        max_history: Maximum errors per engine per series
    
    Returns:
        Configured ErrorHistoryManager instance
    """
    return ErrorHistoryManager(max_history=max_history)
