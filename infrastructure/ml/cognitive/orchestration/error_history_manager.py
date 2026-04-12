"""Error History Manager — Redis-backed with memory fallback.
Env: ML_ERROR_HISTORY_BACKEND = "memory" | "redis" (default: "memory")
Redis: error_history:{series_id}:{engine_name} → List maxlen=50
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from threading import RLock
from typing import Any, Deque, Dict, List, Optional, Set

from .error_history_redis import ErrorHistoryRedis

logger = logging.getLogger(__name__)


class ErrorHistoryManager:
    """Manages prediction error history per series_id and engine."""
    
    def __init__(
        self,
        max_history: int = 50,
        max_series: int = 10000,
        redis_client: Optional[Any] = None,
        backend: Optional[str] = None,
    ) -> None:
        self.max_history = max_history
        self.max_series = max_series
        
        self._backend = backend or os.environ.get("ML_ERROR_HISTORY_BACKEND", "memory")
        
        # Redis component
        self._redis_backend = None
        if self._backend == "redis" and redis_client is not None:
            self._redis_backend = ErrorHistoryRedis(redis_client, max_history)
        elif self._backend == "redis":
            logger.warning("redis_no_client_fallback")
            self._backend = "memory"
        
        # Memory structures
        self._errors: Dict[str, Dict[str, Deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=max_history))
        )
        self._series_access_time: Dict[str, float] = {}
        self._lock = RLock()
        
        logger.info("initialized", extra={"backend": self._backend, "max": max_history})
    
    @property
    def backend(self) -> str:
        return self._backend
    
    def _record_to_memory(self, s: str, e: str, err: float) -> None:
        with self._lock:
            self._errors[s][e].append(err)
            self._series_access_time[s] = time.time()
            if len(self._errors) > self.max_series:
                oldest = min(self._series_access_time, key=self._series_access_time.get)
                del self._errors[oldest]
                del self._series_access_time[oldest]
    
    def record_error(
        self,
        series_id: str,
        engine_name: str,
        error: float,
        regime: str = "",
    ) -> None:
        """Record a prediction error."""
        if error < 0:
            raise ValueError(f"error >= 0 required, got {error}")
        
        if self._redis_backend and self._redis_backend.record(series_id, engine_name, error, regime):
            return
        self._record_to_memory(series_id, engine_name, error)
    
    def get_errors(self, series_id: str, engine_name: str) -> List[float]:
        """Get error history for series and engine."""
        # Try Redis first
        if self._redis_backend is not None:
            errors = self._redis_backend.get_errors(series_id, engine_name)
            if errors is not None:
                return errors
        
        # Memory fallback
        with self._lock:
            if series_id not in self._errors or engine_name not in self._errors[series_id]:
                return []
            return list(self._errors[series_id][engine_name])
    
    def get_error_dict_for_inhibition(
        self,
        series_id: str,
        engine_names: List[str],
    ) -> Dict[str, List[float]]:
        """Get error dict for InhibitionGate."""
        return {name: self.get_errors(series_id, name) for name in engine_names}
    
    def get_all_errors_for_series(self, series_id: str) -> Dict[str, List[float]]:
        """Get all error histories for a series."""
        # Try Redis first
        if self._redis_backend is not None:
            result = self._redis_backend.get_all_engines(series_id)
            if result is not None:
                return result
        
        # Memory fallback
        with self._lock:
            if series_id not in self._errors:
                return {}
            return {name: list(errors) for name, errors in self._errors[series_id].items()}
    
    def reset(self, s: Optional[str] = None, e: Optional[str] = None) -> None:
        if self._redis_backend:
            self._redis_backend.reset(s, e)
        with self._lock:
            if s is None:
                self._errors.clear()
                self._series_access_time.clear()
            elif e is None:
                self._errors.pop(s, None)
                self._series_access_time.pop(s, None)
            else:
                if s in self._errors:
                    self._errors[s].pop(e, None)
    
    def record_errors_from_perceptions(
        self,
        series_id: str,
        perceptions: List,
        actual_value: float,
    ) -> None:
        """Record errors for all perceptions at once."""
        for p in perceptions:
            self.record_error(series_id, p.engine_name, abs(p.predicted_value - actual_value))
        
        if self._backend == "memory":
            with self._lock:
                self._series_access_time[series_id] = time.time()
    
    def get_all_series_ids(self) -> List[str]:
        """Get all series IDs with error history (for gossip protocol)."""
        series_ids: Set[str] = set()
        
        # Get from Redis
        if self._redis_backend is not None:
            series_ids.update(self._redis_backend.get_all_series_ids())
        
        # Get from memory
        with self._lock:
            series_ids.update(self._errors.keys())
        
        return list(series_ids)


def create_error_history_manager(
    max_history: int = 50,
    redis_client: Optional[Any] = None,
) -> ErrorHistoryManager:
    """Factory for ErrorHistoryManager."""
    return ErrorHistoryManager(max_history=max_history, redis_client=redis_client)
