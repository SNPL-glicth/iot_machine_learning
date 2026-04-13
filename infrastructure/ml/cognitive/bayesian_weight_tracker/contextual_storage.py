"""Contextual error storage — data structures and LRU management.

Isolated storage layer for ContextualPlasticityTracker.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from threading import RLock
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ContextualErrorStorage:
    """Storage layer for contextual errors with LRU eviction.
    
    Structure: {series_id: {engine_name: {context_key: deque[errors]}}}
    
    Thread Safety:
        Uses RLock to protect shared dictionary.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        max_contexts_per_engine: int = 20,
    ) -> None:
        self._window_size = window_size
        self._max_contexts = max_contexts_per_engine
        
        # Structure: {series_id: {engine_name: {context_key: deque[errors]}}}
        self._errors: Dict[str, Dict[str, Dict[str, deque]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: deque(maxlen=window_size)))
        )
        
        # LRU tracking: {series_id: {engine_name: {context_key: timestamp}}}
        self._access_time: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        
        self._lock = RLock()
    
    def record(
        self,
        series_id: str,
        engine_name: str,
        context_key: str,
        error: float,
    ) -> int:
        """Record error and return current count.
        
        Returns:
            Number of errors now stored for this context.
        """
        with self._lock:
            now = time.time()
            
            self._errors[series_id][engine_name][context_key].append(error)
            self._access_time[series_id][engine_name][context_key] = now
            
            # LRU eviction if over limit
            contexts = self._errors[series_id][engine_name]
            if len(contexts) > self._max_contexts:
                oldest_ctx = min(
                    self._access_time[series_id][engine_name],
                    key=self._access_time[series_id][engine_name].get
                )
                del self._errors[series_id][engine_name][oldest_ctx]
                del self._access_time[series_id][engine_name][oldest_ctx]
            
            return len(self._errors[series_id][engine_name][context_key])
    
    def get_errors(
        self,
        series_id: str,
        engine_name: str,
        context_key: str,
    ) -> Optional[deque]:
        """Get error deque for a context (None if not found)."""
        with self._lock:
            series_data = self._errors.get(series_id)
            if not series_data:
                return None
            engine_data = series_data.get(engine_name)
            if not engine_data:
                return None
            return engine_data.get(context_key)
    
    def reset(
        self,
        series_id: Optional[str] = None,
        engine_name: Optional[str] = None,
    ) -> None:
        """Reset storage."""
        with self._lock:
            if series_id is None:
                self._errors.clear()
                self._access_time.clear()
                logger.info("contextual_storage_reset_all")
            elif engine_name is None:
                if series_id in self._errors:
                    del self._errors[series_id]
                    if series_id in self._access_time:
                        del self._access_time[series_id]
                    logger.info("contextual_storage_reset_series", extra={"series_id": series_id})
            else:
                if series_id in self._errors and engine_name in self._errors[series_id]:
                    del self._errors[series_id][engine_name]
                    if series_id in self._access_time and engine_name in self._access_time[series_id]:
                        del self._access_time[series_id][engine_name]
                    logger.info(
                        "contextual_storage_reset_engine",
                        extra={"series_id": series_id, "engine_name": engine_name},
                    )
    
    def get_all_contexts(
        self,
        series_id: str,
        engine_name: str,
    ) -> List[str]:
        """Get all context keys for a series/engine."""
        with self._lock:
            return list(self._errors.get(series_id, {}).get(engine_name, {}).keys())
    
    def count_samples(
        self,
        series_id: str,
        engine_name: str,
        context_key: str,
    ) -> int:
        """Count samples for a context."""
        errors = self.get_errors(series_id, engine_name, context_key)
        return len(errors) if errors else 0
