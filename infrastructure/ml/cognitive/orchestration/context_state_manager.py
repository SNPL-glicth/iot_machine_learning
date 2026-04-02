"""Context state manager for series-isolated state tracking with persistence.

R-1: Zero-Leakage implementation. All per-series state is isolated
by series_id to prevent cross-contamination between sensors.

Fase 5: Added async persistence support for Kubernetes recovery.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import EnginePerception

if TYPE_CHECKING:
    from .persistence_adapter import PersistenceAdapter


@dataclass
class SeriesState:
    """Per-series isolated state container.
    
    All mutable state that was previously shared at orchestrator
    level is now stored per-series, preventing cross-contamination.
    """
    
    last_regime: Optional[str] = None
    last_perceptions: List[EnginePerception] = field(default_factory=list)
    last_plasticity_context: Optional[object] = None
    prediction_count: int = 0
    last_prediction_timestamp: Optional[float] = None


class ContextStateManager:
    """Manages per-series state with thread-safe isolation and persistence.
    
    R-1 Refactor: Replaces shared orchestrator state with series-keyed storage.
    Fase 5: Added PersistenceAdapter for Kubernetes pod restart recovery.
    
    Guarantees: Sensor A can never read or influence Sensor B state.
    """
    
    def __init__(
        self, 
        max_series: int = 10000,
        persistence: Optional["PersistenceAdapter"] = None,
    ) -> None:
        self._max_series = max_series
        self._persistence = persistence
        self._state: Dict[str, SeriesState] = {}
        self._lock = threading.RLock()
    
    def get_state(self, series_id: str) -> SeriesState:
        """Get or create isolated state for a series (thread-safe).
        
        If persistence enabled, attempts to load from storage first.
        """
        with self._lock:
            if series_id not in self._state:
                # Try to load from persistence first
                if self._persistence:
                    loaded = self._load_from_persistence_sync(series_id)
                    if loaded:
                        self._state[series_id] = loaded
                        return loaded
                
                if len(self._state) >= self._max_series:
                    self._evict_oldest()
                self._state[series_id] = SeriesState()
            
            # Update timestamp on access
            self._state[series_id].last_prediction_timestamp = time.time()
            return self._state[series_id]
    
    def _load_from_persistence_sync(self, series_id: str) -> Optional[SeriesState]:
        """Synchronous wrapper for async persistence load."""
        if not self._persistence:
            return None
        try:
            # Use event loop if available, otherwise return None
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule async load for later
                asyncio.create_task(self._persistence.load_state(series_id))
                return None
            return loop.run_until_complete(self._persistence.load_state(series_id))
        except RuntimeError:
            return None
    
    def _persist_async(self, series_id: str, state: SeriesState) -> None:
        """Fire-and-forget persistence save."""
        if not self._persistence:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._persistence.save_state(series_id, state))
        except RuntimeError:
            pass
    
    def update_regime(self, series_id: str, regime: Optional[str]) -> None:
        """Update regime for specific series only."""
        state = self.get_state(series_id)
        with self._lock:
            state.last_regime = regime
            self._persist_async(series_id, state)
    
    def update_perceptions(
        self, 
        series_id: str, 
        perceptions: List[EnginePerception]
    ) -> None:
        """Update perceptions for specific series only."""
        state = self.get_state(series_id)
        with self._lock:
            state.last_perceptions = perceptions
            self._persist_async(series_id, state)
    
    def get_regime(self, series_id: str) -> Optional[str]:
        """Get regime for specific series only."""
        return self.get_state(series_id).last_regime
    
    def get_perceptions(self, series_id: str) -> List[EnginePerception]:
        """Get perceptions for specific series only."""
        return self.get_state(series_id).last_perceptions
    
    def increment_prediction_count(self, series_id: str) -> int:
        """Increment and return prediction count for series."""
        state = self.get_state(series_id)
        with self._lock:
            state.prediction_count += 1
            self._persist_async(series_id, state)
            return state.prediction_count
    
    def _evict_oldest(self) -> None:
        """Evict oldest series when at capacity (caller holds lock)."""
        if self._state:
            oldest = min(
                self._state.items(), 
                key=lambda x: x[1].last_prediction_timestamp or 0
            )
            del self._state[oldest[0]]
    
    def clear(self, series_id: Optional[str] = None) -> None:
        """Clear state for specific series or all."""
        with self._lock:
            if series_id is None:
                self._state.clear()
            else:
                self._state.pop(series_id, None)
    
    def get_metrics(self) -> Dict[str, int]:
        """Return metrics for monitoring."""
        with self._lock:
            return {
                "tracked_series": len(self._state),
                "max_series": self._max_series,
                "persistence_enabled": self._persistence is not None,
            }
