"""Engine Health Monitor.

Monitors engine health and auto-inhibits degraded engines.
Tracks consecutive failures and time since last success.
"""

from __future__ import annotations

import logging
from datetime import datetime
from threading import RLock
from typing import Dict, List, Optional, Tuple

from ....domain.entities.plasticity.engine_plasticity_state import EnginePlasticityState
from .inhibition_rules import evaluate_inhibition, build_health_summary

logger = logging.getLogger(__name__)


class EngineHealthMonitor:
    """Monitors engine health and applies auto-inhibition.
    
    Auto-inhibits engines when:
    - consecutive_failures >= threshold
    - hours_since_last_success > max_hours_without_success
    
    Thread Safety:
        Uses RLock to protect shared state dictionary from race conditions.
        RLock allows re-entrance from the same thread (prevents deadlocks).
    
    Attributes:
        failure_threshold: Consecutive failures to trigger inhibition (default: 10)
        max_hours_without_success: Max hours without success (default: 1.0)
        error_tolerance: Error below this is considered success (default: 1.0)
    
    Examples:
        >>> monitor = EngineHealthMonitor(failure_threshold=5)
        >>> state = EnginePlasticityState.create_initial("taylor", "sensor_1")
        >>> # After 5 failures
        >>> for _ in range(5):
        ...     state = state.with_failure(10.0)
        >>> monitor.check_health(state)
        >>> state.is_inhibited
        True
    """
    
    def __init__(
        self,
        failure_threshold: int = 10,
        max_hours_without_success: float = 1.0,
        error_tolerance: float = 1.0,
    ) -> None:
        """Initialize engine health monitor.
        
        Args:
            failure_threshold: Consecutive failures to trigger inhibition
            max_hours_without_success: Max hours without success before inhibition
            error_tolerance: Error below this is considered success
        
        Raises:
            ValueError: If parameters are invalid
        """
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")
        if max_hours_without_success <= 0:
            raise ValueError(f"max_hours_without_success must be > 0, got {max_hours_without_success}")
        if error_tolerance < 0:
            raise ValueError(f"error_tolerance must be >= 0, got {error_tolerance}")
        
        self.failure_threshold = failure_threshold
        self.max_hours_without_success = max_hours_without_success
        self.error_tolerance = error_tolerance
        
        # Track engine states
        self._states: Dict[str, Dict[str, EnginePlasticityState]] = {}
        
        # Thread safety: RLock for protecting shared state dictionary
        self._lock = RLock()
    
    def record_prediction(
        self,
        series_id: str,
        engine_name: str,
        error: float,
    ) -> EnginePlasticityState:
        """Record prediction result and update engine state (thread-safe).
        
        Args:
            series_id: Series identifier
            engine_name: Name of the prediction engine
            error: Absolute prediction error
        
        Returns:
            Updated engine state
        
        Raises:
            ValueError: If error is negative
        """
        if error < 0:
            raise ValueError(f"error must be >= 0, got {error}")
        
        with self._lock:
            # Get or create state
            if series_id not in self._states:
                self._states[series_id] = {}
            
            if engine_name not in self._states[series_id]:
                self._states[series_id][engine_name] = EnginePlasticityState.create_initial(
                    engine_name=engine_name,
                    series_id=series_id,
                )
            
            state = self._states[series_id][engine_name]
            
            # Update state based on error
            is_success = error <= self.error_tolerance
            
            if is_success:
                state = state.with_success(error)
            else:
                state = state.with_failure(error)
            
            # Check for auto-inhibition
            state = self._check_inhibition(state)
            
            # Store updated state
            self._states[series_id][engine_name] = state
            
            return state
    
    def _check_inhibition(
        self,
        state: EnginePlasticityState,
    ) -> EnginePlasticityState:
        """Check if engine should be inhibited (delegates to inhibition_rules)."""
        return evaluate_inhibition(
            state,
            self.failure_threshold,
            self.max_hours_without_success,
        )
    
    def get_state(
        self,
        series_id: str,
        engine_name: str,
    ) -> Optional[EnginePlasticityState]:
        """Get current state for an engine (thread-safe).
        
        Args:
            series_id: Series identifier
            engine_name: Name of the prediction engine
        
        Returns:
            Engine state or None if not tracked
        """
        with self._lock:
            return self._states.get(series_id, {}).get(engine_name)
    
    def get_inhibited_engines(
        self,
        series_id: str,
    ) -> List[Tuple[str, str]]:
        """Get list of inhibited engines for a series (thread-safe).
        
        Args:
            series_id: Series identifier
        
        Returns:
            List of tuples (engine_name, inhibition_reason)
        """
        with self._lock:
            if series_id not in self._states:
                return []
            
            return [
                (engine_name, state.inhibition_reason or "unknown")
                for engine_name, state in self._states[series_id].items()
                if state.is_inhibited
            ]
    
    def is_inhibited(
        self,
        series_id: str,
        engine_name: str,
    ) -> bool:
        """Check if an engine is inhibited (thread-safe).
        
        Args:
            series_id: Series identifier
            engine_name: Name of the prediction engine
        
        Returns:
            True if inhibited, False otherwise
        """
        state = self.get_state(series_id, engine_name)
        return state.is_inhibited if state else False
    
    def reset(
        self,
        series_id: Optional[str] = None,
        engine_name: Optional[str] = None,
    ) -> None:
        """Reset tracked states (thread-safe).
        
        Args:
            series_id: If provided, reset only this series
            engine_name: If provided, reset only this engine (requires series_id)
        """
        with self._lock:
            if series_id is None:
                self._states.clear()
                logger.info("health_monitor_reset_all")
            elif engine_name is None:
                if series_id in self._states:
                    del self._states[series_id]
                    logger.info("health_monitor_reset_series", extra={"series_id": series_id})
            else:
                if series_id in self._states and engine_name in self._states[series_id]:
                    del self._states[series_id][engine_name]
                    logger.info(
                        "health_monitor_reset_engine",
                        extra={"series_id": series_id, "engine_name": engine_name},
                    )
    
    def get_health_summary(
        self,
        series_id: str,
    ) -> Dict[str, Dict]:
        """Get health summary for all engines in a series (thread-safe).
        
        Args:
            series_id: Series identifier
        
        Returns:
            Dict mapping engine_name to health metrics
        """
        with self._lock:
            if series_id not in self._states:
                return {}
            return build_health_summary(self._states[series_id])
