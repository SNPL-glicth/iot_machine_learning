"""
RegimeStateManager for managing regime state history.

Maintains historical regime states per sensor with smooth transitions.
"""

import time
from typing import Dict, Optional, Deque
from collections import deque

from .models.regime_state import RegimeState


class RegimeStateManager:
    """Manager for regime state history with smooth transitions."""
    
    def __init__(
        self,
        max_history_size: int = 100,
        min_regime_duration: float = 300.0,  # 5 minutos
    ):
        """
        Initialize regime state manager.
        
        Args:
            max_history_size: Maximum number of states to keep per sensor
            min_regime_duration: Minimum duration for a regime (seconds)
        """
        self._max_history_size = max_history_size
        self._min_regime_duration = min_regime_duration
        
        # sensor_id -> deque of RegimeState
        self._state_history: Dict[int, Deque[RegimeState]] = {}
    
    def smooth_transition(
        self,
        sensor_id: int,
        new_regime: str,
        current_timestamp: float,
        min_duration: Optional[float] = None,
    ) -> str:
        """
        Smooth transitions to avoid flickering between regimes.
        
        Args:
            sensor_id: Sensor identifier
            new_regime: New regime to transition to
            current_timestamp: Current timestamp
            min_duration: Minimum duration for regime (overrides default)
        
        Returns:
            Smoothed regime (may keep current regime if transition too fast)
        """
        min_dur = min_duration or self._min_regime_duration
        
        # Get current state
        current_state = self._get_current_state(sensor_id)
        
        if current_state is None:
            # First state
            self._add_state(sensor_id, new_regime, current_timestamp)
            return new_regime
        
        # If same regime, update timestamp
        if current_state.regime == new_regime:
            self._update_state(sensor_id, current_timestamp)
            return new_regime
        
        # If different regime, check minimum duration
        duration = current_timestamp - current_state.timestamp
        if duration < min_dur:
            # Transition too fast, keep current regime
            return current_state.regime
        
        # Valid transition, add new state
        self._add_state(sensor_id, new_regime, current_timestamp)
        return new_regime
    
    def get_previous_regime(self, sensor_id: int) -> Optional[str]:
        """
        Get previous regime for a sensor.
        
        Args:
            sensor_id: Sensor identifier
        
        Returns:
            Previous regime or None if not available
        """
        history = self._state_history.get(sensor_id)
        if history and len(history) >= 2:
            return history[-2].regime
        return None
    
    def get_transition_duration(self, sensor_id: int) -> Optional[float]:
        """
        Get duration of current transition.
        
        Args:
            sensor_id: Sensor identifier
        
        Returns:
            Transition duration in seconds or None if not available
        """
        history = self._state_history.get(sensor_id)
        if history and len(history) >= 2:
            return history[-1].timestamp - history[-2].timestamp
        return None
    
    def get_regime_duration(self, sensor_id: int) -> Optional[float]:
        """
        Get duration of current regime.
        
        Args:
            sensor_id: Sensor identifier
        
        Returns:
            Regime duration in seconds or None if not available
        """
        current_state = self._get_current_state(sensor_id)
        if current_state:
            return time.time() - current_state.timestamp
        return None
    
    def get_current_regime(self, sensor_id: int) -> Optional[str]:
        """
        Get current regime for a sensor.
        
        Args:
            sensor_id: Sensor identifier
        
        Returns:
            Current regime or None if not available
        """
        current_state = self._get_current_state(sensor_id)
        return current_state.regime if current_state else None
    
    def _get_current_state(self, sensor_id: int) -> Optional[RegimeState]:
        """Get current state for a sensor."""
        history = self._state_history.get(sensor_id)
        if history and len(history) > 0:
            return history[-1]
        return None
    
    def _add_state(self, sensor_id: int, regime: str, timestamp: float) -> None:
        """Add a new state to history."""
        if sensor_id not in self._state_history:
            self._state_history[sensor_id] = deque(maxlen=self._max_history_size)
        
        self._state_history[sensor_id].append(RegimeState(regime=regime, timestamp=timestamp))
    
    def _update_state(self, sensor_id: int, timestamp: float) -> None:
        """Update timestamp of current state."""
        history = self._state_history.get(sensor_id)
        if history and len(history) > 0:
            # Replace last state with updated timestamp
            current_state = history.pop()
            history.append(RegimeState(regime=current_state.regime, timestamp=timestamp))
    
    def cleanup_sensor(self, sensor_id: int) -> None:
        """
        Remove all states for a sensor (for cleanup).
        
        Args:
            sensor_id: Sensor identifier
        """
        self._state_history.pop(sensor_id, None)
