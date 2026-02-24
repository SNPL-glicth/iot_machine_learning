"""EnginePlasticityState: State tracking for individual prediction engines.

Tracks the learning state and health of a single engine within the
adaptive plasticity system.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass(frozen=True)
class EnginePlasticityState:
    """State tracking for a single prediction engine.
    
    Tracks learning progress, health metrics, and inhibition status
    for adaptive weight adjustment and auto-inhibition.
    
    Attributes:
        engine_name: Name of the prediction engine
        series_id: Series identifier this state applies to
        consecutive_failures: Number of consecutive prediction failures
        consecutive_successes: Number of consecutive prediction successes
        last_error: Most recent prediction error
        last_success_time: Timestamp of last successful prediction
        last_failure_time: Timestamp of last failed prediction
        is_inhibited: Whether engine is currently inhibited
        inhibition_reason: Reason for inhibition (if inhibited)
        inhibition_cooldown_until: Timestamp when inhibition cooldown expires
        total_predictions: Total number of predictions made
        total_errors: Total number of errors
        
    Examples:
        >>> state = EnginePlasticityState(
        ...     engine_name="taylor",
        ...     series_id="sensor_123",
        ...     consecutive_failures=0,
        ...     consecutive_successes=5,
        ...     last_error=0.5,
        ...     last_success_time=datetime.now(),
        ...     is_inhibited=False,
        ... )
        >>> state.failure_rate
        0.0
    """
    
    engine_name: str
    series_id: str
    consecutive_failures: int
    consecutive_successes: int
    last_error: float
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    is_inhibited: bool = False
    inhibition_reason: Optional[str] = None
    inhibition_cooldown_until: Optional[datetime] = None
    total_predictions: int = 0
    total_errors: int = 0
    
    def __post_init__(self) -> None:
        """Validate state parameters."""
        if self.consecutive_failures < 0:
            raise ValueError(f"consecutive_failures must be >= 0, got {self.consecutive_failures}")
        if self.consecutive_successes < 0:
            raise ValueError(f"consecutive_successes must be >= 0, got {self.consecutive_successes}")
        if self.last_error < 0:
            raise ValueError(f"last_error must be >= 0, got {self.last_error}")
        if self.total_predictions < 0:
            raise ValueError(f"total_predictions must be >= 0, got {self.total_predictions}")
        if self.total_errors < 0:
            raise ValueError(f"total_errors must be >= 0, got {self.total_errors}")
        if self.total_errors > self.total_predictions:
            raise ValueError(f"total_errors ({self.total_errors}) cannot exceed total_predictions ({self.total_predictions})")
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate.
        
        Returns:
            Failure rate in [0, 1], or 0.0 if no predictions
        """
        if self.total_predictions == 0:
            return 0.0
        return self.total_errors / self.total_predictions
    
    @property
    def hours_since_last_success(self) -> Optional[float]:
        """Calculate hours since last successful prediction.
        
        Returns:
            Hours since last success, or None if never succeeded
        """
        if self.last_success_time is None:
            return None
        delta = datetime.now() - self.last_success_time
        return delta.total_seconds() / 3600.0
    
    def should_attempt_recovery(self) -> bool:
        """Check if engine should attempt recovery from inhibition.
        
        Returns:
            True if inhibited and cooldown period has expired
        """
        if not self.is_inhibited:
            return False
        if self.inhibition_cooldown_until is None:
            return False
        return datetime.now() >= self.inhibition_cooldown_until
    
    @classmethod
    def create_initial(cls, engine_name: str, series_id: str) -> EnginePlasticityState:
        """Create initial state for a new engine.
        
        Args:
            engine_name: Name of the prediction engine
            series_id: Series identifier
        
        Returns:
            EnginePlasticityState with initial values
        """
        return cls(
            engine_name=engine_name,
            series_id=series_id,
            consecutive_failures=0,
            consecutive_successes=0,
            last_error=0.0,
            last_success_time=None,
            last_failure_time=None,
            is_inhibited=False,
            inhibition_reason=None,
            inhibition_cooldown_until=None,
            total_predictions=0,
            total_errors=0,
        )
    
    def with_success(self, error: float) -> EnginePlasticityState:
        """Create new state after a successful prediction.
        
        Args:
            error: Prediction error (considered success if below threshold)
        
        Returns:
            New EnginePlasticityState with updated success metrics
        """
        return EnginePlasticityState(
            engine_name=self.engine_name,
            series_id=self.series_id,
            consecutive_failures=0,
            consecutive_successes=self.consecutive_successes + 1,
            last_error=error,
            last_success_time=datetime.now(),
            last_failure_time=self.last_failure_time,
            is_inhibited=False,
            inhibition_reason=None,
            inhibition_cooldown_until=None,
            total_predictions=self.total_predictions + 1,
            total_errors=self.total_errors,
        )
    
    def with_failure(self, error: float) -> EnginePlasticityState:
        """Create new state after a failed prediction.
        
        Args:
            error: Prediction error
        
        Returns:
            New EnginePlasticityState with updated failure metrics
        """
        return EnginePlasticityState(
            engine_name=self.engine_name,
            series_id=self.series_id,
            consecutive_failures=self.consecutive_failures + 1,
            consecutive_successes=0,
            last_error=error,
            last_success_time=self.last_success_time,
            last_failure_time=datetime.now(),
            is_inhibited=self.is_inhibited,
            inhibition_reason=self.inhibition_reason,
            inhibition_cooldown_until=self.inhibition_cooldown_until,
            total_predictions=self.total_predictions + 1,
            total_errors=self.total_errors + 1,
        )
    
    def with_inhibition(self, reason: str, cooldown_minutes: int = 5) -> EnginePlasticityState:
        """Create new state with inhibition applied.
        
        Args:
            reason: Reason for inhibition
            cooldown_minutes: Minutes until recovery attempt allowed (default: 5)
        
        Returns:
            New EnginePlasticityState with inhibition set
        """
        return EnginePlasticityState(
            engine_name=self.engine_name,
            series_id=self.series_id,
            consecutive_failures=self.consecutive_failures,
            consecutive_successes=self.consecutive_successes,
            last_error=self.last_error,
            last_success_time=self.last_success_time,
            last_failure_time=self.last_failure_time,
            is_inhibited=True,
            inhibition_reason=reason,
            inhibition_cooldown_until=datetime.now() + timedelta(minutes=cooldown_minutes),
            total_predictions=self.total_predictions,
            total_errors=self.total_errors,
        )
    
    def with_recovery(self) -> EnginePlasticityState:
        """Create new state with inhibition cleared after successful recovery.
        
        Returns:
            New EnginePlasticityState with inhibition cleared
        """
        return EnginePlasticityState(
            engine_name=self.engine_name,
            series_id=self.series_id,
            consecutive_failures=0,
            consecutive_successes=1,
            last_error=self.last_error,
            last_success_time=datetime.now(),
            last_failure_time=self.last_failure_time,
            is_inhibited=False,
            inhibition_reason=None,
            inhibition_cooldown_until=None,
            total_predictions=self.total_predictions,
            total_errors=self.total_errors,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "engine_name": self.engine_name,
            "series_id": self.series_id,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_error": self.last_error,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "is_inhibited": self.is_inhibited,
            "inhibition_reason": self.inhibition_reason,
            "total_predictions": self.total_predictions,
            "total_errors": self.total_errors,
            "failure_rate": self.failure_rate,
        }
