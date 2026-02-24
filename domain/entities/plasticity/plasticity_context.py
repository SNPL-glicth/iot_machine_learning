"""PlasticityContext: Contextual information for adaptive learning.

This frozen dataclass captures the context in which a prediction was made,
enabling context-aware learning rate adaptation and performance tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class RegimeType(str, Enum):
    """Signal regime types for contextual learning."""
    
    STABLE = "stable"
    TRENDING = "trending"
    VOLATILE = "volatile"
    NOISY = "noisy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PlasticityContext:
    """Contextual information for adaptive learning.
    
    Captures the context in which a prediction was made to enable:
    - Adaptive learning rate based on regime, noise, and error patterns
    - Contextual performance tracking (regime + time + volatility)
    - Asymmetric error penalties based on business impact
    
    Attributes:
        regime: Signal regime (STABLE, TRENDING, VOLATILE, NOISY, UNKNOWN)
        noise_ratio: Ratio of noise to signal (0.0 = clean, 1.0 = pure noise)
        volatility: Volatility level (0.0 = stable, 1.0 = highly volatile)
        time_of_day: Hour of day (0-23) for temporal patterns
        consecutive_failures: Number of consecutive prediction failures
        error_magnitude: Magnitude of current prediction error
        is_critical_zone: Whether prediction is in critical threshold zone
        timestamp: When this context was captured
    
    Examples:
        >>> ctx = PlasticityContext(
        ...     regime=RegimeType.STABLE,
        ...     noise_ratio=0.1,
        ...     volatility=0.2,
        ...     time_of_day=14,
        ...     consecutive_failures=0,
        ...     error_magnitude=1.5,
        ...     is_critical_zone=False,
        ... )
        >>> ctx.context_key
        'stable|14|0.2'
    """
    
    regime: RegimeType
    noise_ratio: float
    volatility: float
    time_of_day: int
    consecutive_failures: int
    error_magnitude: float
    is_critical_zone: bool
    timestamp: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Validate context parameters."""
        if not 0.0 <= self.noise_ratio <= 1.0:
            raise ValueError(f"noise_ratio must be in [0, 1], got {self.noise_ratio}")
        if not 0.0 <= self.volatility <= 1.0:
            raise ValueError(f"volatility must be in [0, 1], got {self.volatility}")
        if not 0 <= self.time_of_day <= 23:
            raise ValueError(f"time_of_day must be in [0, 23], got {self.time_of_day}")
        if self.consecutive_failures < 0:
            raise ValueError(f"consecutive_failures must be >= 0, got {self.consecutive_failures}")
        if self.error_magnitude < 0:
            raise ValueError(f"error_magnitude must be >= 0, got {self.error_magnitude}")
    
    @property
    def context_key(self) -> str:
        """Generate context key for performance tracking.
        
        Format: "{regime}|{time_block}|{volatility_binary}"
        
        Time blocks (6-hour windows):
        - 0-5: "0" (night)
        - 6-11: "1" (morning)
        - 12-17: "2" (afternoon)
        - 18-23: "3" (evening)
        
        Volatility binary:
        - 0.0-0.6: "stable"
        - >0.6: "volatile"
        
        This reduces context space from 288 (4×24×3) to 32 (4×4×2).
        
        Returns:
            Context key string for grouping similar contexts
        
        Examples:
            >>> ctx = PlasticityContext(regime=RegimeType.STABLE, noise_ratio=0.1,
            ...                         volatility=0.2, time_of_day=14,
            ...                         consecutive_failures=0, error_magnitude=1.0,
            ...                         is_critical_zone=False)
            >>> ctx.context_key
            'stable|2|stable'
        """
        # Group hours into 6-hour blocks
        time_block = self.time_of_day // 6
        
        # Binary volatility classification
        vol_binary = "volatile" if self.volatility > 0.6 else "stable"
        
        return f"{self.regime.value}|{time_block}|{vol_binary}"
    
    @classmethod
    def create_default(cls, regime: RegimeType = RegimeType.UNKNOWN) -> PlasticityContext:
        """Create a default context with safe values.
        
        Args:
            regime: Signal regime (default: UNKNOWN)
        
        Returns:
            PlasticityContext with default values
        """
        now = datetime.now()
        return cls(
            regime=regime,
            noise_ratio=0.0,
            volatility=0.0,
            time_of_day=now.hour,
            consecutive_failures=0,
            error_magnitude=0.0,
            is_critical_zone=False,
            timestamp=now,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "regime": self.regime.value,
            "noise_ratio": self.noise_ratio,
            "volatility": self.volatility,
            "time_of_day": self.time_of_day,
            "consecutive_failures": self.consecutive_failures,
            "error_magnitude": self.error_magnitude,
            "is_critical_zone": self.is_critical_zone,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
