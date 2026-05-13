"""TimeStep value object (ARCH-SEV-3).

Decouples domain from infrastructure temporal details.

Applies OCP: Adding new time units only requires extending enum.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TimeUnit(Enum):
    """Time units for time step (ARCH-SEV-3).
    
    Applies OCP: New units can be added without modifying TimeStep.
    """
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    UNKNOWN = "unknown"
    
    def to_seconds(self, value: float) -> float:
        """Convert value in this unit to seconds.
        
        Args:
            value: Value in this unit.
        
        Returns:
            Value in seconds.
        """
        conversions = {
            TimeUnit.SECONDS: 1.0,
            TimeUnit.MILLISECONDS: 0.001,
            TimeUnit.MINUTES: 60.0,
            TimeUnit.HOURS: 3600.0,
            TimeUnit.DAYS: 86400.0,
            TimeUnit.UNKNOWN: 1.0,  # Assume seconds
        }
        return value * conversions[self]
    
    @classmethod
    def from_string(cls, unit_str: str) -> TimeUnit:
        """Parse time unit from string.
        
        Args:
            unit_str: Unit string (case-insensitive).
        
        Returns:
            Corresponding TimeUnit enum.
        """
        mapping = {
            "s": cls.SECONDS,
            "sec": cls.SECONDS,
            "seconds": cls.SECONDS,
            "ms": cls.MILLISECONDS,
            "milliseconds": cls.MILLISECONDS,
            "m": cls.MINUTES,
            "min": cls.MINUTES,
            "minutes": cls.MINUTES,
            "h": cls.HOURS,
            "hr": cls.HOURS,
            "hours": cls.HOURS,
            "d": cls.DAYS,
            "day": cls.DAYS,
            "days": cls.DAYS,
        }
        return mapping.get(unit_str.lower(), cls.UNKNOWN)


@dataclass(frozen=True)
class TimeStep:
    """Time step value object (ARCH-SEV-3).
    
    Represents temporal spacing between data points.
    
    Attributes:
        value: Numeric time step value.
        unit: Time unit.
    
    Applies OCP: Domain entity doesn't need to know about unit conversions.
    Applies SRP: Encapsulates time step representation.
    """
    value: float
    unit: TimeUnit = TimeUnit.SECONDS
    
    def __post_init__(self):
        """Validate time step."""
        if self.value < 0:
            raise ValueError(f"TimeStep value must be >= 0, got {self.value}")
    
    def to_seconds(self) -> float:
        """Convert to seconds.
        
        Returns:
            Time step in seconds.
        """
        return self.unit.to_seconds(self.value)
    
    def to_milliseconds(self) -> float:
        """Convert to milliseconds.
        
        Returns:
            Time step in milliseconds.
        """
        return self.to_seconds() * 1000.0
    
    @classmethod
    def from_seconds(cls, seconds: float) -> TimeStep:
        """Create from seconds.
        
        Args:
            seconds: Time step in seconds.
        
        Returns:
            TimeStep instance.
        """
        return cls(value=seconds, unit=TimeUnit.SECONDS)
    
    @classmethod
    def from_string(cls, time_str: str) -> TimeStep:
        """Parse from string like '5s', '100ms', '1h'.
        
        Args:
            time_str: Time string with value and unit.
        
        Returns:
            TimeStep instance.
        
        Examples:
            >>> TimeStep.from_string('5s')
            TimeStep(value=5.0, unit=TimeUnit.SECONDS)
            >>> TimeStep.from_string('100ms')
            TimeStep(value=100.0, unit=TimeUnit.MILLISECONDS)
        """
        import re
        match = re.match(r'([\d.]+)\s*([a-zA-Z]+)', time_str.strip())
        if not match:
            raise ValueError(f"Invalid time string format: {time_str}")
        
        value = float(match.group(1))
        unit_str = match.group(2)
        unit = TimeUnit.from_string(unit_str)
        
        return cls(value=value, unit=unit)
    
    def __str__(self) -> str:
        """String representation."""
        unit_abbr = {
            TimeUnit.SECONDS: "s",
            TimeUnit.MILLISECONDS: "ms",
            TimeUnit.MINUTES: "min",
            TimeUnit.HOURS: "h",
            TimeUnit.DAYS: "d",
            TimeUnit.UNKNOWN: "?",
        }
        return f"{self.value}{unit_abbr[self.unit]}"
    
    def __repr__(self) -> str:
        """Repr."""
        return f"TimeStep(value={self.value}, unit={self.unit.value})"
