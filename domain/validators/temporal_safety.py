"""Temporal Safety Validators — prevent data leakage and temporal violations.

Validates:
- Timestamps are ordered
- No future data (prevents data leakage)
- No excessive gaps
- No NaN/Inf in timestamps
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class TemporalViolation:
    """Represents a temporal safety violation.
    
    Attributes:
        violation_type: Type of violation
        index: Index where violation occurred
        message: Human-readable message
        severity: CRITICAL, HIGH, MEDIUM, LOW
    """
    violation_type: str
    index: Optional[int]
    message: str
    severity: str = "HIGH"


class TemporalSafetyValidator:
    """Validates temporal safety constraints.
    
    Prevents:
    - Future data leakage
    - Unordered timestamps
    - Excessive gaps
    - Invalid timestamps (NaN, Inf)
    """
    
    def __init__(
        self,
        max_gap_seconds: float = 3600.0,
        allow_future_seconds: float = 60.0,
    ) -> None:
        """Initialize validator.
        
        Args:
            max_gap_seconds: Maximum allowed gap between consecutive timestamps
            allow_future_seconds: Tolerance for clock skew (allow up to N seconds in future)
        """
        self._max_gap = max_gap_seconds
        self._future_tolerance = allow_future_seconds
    
    def validate(
        self,
        timestamps: List[float],
        strict: bool = True,
    ) -> Tuple[bool, List[TemporalViolation]]:
        """Validate timestamps for safety.
        
        Args:
            timestamps: List of Unix timestamps
            strict: If True, any violation fails; if False, only CRITICAL fails
        
        Returns:
            (is_valid, violations)
        """
        violations = []
        
        if not timestamps:
            return True, []
        
        # Check for NaN/Inf
        for i, ts in enumerate(timestamps):
            if math.isnan(ts) or math.isinf(ts):
                violations.append(TemporalViolation(
                    violation_type="INVALID_TIMESTAMP",
                    index=i,
                    message=f"Timestamp at index {i} is NaN or Inf: {ts}",
                    severity="CRITICAL",
                ))
        
        if violations and strict:
            return False, violations
        
        # Check for future data (data leakage risk)
        now = time.time()
        for i, ts in enumerate(timestamps):
            if ts > now + self._future_tolerance:
                violations.append(TemporalViolation(
                    violation_type="FUTURE_DATA",
                    index=i,
                    message=f"Timestamp at index {i} is in the future: {ts} > {now}",
                    severity="CRITICAL",
                ))
        
        if any(v.severity == "CRITICAL" for v in violations):
            return False, violations
        
        # Check for ordering
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i-1]:
                violations.append(TemporalViolation(
                    violation_type="UNORDERED",
                    index=i,
                    message=f"Timestamps not ordered: {timestamps[i]} < {timestamps[i-1]}",
                    severity="HIGH",
                ))
        
        if strict and any(v.severity in ["CRITICAL", "HIGH"] for v in violations):
            return False, violations
        
        # Check for excessive gaps
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i-1]
            if gap > self._max_gap:
                violations.append(TemporalViolation(
                    violation_type="EXCESSIVE_GAP",
                    index=i,
                    message=f"Gap too large: {gap}s > {self._max_gap}s",
                    severity="MEDIUM",
                ))
        
        # Final decision
        if strict:
            is_valid = len(violations) == 0
        else:
            is_valid = not any(v.severity == "CRITICAL" for v in violations)
        
        return is_valid, violations
    
    def validate_and_raise(
        self,
        timestamps: List[float],
        strict: bool = True,
    ) -> None:
        """Validate and raise ValueError if invalid.
        
        Args:
            timestamps: List of timestamps
            strict: Strictness level
        
        Raises:
            ValueError: If validation fails
        """
        is_valid, violations = self.validate(timestamps, strict=strict)
        
        if not is_valid:
            critical = [v for v in violations if v.severity == "CRITICAL"]
            high = [v for v in violations if v.severity == "HIGH"]
            
            messages = []
            if critical:
                messages.append(f"CRITICAL violations: {len(critical)}")
                for v in critical[:3]:  # Show first 3
                    messages.append(f"  - {v.message}")
            if high:
                messages.append(f"HIGH violations: {len(high)}")
                for v in high[:3]:
                    messages.append(f"  - {v.message}")
            
            raise ValueError("\n".join(messages))
    
    def sanitize(
        self,
        timestamps: List[float],
        values: List[float],
    ) -> Tuple[List[float], List[float]]:
        """Sanitize timestamps by removing invalid entries.
        
        Args:
            timestamps: List of timestamps
            values: Corresponding values
        
        Returns:
            (sanitized_timestamps, sanitized_values)
        """
        if len(timestamps) != len(values):
            raise ValueError(f"Length mismatch: {len(timestamps)} timestamps vs {len(values)} values")
        
        sanitized_ts = []
        sanitized_vals = []
        now = time.time()
        
        for ts, val in zip(timestamps, values):
            # Skip NaN/Inf timestamps
            if math.isnan(ts) or math.isinf(ts):
                continue
            
            # Skip future data
            if ts > now + self._future_tolerance:
                continue
            
            sanitized_ts.append(ts)
            sanitized_vals.append(val)
        
        # Sort by timestamp
        if sanitized_ts:
            paired = sorted(zip(sanitized_ts, sanitized_vals))
            sanitized_ts, sanitized_vals = zip(*paired)
            return list(sanitized_ts), list(sanitized_vals)
        
        return [], []


def validate_temporal_safety(
    timestamps: List[float],
    strict: bool = True,
    max_gap_seconds: float = 3600.0,
) -> Tuple[bool, List[TemporalViolation]]:
    """Convenience function for temporal validation.
    
    Args:
        timestamps: List of timestamps
        strict: Strictness level
        max_gap_seconds: Maximum gap allowed
    
    Returns:
        (is_valid, violations)
    """
    validator = TemporalSafetyValidator(max_gap_seconds=max_gap_seconds)
    return validator.validate(timestamps, strict=strict)
