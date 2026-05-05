"""Input Validation Layer — fail-fast on invalid data before ML pipeline.

Prevents invalid data from reaching ML logic:
- NaN/Inf rejection
- Ordered timestamps
- Minimum window size
- Value range validation

Fail-safe: returns safe fallback responses instead of crashing.

Modularized:
    - validation_rules.py: Specific validation logic
    - input_validator.py: Orchestration and metrics (this file)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .validation_rules import ValidationResult, ValueValidator, TimestampValidator

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates ML pipeline inputs with fail-fast and safe fallback.
    
    Validation rules:
    - No NaN or Inf values
    - Timestamps must be ordered (ascending)
    - Minimum window size
    - Values within reasonable range
    - No duplicate timestamps
    """
    
    def __init__(
        self,
        min_window_size: int = 3,
        max_window_size: int = 1000,
        allow_duplicates: bool = False,
        value_min: Optional[float] = None,
        value_max: Optional[float] = None,
        strict_mode: bool = True,
    ):
        """Initialize input validator.
        
        Args:
            min_window_size: Minimum required window size
            max_window_size: Maximum allowed window size
            allow_duplicates: Whether to allow duplicate timestamps
            value_min: Minimum allowed value (None = no limit)
            value_max: Maximum allowed value (None = no limit)
            strict_mode: If True, reject invalid data; if False, attempt sanitization
        """
        if min_window_size < 1:
            raise ValueError("min_window_size must be >= 1")
        if max_window_size < min_window_size:
            raise ValueError("max_window_size must be >= min_window_size")
        
        self._min_window_size = min_window_size
        self._max_window_size = max_window_size
        self._allow_duplicates = allow_duplicates
        self._value_min = value_min
        self._value_max = value_max
        self._strict_mode = strict_mode
        
        # Metrics
        self._rejected_count = 0
        self._sanitized_count = 0
        self._valid_count = 0
    
    def validate(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> ValidationResult:
        """Validate input data.
        
        Args:
            values: List of values
            timestamps: Optional list of timestamps
        
        Returns:
            ValidationResult with decision and optional sanitized data
        """
        # Check window size
        if len(values) < self._min_window_size:
            self._rejected_count += 1
            return ValidationResult(
                valid=False,
                error_code="WINDOW_TOO_SMALL",
                error_message=f"Window size {len(values)} < minimum {self._min_window_size}",
            )
        
        if len(values) > self._max_window_size:
            self._rejected_count += 1
            return ValidationResult(
                valid=False,
                error_code="WINDOW_TOO_LARGE",
                error_message=f"Window size {len(values)} > maximum {self._max_window_size}",
            )
        
        # Check timestamps length match
        if timestamps is not None and len(timestamps) != len(values):
            self._rejected_count += 1
            return ValidationResult(
                valid=False,
                error_code="LENGTH_MISMATCH",
                error_message=f"Values length {len(values)} != timestamps length {len(timestamps)}",
            )
        
        # Validate values
        values_result = self._validate_values(values)
        if not values_result.valid:
            self._rejected_count += 1
            return values_result
        
        # Validate timestamps if provided
        if timestamps is not None:
            timestamps_result = self._validate_timestamps(timestamps)
            if not timestamps_result.valid:
                self._rejected_count += 1
                return timestamps_result
            
            # Check for duplicates if not allowed
            if not self._allow_duplicates:
                dup_result = self._check_duplicates(timestamps)
                if not dup_result.valid:
                    self._rejected_count += 1
                    return dup_result
        
        # All checks passed
        self._valid_count += 1
        return ValidationResult(
            valid=True,
            sanitized_values=values_result.sanitized_values or values,
            sanitized_timestamps=timestamps_result.sanitized_timestamps if timestamps is not None else None,
        )
    
    def _validate_values(self, values: List[float]) -> ValidationResult:
        """Validate values for NaN/Inf and range."""
        result = ValueValidator.validate(
            values,
            self._value_min,
            self._value_max,
            self._strict_mode,
        )
        if result.sanitized_values is not None:
            self._sanitized_count += 1
        return result
    
    def _validate_timestamps(self, timestamps: List[float]) -> ValidationResult:
        """Validate timestamps for NaN/Inf and ordering."""
        # Check finite
        result = TimestampValidator.validate_finite(timestamps)
        if not result.valid:
            return result
        
        # Check ordering
        result = TimestampValidator.validate_ordering(timestamps, self._strict_mode)
        if result.sanitized_timestamps is not None:
            self._sanitized_count += 1
        return result
    
    def _check_duplicates(self, timestamps: List[float]) -> ValidationResult:
        """Check for duplicate timestamps."""
        return TimestampValidator.check_duplicates(timestamps)
    
    def get_metrics(self) -> dict:
        """Get validation metrics.
        
        Returns:
            Dict with rejected_count, sanitized_count, valid_count, rejection_rate
        """
        total = self._rejected_count + self._sanitized_count + self._valid_count
        rejection_rate = self._rejected_count / total if total > 0 else 0.0
        
        return {
            "rejected_count": self._rejected_count,
            "sanitized_count": self._sanitized_count,
            "valid_count": self._valid_count,
            "total_requests": total,
            "rejection_rate": rejection_rate,
        }
    
    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self._rejected_count = 0
        self._sanitized_count = 0
        self._valid_count = 0


class ValidationError(Exception):
    """Exception raised when input validation fails.
    
    Attributes:
        result: ValidationResult with details
    """
    
    def __init__(self, result: ValidationResult):
        self.result = result
        message = f"{result.error_code}: {result.error_message}"
        super().__init__(message)
