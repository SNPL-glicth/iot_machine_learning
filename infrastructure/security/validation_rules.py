"""Validation Rules — specific validation logic for values and timestamps.

Contains validation methods for NaN/Inf, ordering, ranges, and duplicates.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Result of input validation.
    
    Attributes:
        valid: Whether input is valid
        error_code: Error code if invalid
        error_message: Human-readable error message
        sanitized_values: Sanitized values (if recoverable)
        sanitized_timestamps: Sanitized timestamps (if recoverable)
    """
    valid: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    sanitized_values: Optional[List[float]] = None
    sanitized_timestamps: Optional[List[float]] = None


class ValueValidator:
    """Validates values for NaN/Inf and range."""
    
    @staticmethod
    def validate(
        values: List[float],
        value_min: Optional[float],
        value_max: Optional[float],
        strict_mode: bool,
    ) -> ValidationResult:
        """Validate values for NaN/Inf and range.
        
        Args:
            values: List of values
            value_min: Minimum allowed value (None = no limit)
            value_max: Maximum allowed value (None = no limit)
            strict_mode: If True, reject invalid; if False, sanitize
        
        Returns:
            ValidationResult
        """
        # Check for NaN/Inf
        invalid_indices = [i for i, v in enumerate(values) if not math.isfinite(v)]
        
        if invalid_indices:
            if strict_mode:
                return ValidationResult(
                    valid=False,
                    error_code="INVALID_VALUES",
                    error_message=f"Found {len(invalid_indices)} NaN/Inf values at indices {invalid_indices[:5]}",
                )
            else:
                # Sanitize: replace with median
                valid_values = [v for v in values if math.isfinite(v)]
                if not valid_values:
                    return ValidationResult(
                        valid=False,
                        error_code="ALL_INVALID",
                        error_message="All values are NaN/Inf",
                    )
                
                median = float(np.median(valid_values))
                sanitized = [v if math.isfinite(v) else median for v in values]
                
                logger.warning(
                    f"input_validator_sanitized_values: replaced {len(invalid_indices)} invalid values with median {median:.2f}"
                )
                
                return ValidationResult(
                    valid=True,
                    sanitized_values=sanitized,
                )
        
        # Check range if specified
        if value_min is not None or value_max is not None:
            out_of_range = []
            for i, v in enumerate(values):
                if value_min is not None and v < value_min:
                    out_of_range.append((i, v, "below_min"))
                if value_max is not None and v > value_max:
                    out_of_range.append((i, v, "above_max"))
            
            if out_of_range:
                if strict_mode:
                    return ValidationResult(
                        valid=False,
                        error_code="OUT_OF_RANGE",
                        error_message=f"Found {len(out_of_range)} out-of-range values",
                    )
                else:
                    # Sanitize: clamp to range
                    sanitized = []
                    for v in values:
                        if value_min is not None and v < value_min:
                            sanitized.append(value_min)
                        elif value_max is not None and v > value_max:
                            sanitized.append(value_max)
                        else:
                            sanitized.append(v)
                    
                    logger.warning(
                        f"input_validator_clamped_values: clamped {len(out_of_range)} values to range [{value_min}, {value_max}]"
                    )
                    
                    return ValidationResult(
                        valid=True,
                        sanitized_values=sanitized,
                    )
        
        return ValidationResult(valid=True)


class TimestampValidator:
    """Validates timestamps for NaN/Inf, ordering, and duplicates."""
    
    @staticmethod
    def validate_finite(timestamps: List[float]) -> ValidationResult:
        """Validate timestamps for NaN/Inf.
        
        Args:
            timestamps: List of timestamps
        
        Returns:
            ValidationResult
        """
        invalid_indices = [i for i, t in enumerate(timestamps) if not math.isfinite(t)]
        
        if invalid_indices:
            return ValidationResult(
                valid=False,
                error_code="INVALID_TIMESTAMPS",
                error_message=f"Found {len(invalid_indices)} NaN/Inf timestamps at indices {invalid_indices[:5]}",
            )
        
        return ValidationResult(valid=True)
    
    @staticmethod
    def validate_ordering(timestamps: List[float], strict_mode: bool) -> ValidationResult:
        """Validate timestamps are in ascending order.
        
        Args:
            timestamps: List of timestamps
            strict_mode: If True, reject unordered; if False, sort
        
        Returns:
            ValidationResult
        """
        out_of_order = [i for i in range(1, len(timestamps)) if timestamps[i] < timestamps[i - 1]]
        
        if out_of_order:
            if strict_mode:
                return ValidationResult(
                    valid=False,
                    error_code="UNORDERED_TIMESTAMPS",
                    error_message=f"Timestamps not in ascending order at indices {out_of_order[:5]}",
                )
            else:
                # Sanitize: sort
                sorted_timestamps = sorted(timestamps)
                
                logger.warning(
                    f"input_validator_sorted_timestamps: sorted {len(timestamps)} timestamps"
                )
                
                return ValidationResult(
                    valid=True,
                    sanitized_timestamps=sorted_timestamps,
                )
        
        return ValidationResult(valid=True)
    
    @staticmethod
    def check_duplicates(timestamps: List[float]) -> ValidationResult:
        """Check for duplicate timestamps.
        
        Args:
            timestamps: List of timestamps
        
        Returns:
            ValidationResult
        """
        seen = set()
        duplicates = []
        for i, t in enumerate(timestamps):
            if t in seen:
                duplicates.append(i)
            seen.add(t)
        
        if duplicates:
            return ValidationResult(
                valid=False,
                error_code="DUPLICATE_TIMESTAMPS",
                error_message=f"Found {len(duplicates)} duplicate timestamps at indices {duplicates[:5]}",
            )
        
        return ValidationResult(valid=True)
