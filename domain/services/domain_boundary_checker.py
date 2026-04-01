"""Domain boundary checker — validates data before pipeline entry.

Ensures data meets minimum quality requirements before processing.
Rejects data with insufficient points, zero variance, excessive missing
values, or infinite values.

Pure domain logic — stateless, no I/O, only numpy for computations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ...domain.entities.results.boundary_result import BoundaryResult

import numpy as np


class DomainBoundaryChecker:
    """Validates series data against domain entry criteria.

    Stateless domain service — determines if data is within acceptable
    domain for cognitive pipeline processing.

    Rejection rules (hard failures):
    - n_points < 3 → insufficient_data
    - all values identical (std == 0.0) → zero_variance
    - nan_ratio > 0.5 → excessive_missing
    - any inf values → infinite_values

    Warning rules (soft issues):
    - n_points < 10 → low_sample_count
    - nan_ratio > 0.2 → high_missing_ratio
    - noise_ratio > 0.8 → extreme_noise

    Quality score: 1.0 base, -0.3 per warning, 0.0 if rejected
    """

    # Thresholds
    MIN_POINTS_HARD: int = 3
    MIN_POINTS_WARN: int = 10

    NAN_RATIO_HARD: float = 0.5
    NAN_RATIO_WARN: float = 0.2

    NOISE_RATIO_WARN: float = 0.8

    # Quality score penalty per warning
    WARNING_PENALTY: float = 0.3

    def check(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
        noise_ratio: float = 0.0,
    ) -> "BoundaryResult":
        """Check if series data is within acceptable domain.

        Args:
            values: List of numeric values
            timestamps: Optional timestamps (for future extensions)
            noise_ratio: Pre-computed noise ratio if available

        Returns:
            BoundaryResult with validation outcome
        """
        from ...domain.entities.results.boundary_result import BoundaryResult

        warnings: List[str] = []
        rejection_reason: Optional[str] = None
        within_domain: bool = True

        # Convert to numpy array for efficient analysis
        arr = np.asarray(values, dtype=np.float64)
        n_points = len(arr)

        # Rule 1: Insufficient data points
        if n_points < self.MIN_POINTS_HARD:
            return BoundaryResult(
                within_domain=False,
                rejection_reason="insufficient_data",
                data_quality_score=0.0,
                warnings=[],
            )

        # Count NaNs
        n_nan = np.isnan(arr).sum()
        nan_ratio = n_nan / n_points if n_points > 0 else 0.0

        # Rule 2: Excessive missing values
        if nan_ratio > self.NAN_RATIO_HARD:
            return BoundaryResult(
                within_domain=False,
                rejection_reason="excessive_missing",
                data_quality_score=0.0,
                warnings=[],
            )

        # Rule 3: Infinite values
        if np.isinf(arr).any():
            return BoundaryResult(
                within_domain=False,
                rejection_reason="infinite_values",
                data_quality_score=0.0,
                warnings=[],
            )

        # Rule 4: Zero variance (all values identical)
        # Use nan-aware computation
        finite_mask = np.isfinite(arr)
        if finite_mask.sum() > 0:
            finite_values = arr[finite_mask]
            if finite_values.std() < 1e-12:
                return BoundaryResult(
                    within_domain=False,
                    rejection_reason="zero_variance",
                    data_quality_score=0.0,
                    warnings=[],
                )
        else:
            # All values are NaN
            return BoundaryResult(
                within_domain=False,
                rejection_reason="excessive_missing",
                data_quality_score=0.0,
                warnings=[],
            )

        # Passed hard checks — within domain, but check for warnings
        # Warning 1: Low sample count
        if n_points < self.MIN_POINTS_WARN:
            warnings.append("low_sample_count")

        # Warning 2: High missing ratio
        if nan_ratio > self.NAN_RATIO_WARN:
            warnings.append("high_missing_ratio")

        # Warning 3: Extreme noise
        if noise_ratio > self.NOISE_RATIO_WARN:
            warnings.append("extreme_noise")

        # Compute quality score
        # Base 1.0, -0.3 per warning, minimum 0.1
        score = max(0.1, 1.0 - (len(warnings) * self.WARNING_PENALTY))

        return BoundaryResult(
            within_domain=True,
            rejection_reason=None,
            data_quality_score=score,
            warnings=warnings,
        )
