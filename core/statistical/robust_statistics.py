"""Estadísticas robustas para datos no-normales.

Fallback cuando normalidad/estacionariedad fallan.
"""

from __future__ import annotations

import numpy as np

from core.parameters.numerical_constants import EPSILON


class RobustStatistics:
    """Estadísticas robustas a outliers y no-normalidad."""

    @staticmethod
    def median_absolute_deviation(data: np.ndarray) -> float:
        """
        MAD = median(|x - median(x)|)
        MAD ≈ 0.6745 * σ bajo normalidad
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return mad

    @staticmethod
    def robust_z_score(data: np.ndarray, value: float) -> float:
        """
        Z-score robusto usando MAD en lugar de std.
        z = (x - median) / (MAD / 0.6745)
        """
        median = np.median(data)
        mad = RobustStatistics.median_absolute_deviation(data)

        if mad < EPSILON.DIVISION:
            return 0.0

        # Convert MAD to σ-equivalent
        mad_sigma = mad / 0.6745
        return (value - median) / mad_sigma

    @staticmethod
    def robust_outlier_bounds(data: np.ndarray, k: float = 3.0) -> tuple[float, float]:
        """
        Bounds robustos: median ± k * MAD
        k=3.0 ≈ 3σ bajo normalidad
        """
        median = np.median(data)
        mad = RobustStatistics.median_absolute_deviation(data)
        mad_sigma = mad / 0.6745

        lower = median - k * mad_sigma
        upper = median + k * mad_sigma
        return lower, upper
