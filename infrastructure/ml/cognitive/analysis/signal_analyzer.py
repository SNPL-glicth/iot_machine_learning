"""Signal analysis — structural feature extraction from raw time series.

Computes a ``StructuralAnalysis`` *before* any engine runs, so all engines
share the same characterization of the input signal.

Features extracted:
    - mean, std, noise_ratio (coefficient of variation)
    - slope (OLS linear trend over last N points)
    - curvature (second derivative at last point)
    - stability, accel_variance (second-order dynamics)
    - trend_strength (normalized slope)
    - regime label (delegates to domain's canonical classifier)
    - dt (median-based time step estimation)

Pure functions — no I/O, no state, no logging.

.. versionchanged:: 2.0
    ``analyze()`` now returns ``StructuralAnalysis`` (domain) instead of
    ``SignalProfile`` (infra).  ``SignalProfile`` is deprecated.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, Union

from core.parameters.numerical_constants import EPSILON
from domain.entities.series.structural_analysis import (
    RegimeType,
    StructuralAnalysis,
    _classify_regime as _domain_classify_regime,
)

from .types import SignalProfile


def _compute_mean_std(values: List[float]) -> tuple[float, float]:
    """Population mean and std."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / n
    return mu, math.sqrt(var)


def _compute_slope(values: List[float], dt: float) -> float:
    """OLS linear slope over the full window.

    Fits y = a + b·t via closed-form OLS:
        b = Σ(t_i - t̄)(y_i - ȳ) / Σ(t_i - t̄)²
    """
    n = len(values)
    if n < 2:
        return 0.0
    t_mean = (n - 1) * dt / 2.0
    y_mean = sum(values) / n
    num = 0.0
    den = 0.0
    for i, y in enumerate(values):
        t_i = i * dt
        diff_t = t_i - t_mean
        num += diff_t * (y - y_mean)
        den += diff_t * diff_t
    if abs(den) < 1e-15:
        return 0.0
    return num / den


def _compute_curvature(values: List[float], dt: float) -> float:
    """Second derivative at the last point via backward differences."""
    if len(values) < 3:
        return 0.0
    dt2 = dt * dt
    if dt2 < 1e-15:
        return 0.0
    return (values[-1] - 2.0 * values[-2] + values[-3]) / dt2


def _classify_regime(
    noise_ratio: float,
    slope: float,
    std: float,
    mean: float = 0.0,
) -> str:
    """Delegate to domain's canonical regime classification.

    Returns a plain ``str`` for backward compatibility with
    ``SignalProfile.regime``.  The single source of truth is
    ``domain.entities.series.structural_analysis._classify_regime``.
    """
    return _domain_classify_regime(noise_ratio, slope, std, mean).value


def _estimate_dt(timestamps: Optional[List[float]]) -> float:
    """Median-based Δt estimation."""
    if timestamps is None or len(timestamps) < 2:
        return 1.0
    diffs = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
        if timestamps[i] > timestamps[i - 1]
    ]
    if not diffs:
        return 1.0
    diffs.sort()
    mid = len(diffs) // 2
    median = diffs[mid] if len(diffs) % 2 else (diffs[mid - 1] + diffs[mid]) / 2
    return max(median, EPSILON.CONFIDENCE)


def _compute_accel_variance(values: List[float], dt: float) -> float:
    """Population variance of the second derivative across the window."""
    n = len(values)
    if n < 4:
        return 0.0
    dt_sq = dt * dt
    accels: List[float] = []
    for i in range(2, n):
        accel = (values[i] - 2.0 * values[i - 1] + values[i - 2]) / dt_sq
        accels.append(accel)
    if len(accels) < 2:
        return 0.0
    mean_a = sum(accels) / len(accels)
    return sum((a - mean_a) ** 2 for a in accels) / len(accels)


def _compute_stability(accel_variance: float, f_t: float) -> float:
    """Normalize accel_variance to [0, 1]."""
    normalizer = abs(f_t) if abs(f_t) > EPSILON.CONFIDENCE else 1.0
    return min(accel_variance / normalizer, 1.0)


_SIGNAL_CACHE_TTL = 1.0


class SignalAnalyzer:
    """Extracts a ``StructuralAnalysis`` from raw values + timestamps.

    Includes a per‑sensor TTL cache (1 s granularity) to avoid
    recomputation on repeated calls for the same sensor within a
    sub‑second window.
    """

    def __init__(self):
        self._cache: Dict[Tuple[str, int], Tuple[float, StructuralAnalysis]] = {}

    def _cache_key(self, sensor_id: str) -> Tuple[str, int]:
        bucket = int(time.time())
        return (sensor_id, bucket)

    def analyze(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
        sensor_id: Optional[str] = None,
    ) -> StructuralAnalysis:
        """Compute structural features of the signal.

        Args:
            values: Time series (most recent last).
            timestamps: Optional Unix timestamps.
            sensor_id: Optional sensor identifier for caching.

        Returns:
            ``StructuralAnalysis`` with all structural features.
        """
        if not values:
            return StructuralAnalysis()

        if sensor_id is not None:
            key = self._cache_key(sensor_id)
            cached = self._cache.get(key)
            if cached is not None:
                return cached[1]

        result = self._compute(values, timestamps)

        if sensor_id is not None:
            key = self._cache_key(sensor_id)
            self._cache[key] = (time.time(), result)

        return result

    def _compute(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> StructuralAnalysis:
        dt = _estimate_dt(timestamps)
        mu, sigma = _compute_mean_std(values)
        noise_ratio = sigma / abs(mu) if abs(mu) > EPSILON.DIVISION else 0.0
        slope = _compute_slope(values, dt)
        curvature = _compute_curvature(values, dt)
        accel_variance = _compute_accel_variance(values, dt)
        stability = _compute_stability(accel_variance, values[-1])
        mean_ref = abs(mu) if abs(mu) > EPSILON.DIVISION else 1.0
        trend_strength = abs(slope) / mean_ref
        regime = _domain_classify_regime(noise_ratio, slope, sigma, mu)

        return StructuralAnalysis(
            n_points=len(values),
            mean=mu,
            std=sigma,
            noise_ratio=noise_ratio,
            slope=slope,
            curvature=curvature,
            stability=stability,
            accel_variance=accel_variance,
            trend_strength=trend_strength,
            regime=regime,
        )
