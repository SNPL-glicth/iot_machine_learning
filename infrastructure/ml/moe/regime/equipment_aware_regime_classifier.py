"""Equipment-aware regime classifier — replaces _classify_regime() logic."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from iot_machine_learning.domain.entities.series.structural_analysis import RegimeType

if TYPE_CHECKING:
    from iot_machine_learning.domain.entities.sensor_profile import SensorProfile


def classify_regime(
    noise_ratio: float,
    slope: float,
    std: float,
    mean: float = 0.0,
    hour_of_day: Optional[int] = None,
    sensor_profile: Optional["SensorProfile"] = None,
) -> RegimeType:
    """Clasifica régimen usando thresholds relativos al equipo si hay SensorProfile."""
    if sensor_profile is not None:
        rd = sensor_profile.relative_deviation(std)
        if rd > 3.0:
            return RegimeType.NOISY
        abs_slope = abs(slope)
        mean_ref = abs(mean) if abs(mean) > 1e-9 else 1.0
        slope_ratio = abs_slope / mean_ref
        if slope_ratio > 0.005 and abs_slope > 0.01:
            return RegimeType.TRENDING
        if rd > 1.0:
            return RegimeType.VOLATILE
        return RegimeType.STABLE

    if noise_ratio > 0.5:
        return RegimeType.NOISY
    abs_slope = abs(slope)
    mean_ref = abs(mean) if abs(mean) > 1e-9 else 1.0
    slope_ratio = abs_slope / mean_ref
    if slope_ratio > 0.005 and abs_slope > 0.01:
        return RegimeType.TRENDING
    if std > 0 and noise_ratio > 0.15:
        base = RegimeType.VOLATILE
        if hour_of_day is not None and (hour_of_day >= 22 or hour_of_day <= 6):
            return RegimeType.NOISY
        return base
    return RegimeType.STABLE
