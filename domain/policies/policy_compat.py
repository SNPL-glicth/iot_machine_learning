"""Backward compatibility helpers for ThresholdPolicy."""
from __future__ import annotations
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .threshold_policy import ThresholdPolicy


def default_policy() -> "ThresholdPolicy":
    """Return the default policy (same thresholds as legacy code)."""
    from .threshold_policy import ThresholdPolicy
    return ThresholdPolicy()


def from_score_thresholds(
    none_max: float = 0.3,
    low_max: float = 0.5,
    medium_max: float = 0.7,
    high_max: float = 0.9,
) -> "ThresholdPolicy":
    """Factory matching legacy AnomalySeverity.from_score() signature."""
    warnings.warn(
        "from_score_thresholds() is deprecated; use ThresholdPolicy() directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .threshold_policy import ThresholdPolicy
    return ThresholdPolicy(score_thresholds=(none_max, low_max, medium_max, high_max))
