"""Configuration for Bayesian weight tracking.

Consolidates BayesianWeightConfig, WeightTrackerConfig, and module-level
constants into a single source of truth.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict

from iot_machine_learning.domain.entities.series.structural_analysis import RegimeType


# Module-level constants (used by other modules directly)
_ALPHA: float = 0.15
_PERSIST_EVERY_N_UPDATES: int = 10
_REDIS_CACHE_TTL_SECONDS: float = 60.0
_IMMEDIATE_PERSIST_THRESHOLD: float = 0.15
_SIGMA2_OBS_DEFAULT: float = 1.0
_SIGMA2_OBS_MIN: float = 0.01
_VARIANCE_WINDOW: int = 20
_VARIANCE_MIN_SAMPLES: int = 5
_MIN_WEIGHT: float = 0.05
_MAX_REGIMES: int = 10

_REGIME_ALPHA: Dict[RegimeType, float] = {
    RegimeType.STABLE: 0.10,
    RegimeType.TRENDING: 0.20,
    RegimeType.VOLATILE: 0.25,
    RegimeType.NOISY: 0.08,
    RegimeType.TRANSITIONAL: 0.18,
}


@dataclass(frozen=True)
class BayesianWeightConfig:
    alpha: float = 0.15
    min_weight: float = 0.05
    immediate_persist_threshold: float = 0.15
    drift_decay_factor: float = 0.5
    drift_variance_expansion: float = 2.0
    sigma2_obs_default: float = 1.0
    sigma2_obs_min: float = 0.01
    prior_variance_scale: float = 1.0
    variance_window: int = 20
    variance_min_samples: int = 5
    convergence_window: int = 10
    convergence_cv_threshold: float = 0.05
    weight_history_maxlen: int = 50
    max_regimes: int = 10
    regime_ttl_seconds: float = 86400.0
    regularization_strength: float = 0.01

    def validate(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0), got {self.alpha}")
        if not (0.0 < self.min_weight < 1.0):
            raise ValueError(f"min_weight must be in (0.0, 1.0), got {self.min_weight}")
        if self.sigma2_obs_min >= self.sigma2_obs_default:
            raise ValueError(
                f"sigma2_obs_min ({self.sigma2_obs_min}) must be "
                f"< sigma2_obs_default ({self.sigma2_obs_default})"
            )
        if self.variance_min_samples > self.variance_window:
            raise ValueError(
                f"variance_min_samples ({self.variance_min_samples}) must be "
                f"<= variance_window ({self.variance_window})"
            )
        if self.convergence_window > self.weight_history_maxlen:
            raise ValueError(
                f"convergence_window ({self.convergence_window}) must be "
                f"<= weight_history_maxlen ({self.weight_history_maxlen})"
            )
        if self.drift_variance_expansion <= 1.0:
            raise ValueError(
                f"drift_variance_expansion must be > 1.0, got {self.drift_variance_expansion}"
            )

    def get_regime_alpha(self, regime: str | RegimeType) -> float:
        if isinstance(regime, RegimeType):
            return _REGIME_ALPHA.get(regime, self.alpha)
        try:
            return _REGIME_ALPHA.get(RegimeType(regime.upper()), self.alpha)
        except (ValueError, AttributeError):
            return self.alpha


class WeightTrackerConfig:
    """Deprecated — use BayesianWeightConfig instead."""

    def __init__(
        self,
        alpha: float = _ALPHA,
        min_weight: float = _MIN_WEIGHT,
        max_regimes: int = _MAX_REGIMES,
        regime_ttl_seconds: float = 86400.0,
        immediate_persist_threshold: float = _IMMEDIATE_PERSIST_THRESHOLD,
        sigma2_obs_default: float = _SIGMA2_OBS_DEFAULT,
        sigma2_obs_min: float = _SIGMA2_OBS_MIN,
        variance_window: int = _VARIANCE_WINDOW,
        variance_min_samples: int = _VARIANCE_MIN_SAMPLES,
    ) -> None:
        warnings.warn(
            "WeightTrackerConfig is deprecated. Use BayesianWeightConfig instead.",
            DeprecationWarning, stacklevel=2,
        )
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_regimes = max(1, max_regimes)
        self.regime_ttl_seconds = regime_ttl_seconds
        self.immediate_persist_threshold = immediate_persist_threshold
        self.sigma2_obs_default = sigma2_obs_default
        self.sigma2_obs_min = sigma2_obs_min
        self.variance_window = variance_window
        self.variance_min_samples = variance_min_samples

    def get_regime_alpha(self, regime: str | RegimeType) -> float:
        if isinstance(regime, RegimeType):
            return _REGIME_ALPHA.get(regime, self.alpha)
        try:
            return _REGIME_ALPHA.get(RegimeType(regime.upper()), self.alpha)
        except ValueError:
            return self.alpha
