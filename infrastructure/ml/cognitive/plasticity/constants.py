"""Plasticity constants — centralized configuration.

All tunable parameters for the plasticity system.
NOTE: These values will be migrated to FeatureFlags in Phase 2.
"""

from __future__ import annotations

from typing import Dict

from iot_machine_learning.domain.entities.series.structural_analysis import RegimeType


# Exponential smoothing factor for accuracy updates
_ALPHA: float = 0.15

# R-2: Regime-specific alpha values for adaptive learning rates
# VOLATILE uses higher alpha (0.25) for faster adaptation to industrial changes
# STABLE uses lower alpha (0.10) to avoid overfitting to noise
_REGIME_ALPHA: Dict[RegimeType, float] = {
    RegimeType.STABLE: 0.10,
    RegimeType.TRENDING: 0.20,
    RegimeType.VOLATILE: 0.25,
    RegimeType.NOISY: 0.08,
    RegimeType.TRANSITIONAL: 0.18,
}

# Minimum weight floor to prevent total suppression
_MIN_WEIGHT: float = 0.05

# Maximum regimes to track before LRU eviction
_MAX_REGIMES: int = 10

# Persist state every N updates (batching for performance)
_PERSIST_EVERY_N_UPDATES: int = 10

# Redis cache TTL for weights (seconds)
_REDIS_CACHE_TTL_SECONDS: float = 60.0

# Default threshold for immediate persistence (accuracy change > threshold)
_IMMEDIATE_PERSIST_THRESHOLD: float = 0.15


class PlasticityConfig:
    """Configuration container for plasticity behavior.

    Allows runtime configuration without modifying constants.
    All parameters injectable for testability (ISO 25010).
    """

    def __init__(
        self,
        alpha: float = _ALPHA,
        min_weight: float = _MIN_WEIGHT,
        max_regimes: int = _MAX_REGIMES,
        regime_ttl_seconds: float = 86400.0,
        immediate_persist_threshold: float = _IMMEDIATE_PERSIST_THRESHOLD,
    ) -> None:
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_regimes = max(1, max_regimes)
        self.regime_ttl_seconds = regime_ttl_seconds
        self.immediate_persist_threshold = immediate_persist_threshold

    def get_regime_alpha(self, regime: str | RegimeType) -> float:
        """Get alpha for a specific regime.

        Args:
            regime: Regime as string or RegimeType enum

        Returns:
            Alpha value for the regime, or default alpha if not found
        """
        if isinstance(regime, RegimeType):
            return _REGIME_ALPHA.get(regime, self.alpha)
        # Handle string input (backward compatibility)
        try:
            regime_enum = RegimeType(regime.upper())
            return _REGIME_ALPHA.get(regime_enum, self.alpha)
        except ValueError:
            return self.alpha
