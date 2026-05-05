"""Weights mixin for BayesianWeightTracker."""
from __future__ import annotations
from typing import Dict, List
from .weight_calculator import compute_weights_from_accuracy


class WeightsMixin:
    """Mixin providing weight retrieval."""

    def get_weights(self, regime: str, engine_names: List[str]) -> Dict[str, float]:
        """Compute regime-contextual weights."""
        namespaced_regime = f"{self._domain_namespace}:{regime}"
        n = len(engine_names)
        if n == 0:
            return {}
        redis_weights = self._redis.get_weights(namespaced_regime, engine_names, self._config.min_weight)
        if redis_weights:
            return redis_weights
        regime_data = self._accuracy.get(namespaced_regime, {})
        return compute_weights_from_accuracy(engine_names, regime_data, self._config.min_weight)

    def has_history(self, regime: str) -> bool:
        """True if any accuracy data exists for this regime."""
        namespaced_regime = f"{self._domain_namespace}:{regime}"
        return bool(self._accuracy.get(namespaced_regime))
