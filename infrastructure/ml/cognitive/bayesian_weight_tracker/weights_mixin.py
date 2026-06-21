"""Weights mixin for BayesianWeightTracker."""
from __future__ import annotations
from typing import Dict, List, Optional
from .weight_calculator import compute_weights_from_accuracy
from .per_sensor_key import build_regime_key, build_fallback_key, should_use_per_sensor


class WeightsMixin:
    """Mixin providing weight retrieval."""

    def get_weights(
        self, regime: str, engine_names: List[str], series_id: Optional[str] = None
    ) -> Dict[str, float]:
        """Return per-sensor weights if enough history, else global fallback."""
        with self._lock:
            if should_use_per_sensor(series_id, self._accuracy, self._domain_namespace, regime):
                key = build_regime_key(self._domain_namespace, regime, series_id)
            else:
                key = build_fallback_key(self._domain_namespace, regime)
            n = len(engine_names)
            if n == 0:
                return {}
            redis_weights = self._redis.get_weights(key, engine_names, self._config.min_weight)
            if redis_weights:
                return redis_weights
            regime_data = self._accuracy.get(key, {})
            return compute_weights_from_accuracy(engine_names, regime_data, self._config.min_weight)

    def has_history(self, regime: str, series_id: Optional[str] = None) -> bool:
        """True if any accuracy data exists for this regime (per-sensor or global)."""
        with self._lock:
            key = build_regime_key(self._domain_namespace, regime, series_id)
            return bool(self._accuracy.get(key))
