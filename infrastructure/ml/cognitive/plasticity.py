"""Regime-contextual weight learning (plasticity).

Analogous to synaptic plasticity: if an engine consistently performs
well in a specific regime, its weight *in that regime* increases.

The tracker maintains a per-regime, per-engine accuracy history
and computes adaptive weights that reflect historical performance
within the current signal regime.

Design:
    - In-memory only (no persistence) — resets on restart.
    - Exponential moving average of inverse error for smoothness.
    - Falls back to uniform weights when no history exists.

Pure logic — no I/O, no logging.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional


# Exponential smoothing factor for accuracy updates
_ALPHA: float = 0.15

# Minimum weight floor to prevent total suppression
_MIN_WEIGHT: float = 0.05

# Maximum regimes to track before LRU eviction
_MAX_REGIMES: int = 10


class PlasticityTracker:
    """Tracks per-regime, per-engine accuracy and computes adaptive weights.

    Attributes:
        _accuracy: Dict[regime][engine_name] → smoothed inverse error.
        _alpha: EMA smoothing factor.
        _min_weight: Minimum weight floor.
        _max_regimes: Maximum regimes before LRU eviction.
        _regime_last_access: Dict[regime] → monotonic timestamp for LRU.
    """

    def __init__(
        self,
        alpha: float = _ALPHA,
        min_weight: float = _MIN_WEIGHT,
        max_regimes: int = _MAX_REGIMES,
        regime_ttl_seconds: float = 86400.0,
    ) -> None:
        self._accuracy: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._alpha = alpha
        self._min_weight = min_weight
        self._max_regimes = max(1, max_regimes)
        self._regime_ttl_seconds = regime_ttl_seconds
        self._regime_last_access: Dict[str, float] = {}
        self._regime_last_update: Dict[str, float] = {}

    def update(
        self,
        regime: str,
        engine_name: str,
        prediction_error: float,
        alpha: Optional[float] = None,
    ) -> None:
        """Record a prediction outcome for an engine in a regime.

        Args:
            regime: Current regime label.
            engine_name: Which engine made the prediction.
            prediction_error: |predicted - actual|.
            alpha: Override EMA smoothing factor for this call only.
                If None, uses self._alpha (default behaviour).
        """
        # Evict LRU regime if at capacity
        if regime not in self._accuracy and len(self._accuracy) >= self._max_regimes:
            coldest = min(self._regime_last_access, key=self._regime_last_access.get)
            del self._accuracy[coldest]
            del self._regime_last_access[coldest]
            self._regime_last_update.pop(coldest, None)
        
        now = time.monotonic()
        
        # TTL decay: if regime hasn't been updated in TTL period, decay all weights by 50%
        if regime in self._regime_last_update:
            elapsed = now - self._regime_last_update[regime]
            if elapsed > self._regime_ttl_seconds:
                for eng in self._accuracy[regime]:
                    self._accuracy[regime][eng] *= 0.5
        
        # Update access and update times
        self._regime_last_access[regime] = now
        self._regime_last_update[regime] = now
        
        effective_alpha = alpha if alpha is not None else self._alpha
        inv_error = 1.0 / (abs(prediction_error) + 1e-9)
        prev = self._accuracy[regime].get(engine_name)

        if prev is None:
            self._accuracy[regime][engine_name] = inv_error
        else:
            self._accuracy[regime][engine_name] = (
                (1.0 - effective_alpha) * prev + effective_alpha * inv_error
            )

    def get_weights(
        self,
        regime: str,
        engine_names: List[str],
    ) -> Dict[str, float]:
        """Compute regime-contextual weights from accumulated accuracy.

        Args:
            regime: Current regime label.
            engine_names: List of engine names to weight.

        Returns:
            Dict[engine_name → weight], normalized to sum to 1.0.
            Falls back to uniform weights if no history for this regime.
        """
        n = len(engine_names)
        if n == 0:
            return {}

        regime_data = self._accuracy.get(regime, {})
        if not regime_data:
            uniform = 1.0 / n
            return {name: uniform for name in engine_names}
        
        # Update access time for LRU tracking
        if regime in self._regime_last_access:
            self._regime_last_access[regime] = time.monotonic()

        raw: Dict[str, float] = {}
        for name in engine_names:
            raw[name] = max(
                self._min_weight,
                regime_data.get(name, self._min_weight),
            )

        total = sum(raw.values())
        if total < 1e-12:
            uniform = 1.0 / n
            return {name: uniform for name in engine_names}

        return {name: w / total for name, w in raw.items()}

    def has_history(self, regime: str) -> bool:
        """True if any accuracy data exists for this regime."""
        return bool(self._accuracy.get(regime))

    def reset(self, regime: Optional[str] = None) -> None:
        """Clear accumulated accuracy data.

        Args:
            regime: If provided, clear only that regime.
                If None, clear all regimes.
        """
        if regime is not None:
            self._accuracy.pop(regime, None)
        else:
            self._accuracy.clear()
