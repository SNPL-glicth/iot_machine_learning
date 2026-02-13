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

from collections import defaultdict
from typing import Dict, List, Optional


# Exponential smoothing factor for accuracy updates
_ALPHA: float = 0.15

# Minimum weight floor to prevent total suppression
_MIN_WEIGHT: float = 0.05


class PlasticityTracker:
    """Tracks per-regime, per-engine accuracy and computes adaptive weights.

    Attributes:
        _accuracy: Dict[regime][engine_name] → smoothed inverse error.
        _alpha: EMA smoothing factor.
        _min_weight: Minimum weight floor.
    """

    def __init__(
        self,
        alpha: float = _ALPHA,
        min_weight: float = _MIN_WEIGHT,
    ) -> None:
        self._accuracy: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._alpha = alpha
        self._min_weight = min_weight

    def update(
        self,
        regime: str,
        engine_name: str,
        prediction_error: float,
    ) -> None:
        """Record a prediction outcome for an engine in a regime.

        Args:
            regime: Current regime label.
            engine_name: Which engine made the prediction.
            prediction_error: |predicted - actual|.
        """
        inv_error = 1.0 / (abs(prediction_error) + 1e-9)
        prev = self._accuracy[regime].get(engine_name)

        if prev is None:
            self._accuracy[regime][engine_name] = inv_error
        else:
            self._accuracy[regime][engine_name] = (
                (1.0 - self._alpha) * prev + self._alpha * inv_error
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
