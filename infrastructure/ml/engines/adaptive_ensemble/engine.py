"""Adaptive ensemble engine (P6).

Selects the most appropriate sub-engine based on a lightweight
signal-regime heuristic.  Contains Taylor, Kalman, Statistical and
Baseline as delegates; avoids the full cognitive orchestrator overhead.

Regime map:
    - "noisy"        → Kalman (separates noise from velocity)
    - "volatile"     → Kalman (covariance-based confidence)
    - "trending"     → Taylor (captures acceleration)
    - "stable"       → Baseline (simple, fast)
    - "transitional" → Statistical (EMA/Holt adaptive smoothing)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

from iot_machine_learning.infrastructure.ml.engines.core.factory import (
    BaselineMovingAverageEngine,
    PredictionEngine,
    PredictionResult,
    register_engine,
)
from iot_machine_learning.infrastructure.ml.engines.kalman import (
    KalmanPredictionEngine,
)
from iot_machine_learning.infrastructure.ml.engines.statistical import (
    StatisticalPredictionEngine,
)
from iot_machine_learning.infrastructure.ml.engines.taylor import (
    TaylorPredictionEngine,
)

logger = logging.getLogger(__name__)


@register_engine("adaptive_ensemble")
class AdaptiveEnsembleEngine(PredictionEngine):
    """Meta-engine that routes to the best sub-engine per regime.

    P6: Optional lightweight alternative to the full cognitive orchestrator.
    """

    def __init__(
        self,
        noise_threshold: float = 0.3,
        trend_threshold: float = 0.02,
        volatile_threshold: float = 0.5,
        *,
        taylor_kwargs: Optional[Dict] = None,
        kalman_kwargs: Optional[Dict] = None,
        statistical_kwargs: Optional[Dict] = None,
        baseline_kwargs: Optional[Dict] = None,
    ) -> None:
        self._noise_threshold = noise_threshold
        self._trend_threshold = trend_threshold
        self._volatile_threshold = volatile_threshold

        self._taylor = TaylorPredictionEngine(**(taylor_kwargs or {}))
        self._kalman = KalmanPredictionEngine(**(kalman_kwargs or {}))
        self._statistical = StatisticalPredictionEngine(**(statistical_kwargs or {}))
        self._baseline = BaselineMovingAverageEngine(**(baseline_kwargs or {}))

        self._engines = {
            "taylor": self._taylor,
            "kalman": self._kalman,
            "statistical": self._statistical,
            "baseline": self._baseline,
        }
        self._last_selected: Optional[str] = None

    @property
    def name(self) -> str:
        return "adaptive_ensemble"

    def can_handle(self, n_points: int) -> bool:
        # OR of sub-engine capabilities
        return any(e.can_handle(n_points) for e in self._engines.values())

    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        """Route to the best sub-engine based on signal regime.

        Regime heuristic: noisy → Statistical, trending → Taylor, stable → Baseline.
        If the preferred engine cannot handle the data, falls back to the
        next capable engine in the chain.

        Args:
            values: Time-series window.
            timestamps: Optional timestamps forwarded to the selected engine.

        Returns:
            PredictionResult enriched with ``ensemble_regime`` and
            ``ensemble_selected`` metadata.

        Raises:
            ValueError: If values is empty.
        """
        if not values:
            raise ValueError("values cannot be empty")

        regime = self._detect_regime(values)
        selected = self._select_engine(regime, values)
        self._last_selected = selected

        result = self._engines[selected].predict(values, timestamps)

        # Wrap metadata with ensemble routing info
        metadata = dict(result.metadata)
        metadata["ensemble_regime"] = regime
        metadata["ensemble_selected"] = selected

        return PredictionResult(
            predicted_value=result.predicted_value,
            confidence=result.confidence,
            trend=result.trend,
            metadata=metadata,
        )

    def supports_uncertainty(self) -> bool:
        return False

    def record_actual(self, predicted: float, actual: float) -> None:
        """Propagate actual value to ALL sub-engines for continuous learning."""
        for name, engine in self._engines.items():
            try:
                engine.record_actual(predicted, actual)
            except Exception:
                logger.debug("ensemble_record_actual_failed", extra={"engine": name})

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _detect_regime(self, values: List[float]) -> str:
        """Lightweight heuristic: noisy / volatile / trending / transitional / stable."""
        n = len(values)
        if n < 2:
            return "stable"

        mean_abs = sum(abs(v) for v in values) / n
        eps = 1e-12
        mean_val = sum(values) / n
        variance = sum((v - mean_val) ** 2 for v in values) / n
        std = math.sqrt(variance)
        noise_ratio = std / (mean_abs + eps)

        # Trend slope on last 10 points
        trend_window = values[-10:]
        slope = self._linear_slope(trend_window) if len(trend_window) >= 2 else 0.0
        slope_rel = abs(slope) / (max(mean_abs, 1.0) + eps)

        # Volatility: high noise + significant trend movement
        if noise_ratio > self._volatile_threshold:
            return "volatile"
        if noise_ratio > self._noise_threshold:
            return "noisy"
        if slope_rel > self._trend_threshold:
            return "trending"
        if slope_rel > self._trend_threshold * 0.3:
            return "transitional"
        return "stable"

    def _select_engine(self, regime: str, values: List[float]) -> str:
        """Map regime to engine name with fallback chain.

        Evidence-based mapping from benchmark (scripts/benchmark_kalman_vs_ensemble.py):
        - stable       → Baseline (simple, fast, optimal for flat signals)
        - noisy        → Baseline (moving average beats Kalman on pure noise)
        - trending     → Taylor (derivatives capture acceleration)
        - volatile     → Kalman (separates real dynamics from noise)
        - transitional → Statistical (EMA/Holt smooths regime changes)
        """
        mapping = {
            "stable": "baseline",
            "noisy": "baseline",
            "trending": "taylor",
            "volatile": "kalman",
            "transitional": "statistical",
        }
        preferred = mapping.get(regime, "baseline")

        # Fallback if preferred engine cannot handle the data
        n = len(values)
        if self._engines[preferred].can_handle(n):
            logger.info(
                "ensemble_engine_selected",
                extra={"regime": regime, "selected": preferred, "points": n},
            )
            return preferred

        # Try remaining engines in order of generality
        for candidate in ("baseline", "statistical", "kalman", "taylor"):
            if self._engines[candidate].can_handle(n):
                return candidate

        # Should never reach here because can_handle is OR of all
        return "baseline"

    @staticmethod
    def _linear_slope(seq: List[float]) -> float:
        n = len(seq)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(seq) / n
        num = sum((i - x_mean) * (seq[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den != 0 else 0.0
