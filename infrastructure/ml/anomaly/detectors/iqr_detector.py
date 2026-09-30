"""IQR sub-detector — detección por rango intercuartílico.

Evalúa si un valor está fuera de los Tukey fences sobre raw sensor values.
Fórmula: [Q1 - k×IQR, Q3 + k×IQR] donde k=IQR_FENCE_MULTIPLIER=1.5.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from core.drift.adaptive_strategy import AdaptiveScaler, HysteresisConfig, UnifiedAdaptiveConfig
from core.drift.drift_coupling import AdaptiveScalerDriftListener, DriftNotifier
from core.parameters.numerical_constants import EPSILON, STAT_THRESHOLDS
from core.statistical.robust_statistics import RobustStatistics
from core.statistical.statistical_validation import NormalityTestResult, NormalityValidator

from ..core.protocol import SubDetector
from ..scoring.training import TrainingStats, compute_training_stats

logger = logging.getLogger(__name__)


class IQRDetector(SubDetector):
    """Sub-detector basado en IQR (Tukey fences).

    Attributes:
        _adaptive: Activa adaptación de fences por volatilidad reciente.
        _rolling_iqr_history: Historial de IQR de últimas ventanas (max 100).
        _value_history: Últimos valores vistos para calcular IQR local.
    """

    def __init__(
        self,
        *,
        adaptive: bool = True,
        max_history: int = 100,
        min_history_entries: int = 5,
        normality_validator: NormalityValidator | None = None,
    ) -> None:
        self._adaptive = adaptive and UnifiedAdaptiveConfig.ADAPTIVE_ENABLED
        self._max_history, self._min_history_entries = max_history, min_history_entries
        self._normality_validator = normality_validator
        self._stats: TrainingStats | None = None
        self._normality_result: NormalityTestResult | None = None
        self._rolling_iqr_history: deque[float] = deque(maxlen=max_history)
        self._value_history: deque[float] = deque(maxlen=max_history)
        self.scaler: AdaptiveScaler | None = None
        if self._adaptive:
            self.scaler = AdaptiveScaler(
                scale_min=0.5, scale_max=2.0,
                hysteresis_config=HysteresisConfig(
                    threshold_increase=UnifiedAdaptiveConfig.HYSTERESIS_INCREASE,
                    threshold_decrease=UnifiedAdaptiveConfig.HYSTERESIS_DECREASE,
                    smooth_factor=UnifiedAdaptiveConfig.SMOOTH_FACTOR,
                    min_samples=self._min_history_entries,
                ),
            )
            DriftNotifier().subscribe(AdaptiveScalerDriftListener(self.scaler))

    @property
    def method_name(self) -> str:
        return "iqr"

    def train(self, values: list[float], **kwargs: object) -> None:
        self._stats = compute_training_stats(values)
        self._value_history.clear()
        self._rolling_iqr_history.clear()
        if self._stats and self._adaptive:
            self._rolling_iqr_history.append(self._stats.iqr)

        # Validate normality if validator provided
        if self._normality_validator is not None and len(values) >= self._normality_validator.min_samples:
            data_array = np.array(values)
            self._normality_result = self._normality_validator.validate(data_array)
            logger.info(
                "iqr_normality_validation",
                extra={
                    "is_normal": self._normality_result.is_normal,
                    "distribution_type": self._normality_result.distribution_type.value,
                    "recommendation": self._normality_result.recommendation,
                    "skewness": self._normality_result.skewness,
                },
            )

    @property
    def _effective_fence_multiplier(self) -> float:
        """Devuelve multiplicador de fences adaptativo o fijo (1.5)."""
        if not self._adaptive or not self.scaler:
            return STAT_THRESHOLDS.IQR_FENCE_MULTIPLIER

        mean_rolling_iqr = sum(self._rolling_iqr_history) / len(self._rolling_iqr_history)
        base_iqr = self._stats.iqr if self._stats and self._stats.iqr > 0 else mean_rolling_iqr

        if base_iqr < EPSILON.DIVISION:
            scale = 1.0
        else:
            scale = self.scaler.compute_scale(mean_rolling_iqr, base_iqr)

        return STAT_THRESHOLDS.IQR_FENCE_MULTIPLIER * scale

    def vote(self, value: float, **kwargs: object) -> float | None:
        if self._stats is None:
            return None

        iqr = self._stats.iqr
        use_mad = (
            self._normality_result is not None
            and self._normality_validator is not None
            and abs(self._normality_result.skewness) >= 0.5
            and len(self._value_history) >= self._normality_validator.min_samples
        )

        if use_mad and self._normality_result is not None:
            data_array = np.array(list(self._value_history) + [value])
            lower, upper = RobustStatistics.robust_outlier_bounds(data_array, k=3.0)
            logger.debug(
                "iqr_using_mad_bounds",
                extra={
                    "recommendation": self._normality_result.recommendation,
                    "skewness": self._normality_result.skewness,
                },
            )
        else:
            multiplier = self._effective_fence_multiplier
            lower = self._stats.q1 - multiplier * iqr
            upper = self._stats.q3 + multiplier * iqr

        if self._adaptive:
            self._value_history.append(value)
            if len(self._value_history) >= 4:
                sorted_vals = sorted(self._value_history)
                n = len(sorted_vals)
                q1_idx = int(n * 0.25)
                q3_idx = int(n * 0.75)
                local_q1 = sorted_vals[q1_idx]
                local_q3 = sorted_vals[q3_idx]
                local_iqr = local_q3 - local_q1
                if local_iqr > 0:
                    self._rolling_iqr_history.append(local_iqr)

        # Voto continuo: distancia normalizada a los fences
        if iqr <= 1e-9:
            distance = 3.0 if (value < lower or value > upper) else 0.0
        elif value < lower:
            distance = (lower - value) / iqr
        elif value > upper:
            distance = (value - upper) / iqr
        else:
            distance = 0.0

        # Mapear distancia a [0, 1] con saturación suave
        vote = min(1.0, distance / 3.0)  # >3 IQRs fuera → 1.0
        return float(vote)

    def raw_score(self, value: float, **kwargs: object) -> float | None:
        if self._stats is None:
            return None
        iqr = self._stats.iqr
        if iqr <= 1e-9:
            return 3.0 if (value < self._stats.q1 or value > self._stats.q3) else 0.0
        if value < self._stats.q1:
            return float((self._stats.q1 - value) / iqr)
        if value > self._stats.q3:
            return float((value - self._stats.q3) / iqr)
        return 0.0

    @property
    def is_trained(self) -> bool:
        return self._stats is not None
