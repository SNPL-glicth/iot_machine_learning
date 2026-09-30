"""Z-score sub-detector — detección por desviación estándar de magnitud.

Una responsabilidad: evaluar si un valor está lejos de la media histórica.
Sin sklearn, sin I/O.
"""
from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from core.drift.adaptive_strategy import AdaptiveScaler, HysteresisConfig, UnifiedAdaptiveConfig
from core.drift.drift_coupling import AdaptiveScalerDriftListener, DriftNotifier
from core.parameters.numerical_constants import EPSILON, STAT_THRESHOLDS
from core.statistical.robust_statistics import RobustStatistics
from core.statistical.statistical_validation import NormalityTestResult, NormalityValidator

from ..core.protocol import SubDetector
from ..scoring.functions import compute_z_score, compute_z_vote
from ..scoring.training import TrainingStats, compute_training_stats

logger = logging.getLogger(__name__)


class ZScoreDetector(SubDetector):
    """Sub-detector basado en Z-score de magnitud.

    Attributes:
        _lower: Z-score debajo del cual el voto es 0.
        _upper: Z-score encima del cual el voto es 1.
        _adaptive: Activa adaptación de thresholds por volatilidad reciente.
        _rolling_std_history: Historial de std de últimas ventanas (max 100).
        _value_history: Últimos valores vistos para calcular std local.
    """

    def __init__(
        self,
        lower: float | None = None,
        upper: float | None = None,
        *,
        adaptive: bool = True,
        max_history: int = 100,
        min_history_entries: int = 5,
        scale_min: float = 0.5,  # MATH-SEV-2
        scale_max: float = 1.5,  # Bound adaptive inflation to prevent blindness during drift
        max_lower: float = 3.5,  # Absolute lower bound
        max_upper: float = 4.5,  # Absolute upper bound
        normality_validator: NormalityValidator | None = None,
    ) -> None:
        self._base_lower = STAT_THRESHOLDS.Z_SCORE_LOWER if lower is None else lower
        self._base_upper = STAT_THRESHOLDS.Z_SCORE_UPPER if upper is None else upper
        self._adaptive = adaptive and UnifiedAdaptiveConfig.ADAPTIVE_ENABLED
        self._max_history, self._min_history_entries = max_history, min_history_entries
        self._scale_min, self._scale_max = scale_min, scale_max
        self._max_lower, self._max_upper = max_lower, max_upper
        self._normality_validator = normality_validator
        self._stats: TrainingStats | None = None
        self._normality_result: NormalityTestResult | None = None
        self._rolling_std_history: deque[float] = deque(maxlen=max_history)
        self._value_history: deque[float] = deque(maxlen=max_history)

        # Usar AdaptiveScaler unificado
        self.scaler: AdaptiveScaler | None = None
        if self._adaptive:
            self.scaler = AdaptiveScaler(
                scale_min=self._scale_min,
                scale_max=self._scale_max,
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
        return "z_score"

    def train(self, values: list[float], **kwargs: object) -> None:
        self._stats = compute_training_stats(values)
        self._value_history.clear()
        self._rolling_std_history.clear()
        if hasattr(self, "_ema_mean"):
            del self._ema_mean
        if self._stats and self._adaptive:
            self._rolling_std_history.append(self._stats.std)

        # Validate normality if validator provided
        if self._normality_validator is not None and len(values) >= self._normality_validator.min_samples:
            self._normality_result = self._normality_validator.validate(np.array(values))
            logger.info(
                "z_score_normality_validation",
                extra={
                    "is_normal": self._normality_result.is_normal,
                    "distribution_type": self._normality_result.distribution_type.value,
                    "recommendation": self._normality_result.recommendation,
                    "shapiro_p": self._normality_result.shapiro_p_value,
                },
            )

    @property
    def _effective_thresholds(self) -> tuple[float, float]:
        """Devuelve (lower, upper) efectivos, adaptativos o fijos."""
        if not self._adaptive or not self.scaler:
            return (min(self._base_lower, self._max_lower), min(self._base_upper, self._max_upper))

        mean_rolling_std = sum(self._rolling_std_history) / len(self._rolling_std_history)
        base_std = self._stats.std if self._stats and self._stats.std > 0 else mean_rolling_std
        if base_std < EPSILON.DIVISION:
            scale = 1.0
        else:
            scale = self.scaler.compute_scale(mean_rolling_std, base_std)

        lower_scaled = self._base_lower * scale
        upper_scaled = self._base_upper * scale
        return (min(lower_scaled, self._max_lower), min(upper_scaled, self._max_upper))

    def vote(self, value: float, **kwargs: object) -> float | None:
        if self._stats is None:
            return None

        # Use robust z-score if distribution is not normal
        if (
            self._normality_result is not None
            and not self._normality_result.is_normal
            and self._normality_validator is not None
            and len(self._value_history) >= self._normality_validator.min_samples
        ):
            data_array = np.array(list(self._value_history) + [value])
            z = RobustStatistics.robust_z_score(data_array, value)
            logger.debug(
                "z_score_using_robust_statistics",
                extra={
                    "recommendation": self._normality_result.recommendation,
                    "distribution_type": self._normality_result.distribution_type.value,
                },
            )
        else:
            # Dual z-score con switch condicional por drift
            z_global = compute_z_score(value, self._stats.mean, self._stats.std)
            if not hasattr(self, "_ema_mean"):
                self._ema_mean = self._stats.mean
            self._ema_mean = 0.1 * value + 0.9 * self._ema_mean
            z_local = compute_z_score(value, self._ema_mean, self._stats.std)
            drift_detected = abs(self._ema_mean - self._stats.mean) > 2.0 * self._stats.std
            z = z_local if drift_detected else z_global

        lower, upper = self._effective_thresholds
        result = compute_z_vote(z, lower, upper)

        if self._adaptive:
            self._value_history.append(value)
            if len(self._value_history) >= 3:
                local_mean = sum(self._value_history) / len(self._value_history)
                local_std = math.sqrt(
                    sum((v - local_mean) ** 2 for v in self._value_history) / len(self._value_history)
                )
                if local_std > 0:
                    self._rolling_std_history.append(local_std)

        return result

    def raw_score(self, value: float, **kwargs: object) -> float | None:
        if self._stats is None:
            return None
        return float(compute_z_score(value, self._stats.mean, self._stats.std))

    @property
    def is_trained(self) -> bool:
        return self._stats is not None

    @property
    def last_z_score(self) -> float:
        return 0.0
