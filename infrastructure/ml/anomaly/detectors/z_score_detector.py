"""Z-score sub-detector — detección por desviación estándar de magnitud.

Una responsabilidad: evaluar si un valor está lejos de la media histórica.
Sin sklearn, sin I/O.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import List, Optional

import numpy as np

from core.parameters.numerical_constants import STAT_THRESHOLDS
from core.drift.adaptive_strategy import (
    AdaptiveScaler,
    UnifiedAdaptiveConfig,
    HysteresisConfig,
)
from core.drift.drift_coupling import AdaptiveScalerDriftListener, DriftNotifier
from core.statistical.statistical_validation import NormalityValidator, NormalityTestResult
from core.statistical.robust_statistics import RobustStatistics

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
        lower: float = None,
        upper: float = None,
        *,
        adaptive: bool = True,
        max_history: int = 100,
        min_history_entries: int = 5,
        scale_min: float = 0.5,  # MATH-SEV-2
        scale_max: float = 5.0,  # FASE-23: UnifiedAdaptiveConfig.SCALE_MAX (fuente de verdad)
        # Hysteresis + smoothing previenen oscilación sin necesitar límite bajo
        max_lower: float = 10.0,  # SEVERO-2: absolute bounds
        max_upper: float = 15.0,  # SEVERO-2: absolute bounds
        normality_validator: Optional[NormalityValidator] = None,
    ) -> None:
        # Use STAT_THRESHOLDS defaults if not provided
        if lower is None:
            lower = STAT_THRESHOLDS.Z_SCORE_LOWER
        if upper is None:
            upper = STAT_THRESHOLDS.Z_SCORE_UPPER
        
        self._base_lower = lower
        self._base_upper = upper
        self._adaptive = adaptive and UnifiedAdaptiveConfig.ADAPTIVE_ENABLED
        self._max_history = max_history
        self._min_history_entries = min_history_entries
        self._scale_min = scale_min  # MATH-SEV-2
        self._scale_max = scale_max  # MATH-SEV-2
        self._max_lower = max_lower  # SEVERO-2
        self._max_upper = max_upper  # SEVERO-2
        self._normality_validator = normality_validator
        self._stats: Optional[TrainingStats] = None
        self._normality_result: Optional[NormalityTestResult] = None
        self._rolling_std_history: deque[float] = deque(maxlen=max_history)
        self._value_history: deque[float] = deque(maxlen=max_history)
        
        # NUEVO: Usar AdaptiveScaler unificado
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
            # Suscribir a drift events
            drift_notifier = DriftNotifier()
            drift_notifier.subscribe(AdaptiveScalerDriftListener(self.scaler))
        else:
            self.scaler = None

    @property
    def method_name(self) -> str:
        return "z_score"

    def train(self, values: List[float], **kwargs: object) -> None:
        self._stats = compute_training_stats(values)
        if self._stats and self._adaptive:
            self._rolling_std_history.append(self._stats.std)

        # Validate normality if validator provided
        if self._normality_validator is not None and len(values) >= self._normality_validator.min_samples:
            data_array = np.array(values)
            self._normality_result = self._normality_validator.validate(data_array)
            logger.info(
                "z_score_normality_validation",
                extra={
                    "is_normal": self._normality_result.is_normal,
                    "distribution_type": self._normality_result.distribution_type.value,
                    "recommendation": self._normality_result.recommendation,
                    "shapiro_p": self._normality_result.shapiro_p_value,
                    "skewness": self._normality_result.skewness,
                    "kurtosis": self._normality_result.kurtosis,
                },
            )

    @property
    def _effective_thresholds(self) -> tuple[float, float]:
        """Devuelve (lower, upper) efectivos, adaptativos o fijos.
        
        MATH-SEV-2: Scale is clamped to [scale_min, scale_max] to prevent
        detector from becoming insensitive.
        
        SEVERO-2: Absolute bounds (max_lower, max_upper) prevent extreme thresholds.
        """
        if not self._adaptive or not self.scaler:
            return (
                min(self._base_lower, self._max_lower),
                min(self._base_upper, self._max_upper),
            )
        
        mean_rolling_std = sum(self._rolling_std_history) / len(self._rolling_std_history)
        base_std = self._stats.std if self._stats and self._stats.std > 0 else mean_rolling_std
        
        if base_std < EPSILON.DIVISION:
            scale = 1.0
        else:
            scale = self.scaler.compute_scale(mean_rolling_std, base_std)
        
        lower_scaled = self._base_lower * scale
        upper_scaled = self._base_upper * scale
        
        return (
            min(lower_scaled, self._max_lower),
            min(upper_scaled, self._max_upper),
        )

    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        if self._stats is None:
            return None

        # Use robust z-score if distribution is not normal
        use_robust = (
            self._normality_result is not None
            and not self._normality_result.is_normal
        )

        if use_robust and len(self._value_history) >= self._normality_validator.min_samples:
            # Use robust statistics (MAD-based z-score)
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
            # Use standard z-score
            z = compute_z_score(value, self._stats.mean, self._stats.std)

        lower, upper = self._effective_thresholds
        result = compute_z_vote(z, lower, upper)

        if self._adaptive:
            self._value_history.append(value)
            if len(self._value_history) >= 3:
                local_mean = sum(self._value_history) / len(self._value_history)
                local_std = math.sqrt(
                    sum((v - local_mean) ** 2 for v in self._value_history)
                    / len(self._value_history)
                )
                if local_std > 0:
                    self._rolling_std_history.append(local_std)

        return result

    @property
    def is_trained(self) -> bool:
        return self._stats is not None

    @property
    def last_z_score(self) -> float:
        """Último Z-score calculado (para narración). Recalcula bajo demanda."""
        return 0.0
