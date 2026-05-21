"""IQR sub-detector — detección por rango intercuartílico.

Una responsabilidad: evaluar si un valor está fuera de los Tukey fences.
Sin sklearn, sin I/O.

IQR Outlier Detector — opera sobre RAW SENSOR VALUES.

Fórmula: [Q1 - k×IQR, Q3 + k×IQR] donde k=IQR_FENCE_MULTIPLIER=1.5

Diferencia con Hampel Filter:
- IQR opera en datos CRUDOS de sensor (pre-predicción)
- IQR usa cuartiles: sensible a distribuciones asimétricas
- Para distribuciones skewed, IQR_FENCE_MULTIPLIER=1.5 puede
  producir falsos positivos en la cola larga
- Phase 5 agrega validación de simetría: si |skewness| > 0.5,
  usar MAD bounds (robust_statistics.py) en lugar de IQR

Bajo normalidad: IQR ≈ 1.35σ → 1.5×IQR ≈ 2.0σ (95.4% confidence)
Comparar con Z_SCORE_LOWER=2.0σ: equivalentes bajo normalidad.

DIFERENCIA INTENCIONAL MAD k=3.0 vs IQR k=1.5:
- IQR k=1.5 (Tukey, 1977): Tukey fences estándar → ~2.0σ bajo normalidad
- MAD k=3.0 (Hampel, 1974): Hampel identifier → ~3.0σ bajo normalidad
- Al cambiar a MAD para datos skewed, se SUBE sensibilidad a 3.0σ porque:
  * Distribuciones asimétricas tienen colas más largas
  * k=2.0 en MAD generaría excesivos falsos positivos en cola larga
  * k=3.0 mantiene balance entre robustez y sensibilidad
- Referencias: Hampel (1974) "The influence curve and its role in robust estimation",
  Tukey (1977) "Exploratory Data Analysis"
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
from ..scoring.functions import compute_iqr_vote
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
        normality_validator: Optional[NormalityValidator] = None,
    ) -> None:
        self._adaptive = adaptive and UnifiedAdaptiveConfig.ADAPTIVE_ENABLED
        self._max_history = max_history
        self._min_history_entries = min_history_entries
        self._normality_validator = normality_validator
        self._stats: Optional[TrainingStats] = None
        self._normality_result: Optional[NormalityTestResult] = None
        self._rolling_iqr_history: deque[float] = deque(maxlen=max_history)
        self._value_history: deque[float] = deque(maxlen=max_history)
        
        # NUEVO: Usar AdaptiveScaler unificado
        if self._adaptive:
            self.scaler = AdaptiveScaler(
                scale_min=0.5,
                scale_max=5.0,  # FASE-23: UnifiedAdaptiveConfig.SCALE_MAX (fuente de verdad)
                # Hysteresis + smoothing previenen oscilación sin necesitar límite bajo
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
        return "iqr"

    def train(self, values: List[float], **kwargs: object) -> None:
        self._stats = compute_training_stats(values)
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

    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        if self._stats is None:
            return None

        # Use MAD bounds if distribution is skewed (|skewness| >= 0.5)
        use_mad = (
            self._normality_result is not None
            and abs(self._normality_result.skewness) >= 0.5
        )

        if use_mad and len(self._value_history) >= self._normality_validator.min_samples:
            # Use robust outlier bounds (MAD-based)
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
            # Use standard IQR bounds
            multiplier = self._effective_fence_multiplier
            q1 = self._stats.q1
            q3 = self._stats.q3
            iqr = self._stats.iqr
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr

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
        if value < lower:
            distance = (lower - value) / iqr
        elif value > upper:
            distance = (value - upper) / iqr
        else:
            distance = 0.0

        # Mapear distancia a [0, 1] con saturación suave
        vote = min(1.0, distance / 3.0)  # >3 IQRs fuera → 1.0
        return float(vote)

    @property
    def is_trained(self) -> bool:
        return self._stats is not None
