"""Clasificador de spikes: delta (cambio legítimo) vs noise (ruido).

Criterios de clasificación:
1. **Magnitud:** ¿El cambio excede N sigmas de la ventana histórica?
2. **Persistencia:** ¿El nuevo nivel se mantiene post-spike?
3. **Alineación con tendencia:** ¿El spike va en dirección del trend previo?

Ejemplo:
- Cambio de estado → valor sube 10 unidades en 1 paso y se mantiene → DELTA_SPIKE
- Glitch de medición → lectura outlier aislada → NOISE_SPIKE

ISO 27001: Todas las decisiones incluyen razones cuantificadas
para trazabilidad y auditoría.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional

from iot_machine_learning.domain.entities.pattern import DeltaSpikeResult, SpikeClassification
from iot_machine_learning.domain.ports.pattern_detection_port import DeltaSpikeClassificationPort

if TYPE_CHECKING:
    from ..anomaly.core.config import AnomalyDetectorConfig

logger = logging.getLogger(__name__)


class DeltaSpikeClassifier(DeltaSpikeClassificationPort):
    """Clasificador enterprise de spikes con explicabilidad completa.

    Attributes:
        _magnitude_threshold_sigma: Umbral en sigmas para considerar spike.
        _persistence_window: Lecturas post-spike para evaluar persistencia.
        _min_history: Mínimo de puntos pre-spike para estadísticas.
    """

    def __init__(
        self,
        magnitude_threshold_sigma: Optional[float] = None,
        persistence_window: Optional[int] = None,
        min_history: Optional[int] = None,
        config: Optional[AnomalyDetectorConfig] = None,
    ) -> None:
        """Initialize DeltaSpikeClassifier.
        
        Args:
            magnitude_threshold_sigma: Deprecated. Use config.delta_magnitude_sigma.
            persistence_window: Deprecated. Use config.delta_persistence_window.
            min_history: Deprecated. Use config.delta_min_history.
            config: Configuration object with delta spike parameters.
                   Takes precedence over individual parameters.
        
        Backward compatibility: If config is None, uses individual params
        or defaults (2.0, 5, 20).
        """
        # Use config if provided, otherwise fall back to individual params
        effective_mag = magnitude_threshold_sigma
        effective_window = persistence_window
        effective_history = min_history
        persistence_threshold = 0.6
        trend_threshold = 0.8
        
        if config is not None:
            effective_mag = config.delta_magnitude_sigma
            effective_window = config.delta_persistence_window
            effective_history = config.delta_min_history
            persistence_threshold = config.delta_persistence_score_threshold
            trend_threshold = config.delta_trend_alignment_threshold
        
        # Apply defaults if still None
        if effective_mag is None:
            effective_mag = 2.0
        if effective_window is None:
            effective_window = 5
        if effective_history is None:
            effective_history = 20
        
        if effective_mag <= 0:
            raise ValueError(
                f"magnitude_threshold_sigma debe ser > 0, recibido {effective_mag}"
            )
        if effective_window < 2:
            raise ValueError(
                f"persistence_window debe ser >= 2, recibido {effective_window}"
            )
        if effective_history < 5:
            raise ValueError(
                f"min_history debe ser >= 5, recibido {effective_history}"
            )

        self._magnitude_threshold_sigma = effective_mag
        self._persistence_window = effective_window
        self._min_history = effective_history
        self._persistence_threshold = persistence_threshold
        self._trend_threshold = trend_threshold

    def classify(
        self,
        values: List[float],
        spike_index: int,
    ) -> DeltaSpikeResult:
        """Clasifica un spike detectado.

        Args:
            values: Serie temporal completa.
            spike_index: Índice donde ocurre el spike.

        Returns:
            ``DeltaSpikeResult`` con clasificación y razones cuantificadas.
        """
        n = len(values)

        # Guard: historia insuficiente
        if spike_index < self._min_history:
            return DeltaSpikeResult(
                is_delta_spike=False,
                confidence=0.0,
                delta_magnitude=0.0,
                persistence_score=0.0,
                classification=SpikeClassification.NORMAL,
                explanation=(
                    f"Historia insuficiente: {spike_index} puntos pre-spike, "
                    f"mínimo requerido {self._min_history}"
                ),
            )

        # --- 1. Magnitud del cambio ---
        pre_window = values[spike_index - self._min_history : spike_index]
        spike_value = values[spike_index]

        pre_mean = sum(pre_window) / len(pre_window)
        pre_var = sum((v - pre_mean) ** 2 for v in pre_window) / len(pre_window)
        pre_std = math.sqrt(pre_var) if pre_var > 0 else 0.0

        delta_magnitude = abs(spike_value - pre_mean)
        z_score = delta_magnitude / max(pre_std, 1e-9)

        # --- 2. Persistencia post-spike ---
        persistence_score = self._compute_persistence(
            values, spike_index, pre_mean, pre_std
        )

        # --- 3. Alineación con tendencia previa ---
        trend_alignment = self._compute_trend_alignment(
            pre_window, spike_value, pre_mean
        )

        # --- 4. Decisión final ---
        is_significant = z_score > self._magnitude_threshold_sigma

        is_delta = is_significant and (
            persistence_score > self._persistence_threshold or trend_alignment > self._trend_threshold
        )

        if is_delta:
            classification = SpikeClassification.DELTA_SPIKE
            confidence = min(0.95, (z_score / 5.0) * persistence_score)
            explanation = (
                f"Cambio legítimo: magnitud {z_score:.1f}σ, "
                f"persistencia {persistence_score:.2f}, "
                f"alineación tendencia {trend_alignment:.2f}"
            )
        elif is_significant:
            classification = SpikeClassification.NOISE_SPIKE
            confidence = min(0.9, 0.5 + persistence_score * 0.3)
            explanation = (
                f"Ruido detectado: magnitud {z_score:.1f}σ pero "
                f"baja persistencia {persistence_score:.2f}"
            )
        else:
            classification = SpikeClassification.NORMAL
            confidence = 0.9
            explanation = (
                f"Variación normal: {z_score:.1f}σ bajo umbral "
                f"{self._magnitude_threshold_sigma}σ"
            )

        logger.info(
            "delta_spike_classified",
            extra={
                "spike_index": spike_index,
                "z_score": round(z_score, 2),
                "persistence": round(persistence_score, 2),
                "trend_alignment": round(trend_alignment, 2),
                "classification": classification.value,
                "confidence": round(confidence, 2),
            },
        )

        return DeltaSpikeResult(
            is_delta_spike=is_delta,
            confidence=confidence,
            delta_magnitude=delta_magnitude,
            persistence_score=persistence_score,
            classification=classification,
            explanation=explanation,
            trend_alignment=trend_alignment,
        )

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _compute_persistence(
        self,
        values: List[float],
        spike_index: int,
        pre_mean: float,
        pre_std: float,
    ) -> float:
        """Evalúa si el nuevo nivel se mantiene post-spike.

        Retorna 0.0 si vuelve al nivel pre-spike (noise).
        Retorna ~1.0 si el nuevo nivel es estable (delta).

        Args:
            values: Serie completa.
            spike_index: Índice del spike.
            pre_mean: Media pre-spike.
            pre_std: Std pre-spike.

        Returns:
            Score de persistencia (0–1).
        """
        end_idx = spike_index + self._persistence_window
        if end_idx > len(values):
            # No hay suficiente data post-spike → neutral
            return 0.5

        post_window = values[spike_index : end_idx]
        post_mean = sum(post_window) / len(post_window)

        # ¿Volvió cerca del nivel pre-spike?
        recovery_to_baseline = abs(post_mean - pre_mean) < max(pre_std, 1e-9)

        if recovery_to_baseline:
            return 0.0

        # Estabilidad del nuevo nivel
        post_var = sum((v - post_mean) ** 2 for v in post_window) / len(post_window)
        post_std = math.sqrt(post_var) if post_var > 0 else 0.0

        level_shift = abs(post_mean - pre_mean)
        if level_shift < 1e-9:
            return 0.0

        persistence = 1.0 - min(post_std / level_shift, 1.0)
        return max(0.0, min(1.0, persistence))

    def _compute_trend_alignment(
        self,
        pre_window: List[float],
        spike_value: float,
        pre_mean: float,
    ) -> float:
        """Evalúa si el spike va en dirección de la tendencia previa.

        Si la serie ya subía y el spike es hacia arriba → más probable delta.

        Args:
            pre_window: Ventana pre-spike.
            spike_value: Valor del spike.
            pre_mean: Media pre-spike.

        Returns:
            Score de alineación (0–1).  0.5 = neutral.
        """
        if len(pre_window) < 3:
            return 0.5

        # Pendiente simple: diferencia entre primera y última mitad
        mid = len(pre_window) // 2
        first_half_mean = sum(pre_window[:mid]) / mid
        second_half_mean = sum(pre_window[mid:]) / (len(pre_window) - mid)

        slope_sign = 1.0 if second_half_mean > first_half_mean else -1.0
        spike_direction = 1.0 if spike_value > pre_mean else -1.0

        if slope_sign == spike_direction:
            return 1.0  # Alineado
        return 0.3  # Contra-tendencia
