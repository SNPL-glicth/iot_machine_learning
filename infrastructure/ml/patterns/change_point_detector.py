"""Detección de puntos de cambio estructural: CUSUM + PELT.

CUSUM (Cumulative Sum): detección online de cambios pequeños persistentes.
Ideal para drift gradual en series temporales.

PELT (Pruned Exact Linear Time): detección offline óptima de múltiples
cambios.  Requiere ``ruptures`` (opcional, fallback a CUSUM si no está).

Referencia:
- Page, E.S. (1954). "Continuous Inspection Schemes"
- Killick et al. (2012). "Optimal Detection of Changepoints"

ISO 27001: Cada cambio detectado incluye índice, tipo, magnitud y
confianza para trazabilidad completa.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from iot_machine_learning.domain.entities.pattern import ChangePoint, ChangePointType
from iot_machine_learning.domain.ports.pattern_detection_port import ChangePointDetectionPort

logger = logging.getLogger(__name__)


class CUSUMDetector(ChangePointDetectionPort):
    """Detección online de cambios usando CUSUM (Cumulative Sum).

    Mantiene estado interno (acumuladores positivo y negativo).
    Detecta cambios cuando el acumulador excede el threshold.

    Attributes:
        _threshold: Sensibilidad (menor = más sensible).
        _drift: Mínimo cambio a detectar (slack parameter).
        _cumsum_pos: Acumulador positivo (detecta incrementos).
        _cumsum_neg: Acumulador negativo (detecta decrementos).
        _baseline_mean: Media de referencia actual.
        _n_seen: Número de valores procesados.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        drift: float = 0.5,
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold debe ser > 0, recibido {threshold}")
        if drift < 0:
            raise ValueError(f"drift debe ser >= 0, recibido {drift}")

        self._threshold = threshold
        self._drift = drift

        # Estado online
        self._cumsum_pos: float = 0.0
        self._cumsum_neg: float = 0.0
        self._baseline_mean: Optional[float] = None
        self._n_seen: int = 0

    def detect_online(self, value: float) -> Optional[ChangePoint]:
        """Detecta cambio en modo online (1 valor a la vez).

        Args:
            value: Nueva observación de la serie.

        Returns:
            ``ChangePoint`` si se detectó cambio, ``None`` si no.
        """
        self._n_seen += 1

        if self._baseline_mean is None:
            self._baseline_mean = value
            return None

        deviation = value - self._baseline_mean

        # Acumuladores CUSUM
        self._cumsum_pos = max(0.0, self._cumsum_pos + deviation - self._drift)
        self._cumsum_neg = max(0.0, self._cumsum_neg - deviation - self._drift)

        # Detectar exceso de threshold
        if self._cumsum_pos > self._threshold:
            magnitude = self._cumsum_pos
            confidence = min(0.95, magnitude / (self._threshold * 2.0))

            # Reset
            old_mean = self._baseline_mean
            self._cumsum_pos = 0.0
            self._cumsum_neg = 0.0
            self._baseline_mean = value

            logger.info(
                "cusum_change_detected",
                extra={
                    "direction": "up",
                    "index": self._n_seen - 1,
                    "magnitude": round(magnitude, 4),
                    "old_mean": round(old_mean, 4),
                    "new_mean": round(value, 4),
                },
            )

            return ChangePoint(
                index=self._n_seen - 1,
                change_type=ChangePointType.LEVEL_SHIFT,
                magnitude=magnitude,
                confidence=confidence,
                before_mean=old_mean,
                after_mean=value,
            )

        if self._cumsum_neg > self._threshold:
            magnitude = self._cumsum_neg
            confidence = min(0.95, magnitude / (self._threshold * 2.0))

            old_mean = self._baseline_mean
            self._cumsum_neg = 0.0
            self._cumsum_pos = 0.0
            self._baseline_mean = value

            logger.info(
                "cusum_change_detected",
                extra={
                    "direction": "down",
                    "index": self._n_seen - 1,
                    "magnitude": round(magnitude, 4),
                    "old_mean": round(old_mean, 4),
                    "new_mean": round(value, 4),
                },
            )

            return ChangePoint(
                index=self._n_seen - 1,
                change_type=ChangePointType.LEVEL_SHIFT,
                magnitude=magnitude,
                confidence=confidence,
                before_mean=old_mean,
                after_mean=value,
            )

        return None

    def detect_batch(self, values: List[float]) -> List[ChangePoint]:
        """Detecta cambios en batch (ventana completa).

        Ejecuta CUSUM sobre toda la serie sin modificar el estado online.

        Args:
            values: Serie temporal completa.

        Returns:
            Lista de ``ChangePoint`` detectados.
        """
        if len(values) < 10:
            return []

        change_points: List[ChangePoint] = []

        # Baseline: media de primeros 20%
        baseline_n = max(10, int(len(values) * 0.2))
        baseline_mean = sum(values[:baseline_n]) / baseline_n

        cumsum_pos = 0.0
        cumsum_neg = 0.0

        for i, value in enumerate(values):
            deviation = value - baseline_mean

            cumsum_pos = max(0.0, cumsum_pos + deviation - self._drift)
            cumsum_neg = max(0.0, cumsum_neg - deviation - self._drift)

            if cumsum_pos > self._threshold:
                magnitude = cumsum_pos
                confidence = min(0.95, magnitude / (self._threshold * 2.0))

                change_points.append(ChangePoint(
                    index=i,
                    timestamp=float(i),
                    change_type=ChangePointType.LEVEL_SHIFT,
                    magnitude=magnitude,
                    confidence=confidence,
                    before_mean=baseline_mean,
                    after_mean=value,
                ))

                cumsum_pos = 0.0
                baseline_mean = value

            elif cumsum_neg > self._threshold:
                magnitude = cumsum_neg
                confidence = min(0.95, magnitude / (self._threshold * 2.0))

                change_points.append(ChangePoint(
                    index=i,
                    timestamp=float(i),
                    change_type=ChangePointType.LEVEL_SHIFT,
                    magnitude=magnitude,
                    confidence=confidence,
                    before_mean=baseline_mean,
                    after_mean=value,
                ))

                cumsum_neg = 0.0
                baseline_mean = value

        return change_points

    def reset(self) -> None:
        """Reinicia estado interno del detector."""
        self._cumsum_pos = 0.0
        self._cumsum_neg = 0.0
        self._baseline_mean = None
        self._n_seen = 0


class PELTDetector(ChangePointDetectionPort):
    """Detección offline de cambios usando PELT.

    Requiere ``ruptures``.  Si no está disponible, delega a CUSUM.

    Attributes:
        _min_segment_size: Tamaño mínimo de segmento entre cambios.
        _penalty: Penalización para evitar sobre-segmentación.
        _cusum_fallback: Instancia de CUSUM para fallback.
    """

    def __init__(
        self,
        min_segment_size: int = 10,
        penalty: float = 3.0,
    ) -> None:
        self._min_segment_size = min_segment_size
        self._penalty = penalty
        self._cusum_fallback = CUSUMDetector()

    def detect_online(self, value: float) -> Optional[ChangePoint]:
        """PELT es offline — delega a CUSUM para online."""
        return self._cusum_fallback.detect_online(value)

    def detect_batch(self, values: List[float]) -> List[ChangePoint]:
        """Detecta cambios usando PELT (o CUSUM como fallback).

        Args:
            values: Serie temporal completa.

        Returns:
            Lista de ``ChangePoint``.
        """
        try:
            import ruptures as rpt
            import numpy as np
        except ImportError:
            logger.info("ruptures_not_available_fallback_cusum")
            return self._cusum_fallback.detect_batch(values)

        if len(values) < self._min_segment_size * 2:
            return []

        # PELT con modelo RBF
        signal = np.array(values)
        algo = rpt.Pelt(model="rbf", min_size=self._min_segment_size)
        algo.fit(signal)
        change_indices = algo.predict(pen=self._penalty)

        change_points: List[ChangePoint] = []
        for idx in change_indices[:-1]:  # Último es siempre len(values)
            if idx < 10 or idx >= len(values) - 5:
                continue

            before = values[max(0, idx - 10) : idx]
            after = values[idx : min(len(values), idx + 10)]

            mean_before = sum(before) / len(before)
            mean_after = sum(after) / len(after)

            var_before = sum((v - mean_before) ** 2 for v in before) / len(before)
            var_after = sum((v - mean_after) ** 2 for v in after) / len(after)
            std_before = math.sqrt(var_before) if var_before > 0 else 1e-9

            level_diff = abs(mean_after - mean_before)
            var_diff = abs(var_after - var_before)

            # Clasificar tipo de cambio
            if level_diff > 2.0 * std_before:
                change_type = ChangePointType.LEVEL_SHIFT
                magnitude = level_diff
            elif var_diff > var_before:
                change_type = ChangePointType.VARIANCE_CHANGE
                magnitude = var_diff
            else:
                change_type = ChangePointType.TREND_CHANGE
                magnitude = level_diff

            change_points.append(ChangePoint(
                index=idx,
                timestamp=float(idx),
                change_type=change_type,
                magnitude=magnitude,
                confidence=0.85,
                before_mean=mean_before,
                after_mean=mean_after,
            ))

        logger.info(
            "pelt_detection_complete",
            extra={
                "n_values": len(values),
                "n_change_points": len(change_points),
            },
        )

        return change_points

    def reset(self) -> None:
        """Reinicia estado (PELT es stateless, resetea CUSUM fallback)."""
        self._cusum_fallback.reset()
