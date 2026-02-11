"""Servicio de dominio para detección de patrones.

Orquesta detección de patrones, change points y clasificación de spikes.
Combina resultados de múltiples detectores para dar una visión completa
del comportamiento del sensor.

Responsabilidades:
- Detectar patrón de comportamiento actual.
- Detectar puntos de cambio estructural.
- Clasificar spikes (delta vs noise).
- Identificar régimen operacional.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from ..entities.pattern import (
    ChangePoint,
    DeltaSpikeResult,
    OperationalRegime,
    PatternResult,
    PatternType,
    SpikeClassification,
)
from ..entities.sensor_reading import SensorWindow
from ..ports.audit_port import AuditPort
from ..ports.pattern_detection_port import (
    ChangePointDetectionPort,
    DeltaSpikeClassificationPort,
    PatternDetectionPort,
    RegimeDetectionPort,
)

logger = logging.getLogger(__name__)


class PatternDomainService:
    """Servicio de dominio que orquesta detección de patrones.

    Combina múltiples capacidades de detección bajo una interfaz
    unificada.  Cada capacidad es opcional — si no se provee el port,
    esa funcionalidad se omite silenciosamente.

    Attributes:
        _pattern_detector: Detector de patrones de comportamiento.
        _change_point_detector: Detector de cambios estructurales.
        _spike_classifier: Clasificador de spikes.
        _regime_detector: Detector de regímenes operacionales.
        _audit: Port de auditoría (opcional).
    """

    def __init__(
        self,
        pattern_detector: Optional[PatternDetectionPort] = None,
        change_point_detector: Optional[ChangePointDetectionPort] = None,
        spike_classifier: Optional[DeltaSpikeClassificationPort] = None,
        regime_detector: Optional[RegimeDetectionPort] = None,
        audit: Optional[AuditPort] = None,
    ) -> None:
        self._pattern_detector = pattern_detector
        self._change_point_detector = change_point_detector
        self._spike_classifier = spike_classifier
        self._regime_detector = regime_detector
        self._audit = audit

    def detect_pattern(self, window: SensorWindow) -> PatternResult:
        """Detecta el patrón de comportamiento actual.

        Args:
            window: Ventana temporal del sensor.

        Returns:
            ``PatternResult`` con tipo y confianza.  Si no hay detector
            configurado, retorna ``STABLE`` con confianza 0.
        """
        if self._pattern_detector is None:
            return PatternResult(
                series_id=str(window.sensor_id),
                pattern_type=PatternType.STABLE,
                confidence=0.0,
                description="Sin detector de patrones configurado",
            )

        try:
            return self._pattern_detector.detect_pattern(window)
        except Exception as exc:
            logger.warning(
                "pattern_detection_failed",
                extra={"series_id": str(window.sensor_id), "error": str(exc)},
            )
            return PatternResult(
                series_id=str(window.sensor_id),
                pattern_type=PatternType.STABLE,
                confidence=0.0,
                description=f"Error en detección: {exc}",
            )

    def detect_change_points(
        self, values: List[float]
    ) -> List[ChangePoint]:
        """Detecta puntos de cambio estructural en batch.

        Args:
            values: Serie temporal completa.

        Returns:
            Lista de ``ChangePoint``.  Vacía si no hay detector o falla.
        """
        if self._change_point_detector is None:
            return []

        try:
            return self._change_point_detector.detect_batch(values)
        except Exception as exc:
            logger.warning(
                "change_point_detection_failed",
                extra={"error": str(exc), "n_values": len(values)},
            )
            return []

    def detect_change_point_online(self, value: float) -> Optional[ChangePoint]:
        """Detecta cambio en modo online (1 valor).

        Args:
            value: Nuevo valor del sensor.

        Returns:
            ``ChangePoint`` si se detectó cambio, ``None`` si no.
        """
        if self._change_point_detector is None:
            return None

        try:
            return self._change_point_detector.detect_online(value)
        except Exception as exc:
            logger.warning(
                "change_point_online_failed",
                extra={"error": str(exc), "value": value},
            )
            return None

    def classify_spike(
        self,
        values: List[float],
        spike_index: int,
    ) -> DeltaSpikeResult:
        """Clasifica un spike como delta (legítimo) o noise (ruido).

        Args:
            values: Serie temporal completa.
            spike_index: Índice donde ocurre el spike.

        Returns:
            ``DeltaSpikeResult``.  Si no hay clasificador, retorna NORMAL.
        """
        if self._spike_classifier is None:
            return DeltaSpikeResult(
                is_delta_spike=False,
                confidence=0.0,
                delta_magnitude=0.0,
                persistence_score=0.0,
                classification=SpikeClassification.NORMAL,
                explanation="Sin clasificador de spikes configurado",
            )

        try:
            return self._spike_classifier.classify(values, spike_index)
        except Exception as exc:
            logger.warning(
                "spike_classification_failed",
                extra={"error": str(exc), "spike_index": spike_index},
            )
            return DeltaSpikeResult(
                is_delta_spike=False,
                confidence=0.0,
                delta_magnitude=0.0,
                persistence_score=0.0,
                classification=SpikeClassification.NORMAL,
                explanation=f"Error en clasificación: {exc}",
            )

    def predict_regime(self, value: float) -> Optional[OperationalRegime]:
        """Predice el régimen operacional actual.

        Args:
            value: Valor actual del sensor.

        Returns:
            ``OperationalRegime`` o ``None`` si no hay detector.
        """
        if self._regime_detector is None or not self._regime_detector.is_trained():
            return None

        try:
            return self._regime_detector.predict_regime(value)
        except Exception as exc:
            logger.warning(
                "regime_prediction_failed",
                extra={"error": str(exc), "value": value},
            )
            return None
