"""Caso de uso: Analizar patrones de comportamiento de sensor.

Orquesta:
1. Cargar ventana de datos (vía StoragePort).
2. Detectar patrón de comportamiento (vía PatternDomainService).
3. Detectar change points (si configurado).
4. Clasificar spikes (si se detectaron).
5. Identificar régimen operacional (si configurado).
6. Retornar DTO consolidado.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from ...domain.ports.storage_port import StoragePort
from ...domain.services.pattern_domain_service import PatternDomainService
from ..dto.prediction_dto import PatternDTO

logger = logging.getLogger(__name__)


class AnalyzePatternsUseCase:
    """Caso de uso para análisis de patrones.

    Attributes:
        _pattern_service: Servicio de dominio de patrones.
        _storage: Port de almacenamiento.
        _window_size: Tamaño de ventana por defecto.
    """

    def __init__(
        self,
        pattern_service: PatternDomainService,
        storage: StoragePort,
        window_size: int = 500,
    ) -> None:
        self._pattern_service = pattern_service
        self._storage = storage
        self._window_size = window_size

    def execute(
        self,
        sensor_id: int,
        window_size: Optional[int] = None,
    ) -> PatternDTO:
        """Ejecuta análisis de patrones para un sensor.

        Args:
            sensor_id: ID del sensor.
            window_size: Override del tamaño de ventana.

        Returns:
            ``PatternDTO`` con patrón, change points, spike y régimen.
        """
        t_start = time.monotonic()
        effective_window = window_size or self._window_size

        # 1. Cargar datos
        window = self._storage.load_sensor_window(
            sensor_id=sensor_id,
            limit=effective_window,
        )

        if window.is_empty:
            return PatternDTO(
                sensor_id=sensor_id,
                pattern_type="stable",
                confidence=0.0,
                description="Sin datos disponibles",
            )

        values = window.values

        # 2. Detectar patrón de comportamiento
        pattern_result = self._pattern_service.detect_pattern(window)

        # 3. Detectar change points
        change_points = self._pattern_service.detect_change_points(values)
        cp_dicts = [
            {
                "index": cp.index,
                "change_type": cp.change_type.value,
                "magnitude": cp.magnitude,
                "confidence": cp.confidence,
            }
            for cp in change_points
        ]

        # 4. Clasificar spike si el patrón detectado es SPIKE
        spike_classification: Optional[str] = None
        if pattern_result.pattern_type.value == "spike" and len(values) > 20:
            # Buscar el índice del spike (mayor desviación de la media)
            mean_val = sum(values) / len(values)
            max_dev_idx = max(range(len(values)), key=lambda i: abs(values[i] - mean_val))

            spike_result = self._pattern_service.classify_spike(values, max_dev_idx)
            spike_classification = spike_result.classification.value

        # 5. Régimen operacional
        current_regime: Optional[str] = None
        if window.last_value is not None:
            regime = self._pattern_service.predict_regime(window.last_value)
            if regime is not None:
                current_regime = regime.name

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        logger.info(
            "use_case_analyze_patterns_complete",
            extra={
                "sensor_id": sensor_id,
                "pattern": pattern_result.pattern_type.value,
                "n_change_points": len(change_points),
                "spike_classification": spike_classification,
                "regime": current_regime,
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )

        return PatternDTO(
            sensor_id=sensor_id,
            pattern_type=pattern_result.pattern_type.value,
            confidence=pattern_result.confidence,
            description=pattern_result.description,
            change_points=cp_dicts,
            spike_classification=spike_classification,
            current_regime=current_regime,
        )
