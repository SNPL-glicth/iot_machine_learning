"""Métricas A/B para comparar baseline vs enterprise en batch runner.

Recolecta métricas por sensor y por engine para evaluar la calidad
de la migración gradual.

Restricción: < 180 líneas.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SensorMetric:
    """Métricas acumuladas para un sensor individual."""

    sensor_id: int
    engine_used: str = ""
    predicted_value: float = 0.0
    confidence: float = 0.0
    elapsed_ms: float = 0.0
    success: bool = True
    timestamp: float = field(default_factory=time.time)


@dataclass
class EngineStats:
    """Estadísticas agregadas por engine."""

    count: int = 0
    total_confidence: float = 0.0
    total_elapsed_ms: float = 0.0
    failures: int = 0

    @property
    def avg_confidence(self) -> float:
        return self.total_confidence / self.count if self.count > 0 else 0.0

    @property
    def avg_elapsed_ms(self) -> float:
        return self.total_elapsed_ms / self.count if self.count > 0 else 0.0


class ABMetricsCollector:
    """Recolector de métricas A/B para batch runner.

    Uso:
        collector = ABMetricsCollector()
        collector.record(sensor_id=42, engine="taylor", ...)
        collector.record(sensor_id=43, engine="baseline_legacy", ...)
        print(collector.summary())
    """

    def __init__(self) -> None:
        self._by_engine: Dict[str, EngineStats] = {}
        self._recent: List[SensorMetric] = []
        self._max_recent = 1000

    def record(
        self,
        sensor_id: int,
        engine_used: str,
        predicted_value: float,
        confidence: float,
        elapsed_ms: float = 0.0,
        success: bool = True,
    ) -> None:
        """Registra una predicción para métricas A/B.

        Args:
            sensor_id: ID del sensor.
            engine_used: Nombre del engine que generó la predicción.
            predicted_value: Valor predicho.
            confidence: Confianza (0–1).
            elapsed_ms: Tiempo de ejecución en ms.
            success: Si la predicción fue exitosa.
        """
        metric = SensorMetric(
            sensor_id=sensor_id,
            engine_used=engine_used,
            predicted_value=predicted_value,
            confidence=confidence,
            elapsed_ms=elapsed_ms,
            success=success,
        )

        # Agregar a recientes (ring buffer)
        if len(self._recent) >= self._max_recent:
            self._recent.pop(0)
        self._recent.append(metric)

        # Agregar a stats por engine
        if engine_used not in self._by_engine:
            self._by_engine[engine_used] = EngineStats()

        stats = self._by_engine[engine_used]
        stats.count += 1
        stats.total_confidence += confidence
        stats.total_elapsed_ms += elapsed_ms
        if not success:
            stats.failures += 1

    def summary(self) -> Dict[str, object]:
        """Resumen de métricas A/B.

        Returns:
            Dict con estadísticas por engine y totales.
        """
        engines = {}
        for name, stats in self._by_engine.items():
            engines[name] = {
                "count": stats.count,
                "avg_confidence": round(stats.avg_confidence, 4),
                "avg_elapsed_ms": round(stats.avg_elapsed_ms, 2),
                "failures": stats.failures,
            }

        total = sum(s.count for s in self._by_engine.values())
        enterprise = sum(
            s.count
            for name, s in self._by_engine.items()
            if not name.startswith("baseline")
        )

        return {
            "total_predictions": total,
            "enterprise_predictions": enterprise,
            "enterprise_ratio": round(enterprise / total, 4) if total > 0 else 0.0,
            "engines": engines,
        }

    def reset(self) -> None:
        """Resetea todas las métricas."""
        self._by_engine.clear()
        self._recent.clear()
