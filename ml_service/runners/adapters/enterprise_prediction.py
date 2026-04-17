"""Adapter: conecta batch runner al enterprise PredictSensorValueUseCase.

ISO 27001 A.12.4: audit trail, fallback garantizado, métricas A/B.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from iot_machine_learning.application.use_cases.predict_sensor_value import (
    PredictSensorValueUseCase,
)
from iot_machine_learning.domain.ports.audit_port import AuditPort
from iot_machine_learning.domain.ports.storage_port import StoragePort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchPredictionResult:
    """Resultado de predicción enterprise compatible con batch runner.

    Contiene los campos que el batch runner necesita para persistir
    la predicción y generar eventos, más extras enterprise opcionales.
    """

    predicted_value: float
    confidence: float
    trend: str
    engine_used: str
    anomaly_score: float = 0.0
    structural_regime: Optional[str] = None
    trace_id: Optional[str] = None
    elapsed_ms: float = 0.0


@dataclass
class _AdapterMetrics:
    """Métricas internas para A/B testing."""

    enterprise_success: int = 0
    enterprise_failure: int = 0
    total_elapsed_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.enterprise_success + self.enterprise_failure
        return self.enterprise_success / total if total > 0 else 0.0

    @property
    def avg_elapsed_ms(self) -> float:
        return (
            self.total_elapsed_ms / self.enterprise_success
            if self.enterprise_success > 0
            else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "enterprise_success": self.enterprise_success,
            "enterprise_failure": self.enterprise_failure,
            "success_rate": round(self.success_rate, 4),
            "avg_elapsed_ms": round(self.avg_elapsed_ms, 2),
        }


class EnterprisePredictionAdapter:
    """Adapta PredictSensorValueUseCase para el batch runner.

    Cumplimiento ISO 27001:
    - Audit log por predicción (event_type="batch_prediction_enterprise")
    - Fallback con razón documentada
    - Métricas de éxito/fallo para A/B testing

    Attributes:
        _storage: StoragePort para cargar SensorWindow con timestamps.
        _use_case: Use case enterprise de predicción.
        _audit: Port de auditoría ISO 27001.
        _metrics: Métricas acumuladas para A/B.
    """

    def __init__(
        self,
        storage: StoragePort,
        use_case: PredictSensorValueUseCase,
        audit: AuditPort,
    ) -> None:
        self._storage = storage
        self._use_case = use_case
        self._audit = audit
        self._metrics = _AdapterMetrics()

    def predict(
        self,
        sensor_id: int,
        window_size: int = 500,
    ) -> BatchPredictionResult:
        """Predice usando enterprise stack con fallback garantizado."""
        t0 = time.monotonic()
        try:
            dto = self._use_case.execute(
                sensor_id=sensor_id, window_size=window_size,
            )
            return self._on_success(
                dto, t0, sensor_id, window_size, "batch_prediction_enterprise",
            )
        except Exception as exc:
            return self._on_failure(exc, t0, sensor_id, window_size)

    def predict_with_window(
        self,
        sensor_window: "SensorWindow",
    ) -> BatchPredictionResult:
        """Predict using a pre-loaded SensorWindow (no SQL reload)."""
        t0 = time.monotonic()
        try:
            dto = self._use_case.execute_with_window(
                sensor_window=sensor_window,
            )
            return self._on_success(
                dto, t0, sensor_window.sensor_id, sensor_window.size,
                "batch_prediction_preloaded",
            )
        except Exception as exc:
            return self._on_failure(
                exc, t0, sensor_window.sensor_id, sensor_window.size,
            )

    def _on_success(self, dto, t0, sensor_id, window_size, event_type):
        elapsed = (time.monotonic() - t0) * 1000.0
        result = BatchPredictionResult(
            predicted_value=dto.predicted_value,
            confidence=dto.confidence_score,
            trend=dto.trend,
            engine_used=dto.engine_name,
            trace_id=dto.audit_trace_id,
            elapsed_ms=elapsed,
        )
        self._metrics.enterprise_success += 1
        self._metrics.total_elapsed_ms += elapsed
        # OBSERVABILITY: Track engine usage per prediction
        from ...metrics.observability import get_observability
        get_observability().engine_usage.record(result.engine_used, sensor_id)
        self._audit.log_event(
            event_type=event_type, action="predict",
            resource=f"sensor_{sensor_id}", user_id="ml_batch_runner",
            details={
                "engine": result.engine_used, "confidence": result.confidence,
                "window_size": window_size, "elapsed_ms": round(elapsed, 2),
            },
            result="success",
        )
        return result

    def _on_failure(self, exc, t0, sensor_id, window_size):
        elapsed = (time.monotonic() - t0) * 1000.0
        reason = f"{type(exc).__name__}: {exc}"
        logger.exception(
            "enterprise_prediction_failed",
            extra={"sensor_id": sensor_id, "error": reason},
        )
        self._metrics.enterprise_failure += 1
        # OBSERVABILITY: Track fallback with counter and per-sensor metrics
        from ...metrics.observability import get_observability
        obs = get_observability()
        obs.fallback.record(sensor_id, reason)
        fallback_rate = obs.get_fallback_rate(self._metrics.enterprise_success + self._metrics.enterprise_failure)
        if fallback_rate > 0.10:
            logger.critical("ml.enterprise.fallback.rate_critical", extra={"rate": fallback_rate})
        self._audit.log_event(
            event_type="batch_prediction_fallback",
            action="fallback_baseline",
            resource=f"sensor_{sensor_id}", user_id="ml_batch_runner",
            details={"reason": reason, "elapsed_ms": round(elapsed, 2), "fallback_rate": fallback_rate},
            result="warning",
        )
        from .fallback_baseline import fallback_to_baseline
        return fallback_to_baseline(self._storage, sensor_id, window_size)

    @property
    def metrics(self) -> dict:
        """Métricas acumuladas para A/B testing."""
        return self._metrics.to_dict()
