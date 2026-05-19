"""Adapter: conecta MetaCognitiveOrchestrator al batch runner.

Mismo contrato que EnterprisePredictionAdapter para drop-in replacement.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import TYPE_CHECKING, Optional

from iot_machine_learning.ml_service.runners.adapters.enterprise_prediction import (
    BatchPredictionResult,
)

if TYPE_CHECKING:
    from iot_machine_learning.domain.ports.storage_port import StoragePort
    from iot_machine_learning.domain.ports.audit_port import AuditPort
    from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import (
        MetaCognitiveOrchestrator,
    )
    from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)


class OrchestratorPredictionAdapter:
    """Adapta MetaCognitiveOrchestrator para el batch runner.

    Contrato idéntico a EnterprisePredictionAdapter:
    - predict(sensor_id, window_size) -> BatchPredictionResult
    - predict_with_window(sensor_window) -> BatchPredictionResult
    """

    def __init__(
        self,
        orchestrator: "MetaCognitiveOrchestrator",
        storage: "StoragePort",
        audit: "AuditPort",
        flags: "FeatureFlags",
    ) -> None:
        self._orchestrator = orchestrator
        self._storage = storage
        self._audit = audit
        self._flags = flags
        self._timeout_seconds = getattr(
            flags, "ML_ORCHESTRATOR_TIMEOUT_S", 5.0
        )

    @property
    def orchestrator(self):
        """Public accessor for the wrapped MetaCognitiveOrchestrator.

        Used by FASE-1A wiring to inject the orchestrator into
        PredictionDomainService as a PredictionPort.
        """
        return self._orchestrator

    def predict(
        self,
        sensor_id: int,
        window_size: int = 500,
    ) -> BatchPredictionResult:
        """Predice usando el orchestrador cognitivo."""
        t0 = time.monotonic()
        try:
            window = self._storage.load_sensor_window(sensor_id, limit=window_size)
            return self._predict_window(window, sensor_id, t0)
        except Exception as exc:
            return self._on_failure(exc, t0, sensor_id, window_size)

    def predict_with_window(
        self,
        sensor_window: "SensorWindow",
    ) -> BatchPredictionResult:
        """Predict usando ventana pre-cargada (no SQL reload)."""
        t0 = time.monotonic()
        try:
            return self._predict_window(
                sensor_window, sensor_window.sensor_id, t0,
            )
        except Exception as exc:
            return self._on_failure(
                exc, t0, sensor_window.sensor_id, sensor_window.size,
            )

    def _run_with_timeout(self, fn, timeout_seconds: float = 5.0):
        """Execute fn in a separate thread with a hard timeout.

        Raises TimeoutError if fn does not complete within timeout_seconds.
        Uses shutdown(wait=False) so a hung thread does not block the caller.
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            future.cancel()
            executor.shutdown(wait=False)
            logger.error(
                "orchestrator_timeout",
                extra={
                    "timeout_seconds": timeout_seconds,
                    "engine": "cognitive_orchestrator",
                },
            )
            raise TimeoutError(
                f"Orchestrator exceeded {timeout_seconds}s budget"
            )
        except Exception:
            executor.shutdown(wait=False)
            raise
        else:
            executor.shutdown(wait=False)

    def _predict_window(
        self,
        window: "SensorWindow",
        sensor_id: int,
        t0: float,
    ) -> BatchPredictionResult:
        values = [r.value for r in window.readings]
        timestamps = [r.timestamp for r in window.readings]

        def _do_predict():
            return self._orchestrator.predict(
                series_id=str(sensor_id),
                values=values,
                timestamps=timestamps,
                flags_snapshot=self._flags,
            )

        result = self._run_with_timeout(
            _do_predict, timeout_seconds=self._timeout_seconds
        )

        elapsed = (time.monotonic() - t0) * 1000.0

        # Extract engine name from cognitive metadata
        engine_name = result.metadata.get("selected_engine", "ensemble") if result.metadata else "ensemble"
        if engine_name == "ensemble":
            engine_name = result.metadata.get("engine_name", "ensemble") if result.metadata else "ensemble"

        return BatchPredictionResult(
            predicted_value=result.predicted_value,
            confidence=result.confidence,
            trend=result.trend,
            engine_used=str(engine_name),
            elapsed_ms=elapsed,
        )

    def _on_failure(self, exc, t0, sensor_id, window_size):
        elapsed = (time.monotonic() - t0) * 1000.0
        reason = f"{type(exc).__name__}: {exc}"
        logger.exception(
            "orchestrator_prediction_failed",
            extra={"sensor_id": sensor_id, "error": reason},
        )
        self._audit.log_event(
            event_type="batch_prediction_fallback",
            action="fallback_baseline",
            resource=f"sensor_{sensor_id}",
            user_id="ml_batch_runner",
            details={"reason": reason, "elapsed_ms": round(elapsed, 2)},
            result="warning",
        )
        from .fallback_baseline import fallback_to_baseline
        return fallback_to_baseline(self._storage, sensor_id, window_size)
