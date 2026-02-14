"""Audit logger específico para batch runner enterprise bridge.

Wrapper ligero sobre AuditPort que agrega contexto batch
(cycle_id, sensor_count, etc.) a cada evento.

Restricción: < 180 líneas.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

from iot_machine_learning.domain.ports.audit_port import AuditPort

logger = logging.getLogger(__name__)


class BatchAuditLogger:
    """Audit logger con contexto de ciclo batch.

    Cada ciclo batch genera un ``cycle_id`` único que se incluye
    en todos los eventos del ciclo para trazabilidad end-to-end.

    Attributes:
        _audit: AuditPort subyacente.
        _cycle_id: ID del ciclo actual.
        _cycle_start: Timestamp de inicio del ciclo.
    """

    def __init__(self, audit: AuditPort) -> None:
        self._audit = audit
        self._cycle_id: Optional[str] = None
        self._cycle_start: float = 0.0

    def start_cycle(self, sensor_count: int) -> str:
        """Inicia un nuevo ciclo batch.

        Args:
            sensor_count: Número de sensores a procesar.

        Returns:
            cycle_id generado.
        """
        self._cycle_id = str(uuid.uuid4())[:12]
        self._cycle_start = time.time()

        self._audit.log_event(
            event_type="batch_cycle_start",
            action="start",
            resource="ml_batch_runner",
            user_id="ml_batch_runner",
            details={
                "cycle_id": self._cycle_id,
                "sensor_count": sensor_count,
            },
            result="success",
        )

        return self._cycle_id

    def end_cycle(
        self,
        processed: int,
        errors: int,
        enterprise_count: int,
        baseline_count: int,
    ) -> None:
        """Finaliza el ciclo batch actual.

        Args:
            processed: Sensores procesados exitosamente.
            errors: Sensores con error.
            enterprise_count: Predicciones via enterprise.
            baseline_count: Predicciones via baseline.
        """
        elapsed = time.time() - self._cycle_start if self._cycle_start else 0.0

        self._audit.log_event(
            event_type="batch_cycle_end",
            action="complete",
            resource="ml_batch_runner",
            user_id="ml_batch_runner",
            details={
                "cycle_id": self._cycle_id,
                "processed": processed,
                "errors": errors,
                "enterprise_count": enterprise_count,
                "baseline_count": baseline_count,
                "elapsed_seconds": round(elapsed, 2),
            },
            result="success" if errors == 0 else "warning",
        )

    def log_sensor_routing(
        self,
        sensor_id: int,
        route: str,
        reason: str,
    ) -> None:
        """Registra la decisión de routing para un sensor.

        Args:
            sensor_id: ID del sensor.
            route: "enterprise" o "baseline_legacy".
            reason: Razón de la decisión (flag, whitelist, etc.).
        """
        self._audit.log_event(
            event_type="batch_sensor_routing",
            action="route",
            resource=f"sensor_{sensor_id}",
            user_id="ml_batch_runner",
            details={
                "cycle_id": self._cycle_id,
                "route": route,
                "reason": reason,
            },
            result="success",
        )

    @property
    def cycle_id(self) -> Optional[str]:
        return self._cycle_id
