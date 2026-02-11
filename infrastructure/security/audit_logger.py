"""Audit Logger conforme a ISO 27001 Anexo A.12.4.

Requisitos implementados:
- A.12.4.1 Event logging: Timestamp, user_id, action, resource, result.
- A.12.4.3 Administrator logs: Before/after state para cambios de config.
- A.12.4.4 Clock synchronization: Timestamps en UTC ISO format.
- Integridad: Hash SHA-256 truncado por entrada para detectar tampering.

Destino configurable: archivo JSON Lines (.jsonl) por defecto.
Cada línea es un JSON independiente para facilitar parsing con
herramientas estándar (jq, grep, SIEM).

Thread-safe: usa logging.FileHandler que es thread-safe por defecto.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...domain.ports.audit_port import AuditPort

logger = logging.getLogger(__name__)


class FileAuditLogger(AuditPort):
    """Implementación de audit logging a archivo JSON Lines.

    Cada evento se escribe como una línea JSON en el archivo de log.
    Incluye hash de integridad para detectar modificaciones.

    Attributes:
        _log_file: Ruta al archivo de audit log.
        _include_hash: Si ``True``, incluye hash SHA-256 por entrada.
        _audit_logger: Logger dedicado para auditoría.
        _lock: Lock para operaciones thread-safe adicionales.
    """

    def __init__(
        self,
        log_file: Path,
        include_hash: bool = True,
        log_to_console: bool = False,
    ) -> None:
        """Inicializa el audit logger.

        Args:
            log_file: Ruta al archivo de log (se crea si no existe).
            include_hash: Si ``True``, incluye hash de integridad.
            log_to_console: Si ``True``, también imprime a consola.
        """
        self._log_file = Path(log_file)
        self._include_hash = include_hash
        self._lock = threading.Lock()

        # Crear directorio si no existe
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

        # Logger dedicado con nombre único por ruta absoluta
        # (evita colisiones entre instancias con mismo filename)
        unique_id = hashlib.md5(str(self._log_file.resolve()).encode()).hexdigest()[:8]
        logger_name = f"audit.{unique_id}"

        self._audit_logger = logging.getLogger(logger_name)
        self._audit_logger.setLevel(logging.INFO)
        self._audit_logger.propagate = False

        # Limpiar handlers previos (evita stale refs en tests)
        for h in list(self._audit_logger.handlers):
            self._audit_logger.removeHandler(h)
            h.close()

        file_handler = logging.FileHandler(str(self._log_file))
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self._audit_logger.addHandler(file_handler)

        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            self._audit_logger.addHandler(console_handler)

    def log_event(
        self,
        event_type: str,
        action: str,
        resource: str,
        details: Dict[str, Any],
        result: str = "success",
        user_id: Optional[str] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registra un evento de auditoría.

        Args:
            event_type: Categoría del evento.
            action: Acción realizada.
            resource: Recurso afectado.
            details: Información adicional.
            result: Resultado de la acción.
            user_id: ID del usuario/proceso.
            before_state: Estado previo (para cambios).
            after_state: Estado nuevo (para cambios).
        """
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "user_id": user_id or "system",
            "action": action,
            "resource": resource,
            "result": result,
            "details": details,
        }

        if before_state is not None:
            entry["before_state"] = before_state

        if after_state is not None:
            entry["after_state"] = after_state

        # Hash de integridad (sobre el contenido sin el hash)
        if self._include_hash:
            content_str = json.dumps(entry, sort_keys=True, default=str)
            entry["integrity_hash"] = hashlib.sha256(
                content_str.encode("utf-8")
            ).hexdigest()[:16]

        # Escribir y flush (audit trail debe persistir inmediatamente)
        with self._lock:
            self._audit_logger.info(
                json.dumps(entry, sort_keys=False, default=str)
            )
            for handler in self._audit_logger.handlers:
                handler.flush()

    def log_prediction(
        self,
        sensor_id: int,
        predicted_value: float,
        confidence: float,
        engine_name: str,
        trace_id: Optional[str] = None,
    ) -> None:
        """Registra predicción generada."""
        self.log_event(
            event_type="prediction",
            action="predict",
            resource=f"sensor_{sensor_id}",
            details={
                "predicted_value": round(predicted_value, 6),
                "confidence": round(confidence, 4),
                "engine": engine_name,
                "trace_id": trace_id,
            },
        )

    def log_anomaly(
        self,
        sensor_id: int,
        value: float,
        score: float,
        explanation: str,
        trace_id: Optional[str] = None,
    ) -> None:
        """Registra detección de anomalía."""
        self.log_event(
            event_type="anomaly_detection",
            action="detect_anomaly",
            resource=f"sensor_{sensor_id}",
            details={
                "value": round(value, 6),
                "anomaly_score": round(score, 4),
                "explanation": explanation,
                "trace_id": trace_id,
            },
        )

    def log_config_change(
        self,
        config_key: str,
        old_value: Any,
        new_value: Any,
        user_id: str,
    ) -> None:
        """Registra cambio de configuración."""
        self.log_event(
            event_type="config_change",
            user_id=user_id,
            action="update_config",
            resource=config_key,
            details={},
            before_state={"value": old_value},
            after_state={"value": new_value},
        )

    def log_data_access(
        self,
        sensor_id: int,
        action: str,
        user_id: Optional[str] = None,
        n_records: int = 0,
    ) -> None:
        """Registra acceso a datos de sensor (ISO 27001 A.9.4.2)."""
        self.log_event(
            event_type="data_access",
            user_id=user_id,
            action=action,
            resource=f"sensor_{sensor_id}",
            details={"n_records": n_records},
        )


class NullAuditLogger(AuditPort):
    """Audit logger no-op para testing y cuando auditoría está desactivada."""

    def log_event(self, event_type: str, action: str, resource: str,
                  details: Dict[str, Any], result: str = "success",
                  user_id: Optional[str] = None,
                  before_state: Optional[Dict[str, Any]] = None,
                  after_state: Optional[Dict[str, Any]] = None) -> None:
        pass

    def log_prediction(self, sensor_id: int, predicted_value: float,
                       confidence: float, engine_name: str,
                       trace_id: Optional[str] = None) -> None:
        pass

    def log_anomaly(self, sensor_id: int, value: float, score: float,
                    explanation: str, trace_id: Optional[str] = None) -> None:
        pass

    def log_config_change(self, config_key: str, old_value: Any,
                          new_value: Any, user_id: str) -> None:
        pass
