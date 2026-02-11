"""Port de auditoría — contrato para logging conforme a ISO 27001.

ISO 27001 Anexo A.12.4: Logging y monitoreo.
Todo evento ML auditable pasa por este port.
La implementación concreta decide el destino (archivo, BD, SIEM, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class AuditPort(ABC):
    """Contrato para audit logging ISO 27001.

    Eventos auditables:
    - Predicciones generadas (A.12.4.1)
    - Anomalías detectadas (A.12.4.1)
    - Cambios de configuración (A.12.4.3)
    - Acceso a datos de sensores (A.9.4.2)
    - Decisiones automatizadas (A.12.4.1)
    """

    @abstractmethod
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
            event_type: Categoría (``"prediction"``, ``"anomaly_detection"``,
                ``"config_change"``, ``"data_access"``).
            action: Acción realizada (``"predict"``, ``"detect"``, ``"update"``).
            resource: Recurso afectado (``"sensor_123"``, ``"model_taylor"``).
            details: Información adicional del evento.
            result: Resultado (``"success"``, ``"failure"``, ``"warning"``).
            user_id: ID del usuario/proceso que ejecutó la acción.
            before_state: Estado previo (para cambios de configuración).
            after_state: Estado nuevo (para cambios de configuración).
        """
        ...

    @abstractmethod
    def log_prediction(
        self,
        sensor_id: int,
        predicted_value: float,
        confidence: float,
        engine_name: str,
        trace_id: Optional[str] = None,
    ) -> None:
        """Helper: registra predicción generada."""
        ...

    @abstractmethod
    def log_anomaly(
        self,
        sensor_id: int,
        value: float,
        score: float,
        explanation: str,
        trace_id: Optional[str] = None,
    ) -> None:
        """Helper: registra detección de anomalía."""
        ...

    @abstractmethod
    def log_config_change(
        self,
        config_key: str,
        old_value: Any,
        new_value: Any,
        user_id: str,
    ) -> None:
        """Helper: registra cambio de configuración."""
        ...
