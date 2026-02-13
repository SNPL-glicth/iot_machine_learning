"""Port de auditoría — contrato para logging conforme a ISO 27001.

ISO 27001 Anexo A.12.4: Logging y monitoreo.
Todo evento ML auditable pasa por este port.
La implementación concreta decide el destino (archivo, BD, SIEM, etc.).

Dual interface:
- Métodos ``sensor_id: int`` (legacy IoT) — abstractos.
- Métodos ``series_id: str`` (agnósticos) — default bridge a legacy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..validators.input_guard import safe_series_id_to_int


class AuditPort(ABC):
    """Contrato para audit logging ISO 27001.

    Eventos auditables:
    - Predicciones generadas (A.12.4.1)
    - Anomalías detectadas (A.12.4.1)
    - Cambios de configuración (A.12.4.3)
    - Acceso a datos de sensores (A.9.4.2)
    - Decisiones automatizadas (A.12.4.1)

    Dual interface:
        - ``log_prediction(sensor_id: int, ...)`` — legacy IoT.
        - ``log_series_prediction(series_id: str, ...)`` — agnóstico.
          Default bridge: convierte ``series_id`` a ``int`` y delega.
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

    # ── Legacy IoT helpers (sensor_id: int) ───────────────────────────

    @abstractmethod
    def log_prediction(
        self,
        sensor_id: int,
        predicted_value: float,
        confidence: float,
        engine_name: str,
        trace_id: Optional[str] = None,
    ) -> None:
        """Helper: registra predicción generada (legacy sensor_id)."""
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
        """Helper: registra detección de anomalía (legacy sensor_id)."""
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

    # ── Agnostic helpers (series_id: str) ─────────────────────────────
    # Default bridges convert series_id → sensor_id and delegate.
    # Implementations may override for native series_id support.

    def log_series_prediction(
        self,
        series_id: str,
        predicted_value: float,
        confidence: float,
        engine_name: str,
        trace_id: Optional[str] = None,
    ) -> None:
        """Registra predicción generada (agnóstico).

        Default bridge: convierte a ``int`` y delega a ``log_prediction``.

        Args:
            series_id: Identificador de la serie.
            predicted_value: Valor predicho.
            confidence: Confianza (0–1).
            engine_name: Motor que generó la predicción.
            trace_id: ID de trazabilidad.
        """
        sensor_id = safe_series_id_to_int(series_id)
        self.log_prediction(sensor_id, predicted_value, confidence,
                            engine_name, trace_id)

    def log_series_anomaly(
        self,
        series_id: str,
        value: float,
        score: float,
        explanation: str,
        trace_id: Optional[str] = None,
    ) -> None:
        """Registra detección de anomalía (agnóstico).

        Default bridge: convierte a ``int`` y delega a ``log_anomaly``.

        Args:
            series_id: Identificador de la serie.
            value: Valor que disparó la anomalía.
            score: Anomaly score (0–1).
            explanation: Explicación legible.
            trace_id: ID de trazabilidad.
        """
        sensor_id = safe_series_id_to_int(series_id)
        self.log_anomaly(sensor_id, value, score, explanation, trace_id)
