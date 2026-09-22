"""AlertSuppressor — evalúa si una alerta debe emitirse o suprimirse.

Reglas de supresión con overrides de seguridad:
1. ESCALATION OVERRIDE (nunca suprimir)
2. CRITICALITY OVERRIDE (nunca suprimir)
3. PRIORITY ESCALATION (nunca suprimir)
4. COOLDOWN (suprimir si idéntica dentro de ventana)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from ...entities.decision import Decision, DecisionContext
from ...ports.alert_state_repository_port import (
    AlertStateRepositoryPort,
    InMemoryAlertStateRepository,
)


@dataclass(frozen=True)
class SuppressionResult:
    """Resultado de evaluación de supresión."""
    should_emit: bool
    reason: str
    suppressed_count: int


class AlertSuppressor:
    """Evalúa si una alerta debe suprimirse según reglas contextuales.

    Stateless en lógica de decisión; delega la persistencia de estado a AlertStateRepositoryPort.
    """

    def __init__(
        self,
        state_repository: Optional[AlertStateRepositoryPort] = None,
        escalation_threshold: int = 5,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize suppressor.

        Args:
            state_repository: Port for alert suppression state persistence.
            escalation_threshold: Number of consecutive anomalies before escalation override.
        """
        if state_repository is not None:
            self._state_repository = state_repository
        elif "state_store" in kwargs and isinstance(kwargs["state_store"], AlertStateRepositoryPort):
            self._state_repository = kwargs["state_store"]
        else:
            self._state_repository = InMemoryAlertStateRepository()
        self._escalation_threshold = escalation_threshold

    def _get_escalation_threshold(self) -> int:
        """Obtener umbral de escalación."""
        return self._escalation_threshold

    def get_suppressed_count(self, series_id: str) -> int:
        """Obtener contador de alertas suprimidas para una serie."""
        return self._state_repository.get_suppressed_count(series_id)

    def evaluate(
        self,
        context: DecisionContext,
        decision: Decision,
    ) -> SuppressionResult:
        """Evaluar si la alerta debe emitirse o suprimirse.
        
        Args:
            context: Contexto enriquecido
            decision: Decisión calculada
        
        Returns:
            SuppressionResult con should_emit, reason, suppressed_count
        """
        series_id = context.series_id
        severity = context.severity.severity if context.severity else "info"
        
        # REGLA 1: ESCALATION OVERRIDE
        # consecutive_anomalies > threshold → nunca suprimir
        if context.consecutive_anomalies > self._get_escalation_threshold():
            self._state_repository.save_alert(series_id, decision.action, decision.priority, severity)
            return SuppressionResult(
                should_emit=True,
                reason="escalation_override",
                suppressed_count=0,
            )
        
        # REGLA 2: CRITICALITY OVERRIDE
        # severity == CRITICAL → nunca suprimir
        if severity.upper() == "CRITICAL":
            self._state_repository.save_alert(series_id, decision.action, decision.priority, severity)
            return SuppressionResult(
                should_emit=True,
                reason="criticality_override",
                suppressed_count=0,
            )
        
        # Recuperar última alerta
        last_alert = self._state_repository.get_last_alert(series_id)
        now = time.time()
        
        # REGLA 3: PRIORITY ESCALATION
        # Si nueva priority < última priority (más urgente) → emitir
        if last_alert is not None:
            last_priority = last_alert.get("priority", 5)
            if decision.priority < last_priority:
                self._state_repository.save_alert(series_id, decision.action, decision.priority, severity)
                return SuppressionResult(
                    should_emit=True,
                    reason="priority_escalation",
                    suppressed_count=0,
                )
        
        # REGLA 4: COOLDOWN
        # Si última alerta existe y está dentro de ventana y misma acción
        window_seconds = context.suppression_window_minutes * 60
        
        if last_alert is not None:
            last_time = last_alert.get("timestamp", 0)
            last_action = last_alert.get("action", "")
            time_diff = now - last_time
            
            if time_diff < window_seconds and decision.action == last_action:
                # Suprimir esta alerta
                suppressed_count = self._state_repository.increment_suppressed(series_id)
                return SuppressionResult(
                    should_emit=False,
                    reason="cooldown",
                    suppressed_count=suppressed_count,
                )
        
        # DEFAULT: Emitir
        self._state_repository.save_alert(series_id, decision.action, decision.priority, severity)
        return SuppressionResult(
            should_emit=True,
            reason="default_emit",
            suppressed_count=0,
        )
