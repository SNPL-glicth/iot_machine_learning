"""AlertSuppressor — evalúa si una alerta debe emitirse o suprimirse.

Reglas de supresión con overrides de seguridad:
1. ESCALATION OVERRIDE (nunca suprimir)
2. CRITICALITY OVERRIDE (nunca suprimir)
3. PRIORITY ESCALATION (nunca suprimir)
4. COOLDOWN (suprimir si idéntica dentro de ventana)

Estado de última alerta en Redis:
  key: last_alert:{series_id} → JSON {action, priority, timestamp, severity}
  TTL: 1 hora
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...infrastructure.redis_keys import RedisKeys
from ..entities.decision import Decision, DecisionContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SuppressionResult:
    """Resultado de evaluación de supresión."""
    should_emit: bool
    reason: str
    suppressed_count: int


class AlertSuppressor:
    """Evalúa si una alerta debe suprimirse según reglas contextuales.
    
    Stateless: recibe todo el estado necesario como parámetros.
    El estado de última alerta se guarda externamente en Redis.
    
    Feature flag: ML_DECISION_SUPPRESSION_WINDOW_MINUTES (leído en cada evaluación)
    """
    
    def __init__(
        self,
        redis_client: Optional[Any] = None,
    ) -> None:
        """Initialize suppressor.

        Args:
            redis_client: Redis client para estado de alertas (optional)
        """
        self._redis = redis_client

    def _get_key_ttl(self) -> int:
        """Leer TTL de keys desde flags (hot-reload)."""
        try:
            from ...ml_service.config.feature_flags import get_feature_flags
            return int(get_feature_flags().ML_ANOMALY_KEY_TTL_SECONDS)
        except Exception:
            return 3600  # fallback: 1 hora
    
    def _get_window_minutes(self) -> float:
        """Leer ventana de supresión desde flags (hot-reload)."""
        try:
            from ...ml_service.config.feature_flags import get_feature_flags
            flags = get_feature_flags()
            return flags.ML_DECISION_SUPPRESSION_WINDOW_MINUTES
        except Exception:
            return 5.0  # fallback seguro

    def _get_escalation_threshold(self) -> int:
        """Leer umbral de escalación desde flags (hot-reload)."""
        try:
            from ...ml_service.config.feature_flags import get_feature_flags
            return int(get_feature_flags().ML_DECISION_ESCALATION_THRESHOLD)
        except Exception:
            return 5  # fallback
    
    def _get_redis_key(self, series_id: str) -> str:
        """Generar key para última alerta."""
        return RedisKeys.last_alert(series_id)
    
    def _get_last_alert(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Recuperar última alerta desde Redis."""
        if self._redis is None:
            return None
        
        try:
            key = self._get_redis_key(series_id)
            data = self._redis.get(key)
            if data is None:
                return None
            
            # Parse JSON
            json_str = data.decode() if isinstance(data, bytes) else data
            return json.loads(json_str)
        except Exception as e:
            logger.warning(
                "redis_get_last_alert_failed",
                extra={"series_id": series_id, "error": str(e)},
            )
            return None
    
    def _save_alert(
        self,
        series_id: str,
        action: str,
        priority: int,
        severity: str,
    ) -> None:
        """Guardar alerta emitida en Redis."""
        if self._redis is None:
            return
        
        try:
            key = self._get_redis_key(series_id)
            value = json.dumps({
                "action": action,
                "priority": priority,
                "timestamp": time.time(),
                "severity": severity,
            })
            self._redis.setex(key, self._get_key_ttl(), value)
        except Exception as e:
            logger.warning(
                "redis_save_alert_failed",
                extra={"series_id": series_id, "error": str(e)},
            )
    
    def _increment_suppressed(self, series_id: str) -> int:
        """Incrementar contador de suprimidas. Retorna nuevo valor."""
        if self._redis is None:
            return 0

        try:
            key = RedisKeys.suppressed(series_id)
            count = self._redis.incr(key)
            # Refrescar TTL
            self._redis.expire(key, self._get_key_ttl())
            return int(count) if count else 0
        except Exception as e:
            logger.warning(
                "redis_increment_suppressed_failed",
                extra={"series_id": series_id, "error": str(e)},
            )
            return 0
    
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
            self._save_alert(series_id, decision.action, decision.priority, severity)
            return SuppressionResult(
                should_emit=True,
                reason="escalation_override",
                suppressed_count=0,
            )
        
        # REGLA 2: CRITICALITY OVERRIDE
        # severity == CRITICAL → nunca suprimir
        if severity.upper() == "CRITICAL":
            self._save_alert(series_id, decision.action, decision.priority, severity)
            return SuppressionResult(
                should_emit=True,
                reason="criticality_override",
                suppressed_count=0,
            )
        
        # Recuperar última alerta
        last_alert = self._get_last_alert(series_id)
        now = time.time()
        
        # REGLA 3: PRIORITY ESCALATION
        # Si nueva priority < última priority (más urgente) → emitir
        if last_alert is not None:
            last_priority = last_alert.get("priority", 5)
            if decision.priority < last_priority:
                self._save_alert(series_id, decision.action, decision.priority, severity)
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
                suppressed_count = self._increment_suppressed(series_id)
                return SuppressionResult(
                    should_emit=False,
                    reason="cooldown",
                    suppressed_count=suppressed_count,
                )
        
        # DEFAULT: Emitir
        self._save_alert(series_id, decision.action, decision.priority, severity)
        return SuppressionResult(
            should_emit=True,
            reason="default_emit",
            suppressed_count=0,
        )
    
    def get_suppressed_count(self, series_id: str) -> int:
        """Obtener contador de alertas suprimidas para una serie."""
        if self._redis is None:
            return 0
        
        try:
            key = f"suppressed:{series_id}"
            value = self._redis.get(key)
            if value is None:
                return 0
            return int(value.decode() if isinstance(value, bytes) else value)
        except Exception:
            return 0
