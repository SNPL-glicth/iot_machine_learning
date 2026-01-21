"""Módulo de Explicación Contextual para ML.

FALENCIA 3: El AI Explainer está desacoplado y procesa tarde, con información mínima.

Este módulo implementa:
- Integración temprana del explainer en el flujo de predicción
- Enriquecimiento del contexto antes de llamar al LLM
- Fallback a templates cuando el LLM no está disponible
- Cache de explicaciones para evitar llamadas redundantes
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Any

import httpx
from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


@dataclass
class EnrichedContext:
    """Contexto enriquecido para el explainer."""
    
    # Información del sensor
    sensor_id: int
    sensor_type: str
    sensor_name: str
    unit: str
    
    # Información del dispositivo
    device_id: int
    device_name: str
    device_type: str
    location: str
    
    # Valores y predicción
    current_value: float
    predicted_value: float
    trend: str
    confidence: float
    horizon_minutes: int
    
    # Anomalía
    is_anomaly: bool
    anomaly_score: float
    
    # Umbrales del usuario
    user_threshold_min: Optional[float]
    user_threshold_max: Optional[float]
    
    # Historial reciente
    recent_avg: Optional[float]
    recent_min: Optional[float]
    recent_max: Optional[float]
    recent_std: Optional[float]
    
    # Eventos correlacionados
    correlated_events: list[dict]
    
    # Historial de eventos similares
    similar_events_count: int
    last_similar_event_at: Optional[datetime]
    
    def to_dict(self) -> dict:
        return {
            "sensor": {
                "id": self.sensor_id,
                "type": self.sensor_type,
                "name": self.sensor_name,
                "unit": self.unit,
            },
            "device": {
                "id": self.device_id,
                "name": self.device_name,
                "type": self.device_type,
                "location": self.location,
            },
            "prediction": {
                "current_value": self.current_value,
                "predicted_value": self.predicted_value,
                "trend": self.trend,
                "confidence": self.confidence,
                "horizon_minutes": self.horizon_minutes,
            },
            "anomaly": {
                "is_anomaly": self.is_anomaly,
                "score": self.anomaly_score,
            },
            "thresholds": {
                "min": self.user_threshold_min,
                "max": self.user_threshold_max,
            },
            "recent_stats": {
                "avg": self.recent_avg,
                "min": self.recent_min,
                "max": self.recent_max,
                "std": self.recent_std,
            },
            "correlated_events": self.correlated_events,
            "history": {
                "similar_events_count": self.similar_events_count,
                "last_similar_event_at": self.last_similar_event_at.isoformat() if self.last_similar_event_at else None,
            },
        }


@dataclass
class ExplanationResult:
    """Resultado de la explicación contextual."""
    severity: str
    explanation: str
    possible_causes: list[str]
    recommended_action: str
    confidence: float
    source: str  # 'llm', 'template', 'fallback'
    generated_at: datetime
    
    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "explanation": self.explanation,
            "possible_causes": self.possible_causes,
            "recommended_action": self.recommended_action,
            "confidence": self.confidence,
            "source": self.source,
            "generated_at": self.generated_at.isoformat(),
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class ContextualExplainer:
    """Explainer contextual que enriquece la información antes de generar explicaciones.
    
    ESTRATEGIA:
    1. Recopilar contexto completo (sensor, dispositivo, historial, correlaciones)
    2. Intentar llamar al AI Explainer con contexto enriquecido
    3. Si falla, usar templates basados en reglas
    4. Siempre devolver una explicación útil
    """
    
    AI_EXPLAINER_TIMEOUT = 2.0  # segundos
    
    def __init__(self, conn: Connection):
        self._conn = conn
        self._ai_explainer_url = os.getenv("AI_EXPLAINER_URL", "http://localhost:8003")
    
    def build_enriched_context(
        self,
        *,
        sensor_id: int,
        predicted_value: float,
        trend: str,
        confidence: float,
        horizon_minutes: int,
        is_anomaly: bool,
        anomaly_score: float,
    ) -> EnrichedContext:
        """Construye contexto enriquecido para el explainer."""
        
        # Obtener información del sensor y dispositivo
        sensor_info = self._get_sensor_info(sensor_id)
        
        # Obtener valor actual
        current_value = self._get_current_value(sensor_id)
        
        # Obtener umbrales del usuario
        thresholds = self._get_user_thresholds(sensor_id)
        
        # Obtener estadísticas recientes
        recent_stats = self._get_recent_stats(sensor_id)
        
        # Obtener eventos correlacionados
        correlated = self._get_correlated_events(sensor_id)
        
        # Obtener historial de eventos similares
        history = self._get_similar_events_history(sensor_id, is_anomaly)
        
        return EnrichedContext(
            sensor_id=sensor_id,
            sensor_type=sensor_info.get("sensor_type", "unknown"),
            sensor_name=sensor_info.get("sensor_name", f"Sensor {sensor_id}"),
            unit=sensor_info.get("unit", ""),
            device_id=sensor_info.get("device_id", 0),
            device_name=sensor_info.get("device_name", "Dispositivo"),
            device_type=sensor_info.get("device_type", ""),
            location=sensor_info.get("device_name", "ubicación desconocida"),
            current_value=current_value,
            predicted_value=predicted_value,
            trend=trend,
            confidence=confidence,
            horizon_minutes=horizon_minutes,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            user_threshold_min=thresholds.get("min"),
            user_threshold_max=thresholds.get("max"),
            recent_avg=recent_stats.get("avg"),
            recent_min=recent_stats.get("min"),
            recent_max=recent_stats.get("max"),
            recent_std=recent_stats.get("std"),
            correlated_events=correlated,
            similar_events_count=history.get("count", 0),
            last_similar_event_at=history.get("last_at"),
        )
    
    async def explain_async(self, context: EnrichedContext) -> ExplanationResult:
        """Genera explicación de forma asíncrona, intentando LLM primero."""
        
        # Intentar AI Explainer
        try:
            result = await self._call_ai_explainer(context)
            if result:
                return result
        except Exception as e:
            logger.warning(
                "[CONTEXTUAL_EXPLAINER] AI Explainer failed for sensor_id=%s: %s",
                context.sensor_id, str(e)
            )
        
        # Fallback a templates
        return self._generate_template_explanation(context)
    
    def explain_sync(self, context: EnrichedContext) -> ExplanationResult:
        """Genera explicación de forma síncrona (solo templates, sin LLM)."""
        return self._generate_template_explanation(context)
    
    async def _call_ai_explainer(self, context: EnrichedContext) -> Optional[ExplanationResult]:
        """Llama al AI Explainer con contexto enriquecido."""
        
        url = f"{self._ai_explainer_url.rstrip('/')}/explain/anomaly"
        
        # Construir payload enriquecido
        payload = {
            "context": "industrial_iot_monitoring",
            "model_output": {
                "metric": context.sensor_type,
                "observed_value": context.predicted_value,
                "expected_range": self._format_expected_range(context),
                "anomaly_score": context.anomaly_score,
                "model": "sklearn_regression_iforest",
                "model_version": "1.0.0",
            },
            "enriched_context": context.to_dict(),
        }
        
        async with httpx.AsyncClient(timeout=self.AI_EXPLAINER_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        
        return ExplanationResult(
            severity=data.get("severity", "MEDIUM"),
            explanation=data.get("explanation", ""),
            possible_causes=data.get("possible_causes", []),
            recommended_action=data.get("recommended_action", ""),
            confidence=data.get("confidence", 0.5),
            source="llm",
            generated_at=datetime.now(timezone.utc),
        )
    
    def _generate_template_explanation(self, context: EnrichedContext) -> ExplanationResult:
        """Genera explicación basada en templates cuando el LLM no está disponible."""
        
        # Determinar severidad
        severity = self._determine_severity(context)
        
        # Generar explicación
        explanation = self._build_explanation_text(context, severity)
        
        # Generar causas posibles
        causes = self._generate_possible_causes(context)
        
        # Generar acción recomendada
        action = self._generate_recommended_action(context, severity)
        
        # Calcular confianza del template
        confidence = self._calculate_template_confidence(context)
        
        return ExplanationResult(
            severity=severity,
            explanation=explanation,
            possible_causes=causes,
            recommended_action=action,
            confidence=confidence,
            source="template",
            generated_at=datetime.now(timezone.utc),
        )
    
    def _determine_severity(self, context: EnrichedContext) -> str:
        """Determina la severidad basada en el contexto."""
        
        # Verificar si está fuera de umbrales del usuario
        out_of_range = False
        if context.user_threshold_min is not None:
            out_of_range = context.predicted_value < context.user_threshold_min
        if context.user_threshold_max is not None:
            out_of_range = out_of_range or context.predicted_value > context.user_threshold_max
        
        if out_of_range:
            return "CRITICAL"
        
        if context.is_anomaly and context.anomaly_score >= 0.8:
            return "HIGH"
        
        if context.is_anomaly or context.anomaly_score >= 0.5:
            return "MEDIUM"
        
        return "LOW"
    
    def _build_explanation_text(self, context: EnrichedContext, severity: str) -> str:
        """Construye el texto de explicación."""
        
        trend_desc = {
            "rising": "tendencia ascendente",
            "falling": "tendencia descendente",
            "stable": "comportamiento estable",
        }.get(context.trend, context.trend)
        
        delta = context.predicted_value - context.current_value
        delta_pct = (delta / context.current_value * 100) if context.current_value != 0 else 0
        
        parts = [
            f"El sensor {context.sensor_name} ({context.sensor_type}) en {context.location} "
            f"muestra {trend_desc}.",
        ]
        
        parts.append(
            f"Valor actual: {context.current_value:.2f} {context.unit}. "
            f"Predicción a {context.horizon_minutes} min: {context.predicted_value:.2f} {context.unit} "
            f"(cambio de {delta:+.2f}, {delta_pct:+.1f}%)."
        )
        
        if context.is_anomaly:
            parts.append(
                f"Se detectó comportamiento anómalo (score: {context.anomaly_score:.2f})."
            )
        
        if context.similar_events_count > 0:
            parts.append(
                f"Este patrón se ha observado {context.similar_events_count} veces en los últimos 30 días."
            )
        
        if context.correlated_events:
            parts.append(
                f"Hay {len(context.correlated_events)} eventos relacionados en otros sensores del mismo dispositivo."
            )
        
        return " ".join(parts)
    
    def _generate_possible_causes(self, context: EnrichedContext) -> list[str]:
        """Genera lista de causas posibles basadas en el tipo de sensor."""
        
        causes = []
        sensor_type = context.sensor_type.lower()
        
        if sensor_type == "temperature":
            if context.trend == "rising":
                causes = [
                    "Falla en sistema de climatización",
                    "Aumento de carga térmica (equipos adicionales)",
                    "Obstrucción en ventilación",
                    "Falla del sensor (descalibración)",
                ]
            else:
                causes = [
                    "Climatización en modo máximo",
                    "Pérdida de aislamiento térmico",
                    "Falla del sensor",
                ]
        
        elif sensor_type == "humidity":
            if context.trend == "rising":
                causes = [
                    "Filtración de agua",
                    "Falla en deshumidificador",
                    "Condensación por diferencia térmica",
                ]
            else:
                causes = [
                    "Ambiente muy seco",
                    "Sistema de climatización extrayendo humedad",
                ]
        
        elif sensor_type in {"power", "voltage"}:
            causes = [
                "Fluctuación en suministro eléctrico",
                "Sobrecarga en circuito",
                "Conexión defectuosa",
                "Falla en equipo conectado",
            ]
        
        elif sensor_type == "air_quality":
            causes = [
                "Ventilación insuficiente",
                "Acumulación de CO2 por ocupación",
                "Falla en sistema de renovación de aire",
            ]
        
        else:
            causes = [
                "Cambio en condiciones operativas",
                "Posible falla del sensor",
                "Variación ambiental externa",
            ]
        
        return causes[:4]  # Máximo 4 causas
    
    def _generate_recommended_action(self, context: EnrichedContext, severity: str) -> str:
        """Genera acción recomendada basada en severidad y contexto."""
        
        if severity == "CRITICAL":
            return (
                f"ACCIÓN INMEDIATA: Verificar físicamente el sensor {context.sensor_name} "
                f"en {context.location}. Revisar sistemas relacionados y notificar a supervisión."
            )
        
        if severity == "HIGH":
            return (
                f"Programar inspección prioritaria del sensor {context.sensor_name} "
                f"y sistemas asociados en {context.location} dentro de las próximas 2 horas."
            )
        
        if severity == "MEDIUM":
            return (
                f"Monitorear de cerca el sensor {context.sensor_name}. "
                f"Si la tendencia continúa, programar revisión para el próximo turno."
            )
        
        return "Continuar monitoreo normal. No se requiere acción inmediata."
    
    def _calculate_template_confidence(self, context: EnrichedContext) -> float:
        """Calcula la confianza de la explicación basada en template."""
        
        base_confidence = 0.6
        
        # Aumentar si hay historial
        if context.similar_events_count > 3:
            base_confidence += 0.1
        
        # Aumentar si hay umbrales definidos
        if context.user_threshold_min is not None or context.user_threshold_max is not None:
            base_confidence += 0.1
        
        # Aumentar si hay estadísticas recientes
        if context.recent_avg is not None:
            base_confidence += 0.05
        
        return min(0.85, base_confidence)
    
    def _format_expected_range(self, context: EnrichedContext) -> str:
        """Formatea el rango esperado para el AI Explainer."""
        
        if context.user_threshold_min is not None and context.user_threshold_max is not None:
            return f"{context.user_threshold_min}-{context.user_threshold_max}"
        if context.user_threshold_min is not None:
            return f">= {context.user_threshold_min}"
        if context.user_threshold_max is not None:
            return f"<= {context.user_threshold_max}"
        if context.recent_min is not None and context.recent_max is not None:
            return f"{context.recent_min:.2f}-{context.recent_max:.2f} (histórico)"
        return "unknown"
    
    # -------------------------------------------------------------------------
    # Métodos de acceso a datos
    # -------------------------------------------------------------------------
    
    def _get_sensor_info(self, sensor_id: int) -> dict:
        """Obtiene información del sensor y su dispositivo."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT 
                        s.id, s.sensor_type, s.name AS sensor_name, s.unit,
                        d.id AS device_id, d.name AS device_name, d.device_type
                    FROM dbo.sensors s
                    JOIN dbo.devices d ON d.id = s.device_id
                    WHERE s.id = :sensor_id
                """),
                {"sensor_id": sensor_id},
            ).fetchone()
            
            if row:
                return {
                    "sensor_type": str(row.sensor_type or "unknown"),
                    "sensor_name": str(row.sensor_name or f"Sensor {sensor_id}"),
                    "unit": str(row.unit or ""),
                    "device_id": int(row.device_id),
                    "device_name": str(row.device_name or "Dispositivo"),
                    "device_type": str(row.device_type or ""),
                }
        except Exception:
            pass
        
        return {}
    
    def _get_current_value(self, sensor_id: int) -> float:
        """Obtiene el valor actual del sensor."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT TOP 1 value
                    FROM dbo.sensor_readings
                    WHERE sensor_id = :sensor_id
                    ORDER BY timestamp DESC
                """),
                {"sensor_id": sensor_id},
            ).fetchone()
            
            if row and row[0] is not None:
                return float(row[0])
        except Exception:
            pass
        
        return 0.0
    
    def _get_user_thresholds(self, sensor_id: int) -> dict:
        """Obtiene los umbrales definidos por el usuario."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT threshold_value_min, threshold_value_max
                    FROM dbo.alert_thresholds
                    WHERE sensor_id = :sensor_id
                      AND is_active = 1
                      AND condition_type = 'out_of_range'
                    ORDER BY id ASC
                """),
                {"sensor_id": sensor_id},
            ).fetchone()
            
            if row:
                return {
                    "min": float(row[0]) if row[0] is not None else None,
                    "max": float(row[1]) if row[1] is not None else None,
                }
        except Exception:
            pass
        
        return {}
    
    def _get_recent_stats(self, sensor_id: int, hours: int = 24) -> dict:
        """Obtiene estadísticas de las últimas horas."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT 
                        AVG(value) AS avg_val,
                        MIN(value) AS min_val,
                        MAX(value) AS max_val,
                        STDEV(value) AS std_val
                    FROM dbo.sensor_readings
                    WHERE sensor_id = :sensor_id
                      AND timestamp >= DATEADD(hour, -:hours, GETDATE())
                """),
                {"sensor_id": sensor_id, "hours": hours},
            ).fetchone()
            
            if row and row.avg_val is not None:
                return {
                    "avg": float(row.avg_val),
                    "min": float(row.min_val) if row.min_val else None,
                    "max": float(row.max_val) if row.max_val else None,
                    "std": float(row.std_val) if row.std_val else None,
                }
        except Exception:
            pass
        
        return {}
    
    def _get_correlated_events(self, sensor_id: int, minutes: int = 30) -> list[dict]:
        """Obtiene eventos de sensores correlacionados (mismo dispositivo)."""
        try:
            rows = self._conn.execute(
                text("""
                    SELECT 
                        e.id, e.sensor_id, s.sensor_type, e.event_type, e.title
                    FROM dbo.ml_events e
                    JOIN dbo.sensors s ON s.id = e.sensor_id
                    WHERE s.device_id = (SELECT device_id FROM dbo.sensors WHERE id = :sensor_id)
                      AND e.sensor_id != :sensor_id
                      AND e.created_at >= DATEADD(minute, -:minutes, GETDATE())
                      AND e.status IN ('active', 'acknowledged')
                """),
                {"sensor_id": sensor_id, "minutes": minutes},
            ).fetchall()
            
            return [
                {
                    "event_id": int(r.id),
                    "sensor_id": int(r.sensor_id),
                    "sensor_type": str(r.sensor_type or ""),
                    "event_type": str(r.event_type or ""),
                    "title": str(r.title or ""),
                }
                for r in rows
            ]
        except Exception:
            pass
        
        return []
    
    def _get_similar_events_history(self, sensor_id: int, is_anomaly: bool) -> dict:
        """Obtiene historial de eventos similares."""
        try:
            event_code = "ANOMALY_DETECTED" if is_anomaly else "PRED_THRESHOLD_BREACH"
            
            row = self._conn.execute(
                text("""
                    SELECT 
                        COUNT(*) AS cnt,
                        MAX(created_at) AS last_at
                    FROM dbo.ml_events
                    WHERE sensor_id = :sensor_id
                      AND event_code = :event_code
                      AND created_at >= DATEADD(day, -30, GETDATE())
                """),
                {"sensor_id": sensor_id, "event_code": event_code},
            ).fetchone()
            
            if row and row.cnt > 0:
                return {
                    "count": int(row.cnt),
                    "last_at": row.last_at,
                }
        except Exception:
            pass
        
        return {"count": 0, "last_at": None}


def create_contextual_explanation(
    conn: Connection,
    *,
    sensor_id: int,
    predicted_value: float,
    trend: str,
    confidence: float,
    horizon_minutes: int,
    is_anomaly: bool,
    anomaly_score: float,
) -> ExplanationResult:
    """Función de conveniencia para crear una explicación contextual síncrona.
    
    Esta función es el punto de entrada principal para el batch runner.
    """
    explainer = ContextualExplainer(conn)
    context = explainer.build_enriched_context(
        sensor_id=sensor_id,
        predicted_value=predicted_value,
        trend=trend,
        confidence=confidence,
        horizon_minutes=horizon_minutes,
        is_anomaly=is_anomaly,
        anomaly_score=anomaly_score,
    )
    return explainer.explain_sync(context)
