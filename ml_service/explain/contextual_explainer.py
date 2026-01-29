"""Módulo de Explicación Contextual para ML.

REFACTORIZADO 2026-01-29:
- Modelos extraídos a models/
- Servicios extraídos a services/
- Este archivo ahora es solo el orquestador (~150 líneas, antes 666)

Estructura:
- models/enriched_context.py: EnrichedContext dataclass
- models/explanation_result.py: ExplanationResult dataclass
- services/data_loader.py: Acceso a datos
- services/template_generator.py: Generación de templates
- services/ai_client.py: Cliente AI Explainer
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.engine import Connection

from .models import EnrichedContext, ExplanationResult
from .services import ExplainerDataLoader, TemplateExplanationGenerator, AIExplainerClient

logger = logging.getLogger(__name__)


class ContextualExplainer:
    """Explainer contextual que enriquece la información antes de generar explicaciones.
    
    ESTRATEGIA:
    1. Recopilar contexto completo (sensor, dispositivo, historial, correlaciones)
    2. Intentar llamar al AI Explainer con contexto enriquecido
    3. Si falla, usar templates basados en reglas
    4. Siempre devolver una explicación útil
    """
    
    def __init__(self, conn: Connection):
        self._conn = conn
        self._data_loader = ExplainerDataLoader(conn)
        self._template_generator = TemplateExplanationGenerator()
        self._ai_client = AIExplainerClient()
    
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
        sensor_info = self._data_loader.get_sensor_info(sensor_id)
        
        # Obtener valor actual
        current_value = self._data_loader.get_current_value(sensor_id)
        
        # Obtener umbrales del usuario
        thresholds = self._data_loader.get_user_thresholds(sensor_id)
        
        # Obtener estadísticas recientes
        recent_stats = self._data_loader.get_recent_stats(sensor_id)
        
        # Obtener eventos correlacionados
        correlated = self._data_loader.get_correlated_events(sensor_id)
        
        # Obtener historial de eventos similares
        history = self._data_loader.get_similar_events_history(sensor_id, is_anomaly)
        
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
            result = await self._ai_client.explain_async(context)
            if result:
                return result
        except Exception as e:
            logger.warning(
                "[CONTEXTUAL_EXPLAINER] AI Explainer failed for sensor_id=%s: %s",
                context.sensor_id, str(e)
            )
        
        # Fallback a templates
        return self._template_generator.generate(context)
    
    def explain_sync(self, context: EnrichedContext) -> ExplanationResult:
        """Genera explicación de forma síncrona (solo templates, sin LLM)."""
        return self._template_generator.generate(context)


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
