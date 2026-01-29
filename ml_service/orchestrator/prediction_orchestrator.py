"""Orquestador de Predicciones ML.

REFACTORIZADO 2026-01-29:
- Modelos extraídos a models/
- Servicios extraídos a services/
- Este archivo ahora es solo el orquestador (~200 líneas, antes 542)

Estructura:
- models/enriched_prediction.py: EnrichedPrediction dataclass
- services/payload_builder.py: Constructor de payload enriquecido
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.engine import Connection

from iot_machine_learning.ml_service.context.decision_context import (
    DecisionContext,
    DecisionContextBuilder,
)
from iot_machine_learning.ml_service.context.operational_context import (
    OperationalContext,
    OperationalContextBuilder,
    adjust_severity_with_context,
)
from iot_machine_learning.ml_service.correlation.sensor_correlator import (
    CorrelationResult,
    SensorCorrelator,
)
from iot_machine_learning.ml_service.explain.contextual_explainer import (
    ExplanationResult,
    ContextualExplainer,
    create_contextual_explanation,
)
from iot_machine_learning.ml_service.memory.decision_memory import (
    DecisionMemory,
    HistoricalInsight,
    record_ml_decision,
)
from iot_machine_learning.ml_service.repository.sensor_repository import (
    SensorMetadata,
    load_sensor_metadata,
    get_device_id_for_sensor,
)

from .models import EnrichedPrediction
from .services import EnrichedPayloadBuilder

logger = logging.getLogger(__name__)


class PredictionOrchestrator:
    """Orquestador que integra todos los módulos de contexto.
    
    Este es el punto de entrada principal para generar predicciones
    enriquecidas con todo el contexto disponible.
    """
    
    def __init__(self, conn: Connection):
        self._conn = conn
        self._decision_builder = DecisionContextBuilder(conn)
        self._operational_builder = OperationalContextBuilder(conn)
        self._correlator = SensorCorrelator(conn)
        self._memory = DecisionMemory(conn)
        self._explainer = ContextualExplainer(conn)
        self._payload_builder = EnrichedPayloadBuilder()
    
    def enrich_prediction(
        self,
        *,
        sensor_id: int,
        predicted_value: float,
        current_value: float,
        trend: str,
        confidence: float,
        anomaly_score: float,
        is_anomaly: bool,
        base_severity: str,
        risk_level: str,
        horizon_minutes: int = 10,
    ) -> EnrichedPrediction:
        """Enriquece una predicción con todo el contexto disponible."""
        
        # 1. Cargar metadata del sensor
        sensor_meta, device_id = self._load_sensor_metadata(sensor_id)
        
        # 2. Obtener correlaciones
        correlation_result = self._get_correlations(sensor_id)
        
        # 3. Consultar historial de decisiones similares
        historical_insight = self._get_historical_insight(
            sensor_meta.sensor_type, base_severity, trend, is_anomaly
        )
        
        # 4. Ajustar severidad por contexto operacional
        adjusted_severity, operational_context = self._adjust_severity(
            sensor_id, device_id, base_severity
        )
        
        # 5. Generar contexto de decisión accionable
        decision_context = self._build_decision_context(
            sensor_id, device_id, sensor_meta, predicted_value, current_value,
            trend, confidence, horizon_minutes, adjusted_severity, risk_level,
            is_anomaly, anomaly_score
        )
        
        # 6. Crear explicación contextual
        explanation_result = self._create_explanation(
            sensor_id, predicted_value, trend, confidence,
            horizon_minutes, is_anomaly, anomaly_score
        )
        
        # 7. Construir payload enriquecido
        enriched_payload = self._payload_builder.build(
            sensor_meta=sensor_meta,
            device_id=device_id,
            predicted_value=predicted_value,
            current_value=current_value,
            trend=trend,
            confidence=confidence,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            severity=adjusted_severity,
            risk_level=risk_level,
            horizon_minutes=horizon_minutes,
            historical_insight=historical_insight,
            correlation_result=correlation_result,
            operational_context=operational_context,
        )
        
        # 8. Registrar decisión en memoria
        self._record_decision(
            sensor_id, device_id, sensor_meta, adjusted_severity,
            is_anomaly, anomaly_score, predicted_value, trend, decision_context
        )
        
        return EnrichedPrediction(
            sensor_id=sensor_id,
            device_id=device_id,
            predicted_value=predicted_value,
            current_value=current_value,
            trend=trend,
            confidence=confidence,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            base_severity=base_severity,
            adjusted_severity=adjusted_severity,
            risk_level=risk_level,
            decision_context=decision_context,
            operational_context=operational_context,
            correlation_result=correlation_result,
            historical_insight=historical_insight,
            explanation_result=explanation_result,
            enriched_payload=enriched_payload,
            generated_at=datetime.now(timezone.utc),
        )
    
    def _load_sensor_metadata(self, sensor_id: int) -> tuple[SensorMetadata, int]:
        """Carga metadata del sensor."""
        try:
            sensor_meta = load_sensor_metadata(self._conn, sensor_id)
            device_id = get_device_id_for_sensor(self._conn, sensor_id)
            return sensor_meta, device_id
        except Exception as e:
            logger.warning("[ORCHESTRATOR] Failed to load sensor metadata: %s", str(e))
            return SensorMetadata(
                sensor_id=sensor_id,
                sensor_type="unknown",
                unit="",
                location="unknown",
                criticality="medium",
            ), 0
    
    def _get_correlations(self, sensor_id: int) -> Optional[CorrelationResult]:
        """Obtiene correlaciones con otros sensores."""
        try:
            return self._correlator.analyze_all_devices_for_sensor(sensor_id)
        except Exception as e:
            logger.warning("[ORCHESTRATOR] Failed to analyze correlations: %s", str(e))
            return None
    
    def _get_historical_insight(
        self, sensor_type: str, base_severity: str, trend: str, is_anomaly: bool
    ) -> Optional[HistoricalInsight]:
        """Consulta historial de decisiones similares."""
        try:
            event_code = "ANOMALY_DETECTED" if is_anomaly else "PRED_THRESHOLD_BREACH"
            return self._memory.get_historical_insight(
                sensor_type=sensor_type,
                severity=base_severity,
                trend=trend,
                event_code=event_code,
            )
        except Exception as e:
            logger.warning("[ORCHESTRATOR] Failed to get historical insight: %s", str(e))
            return None
    
    def _adjust_severity(
        self, sensor_id: int, device_id: int, base_severity: str
    ) -> tuple[str, Optional[OperationalContext]]:
        """Ajusta severidad por contexto operacional."""
        try:
            return adjust_severity_with_context(
                self._conn,
                sensor_id=sensor_id,
                device_id=device_id,
                base_severity=base_severity,
            )
        except Exception as e:
            logger.warning("[ORCHESTRATOR] Failed to adjust severity: %s", str(e))
            return base_severity, None
    
    def _build_decision_context(
        self, sensor_id: int, device_id: int, sensor_meta: SensorMetadata,
        predicted_value: float, current_value: float, trend: str,
        confidence: float, horizon_minutes: int, severity: str,
        risk_level: str, is_anomaly: bool, anomaly_score: float
    ) -> Optional[DecisionContext]:
        """Genera contexto de decisión accionable."""
        try:
            return self._decision_builder.build(
                sensor_id=sensor_id,
                device_id=device_id,
                sensor_type=sensor_meta.sensor_type,
                location=sensor_meta.location,
                predicted_value=predicted_value,
                current_value=current_value,
                trend=trend,
                confidence=confidence,
                prediction_horizon_minutes=horizon_minutes,
                severity=severity,
                risk_level=risk_level,
                anomaly_detected=is_anomaly,
                anomaly_score=anomaly_score,
            )
        except Exception as e:
            logger.warning("[ORCHESTRATOR] Failed to build decision context: %s", str(e))
            return None
    
    def _create_explanation(
        self, sensor_id: int, predicted_value: float, trend: str,
        confidence: float, horizon_minutes: int, is_anomaly: bool, anomaly_score: float
    ) -> Optional[ExplanationResult]:
        """Crea explicación contextual."""
        try:
            return create_contextual_explanation(
                self._conn,
                sensor_id=sensor_id,
                predicted_value=predicted_value,
                trend=trend,
                confidence=confidence,
                horizon_minutes=horizon_minutes,
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
            )
        except Exception as e:
            logger.warning("[ORCHESTRATOR] Failed to create explanation: %s", str(e))
            return None
    
    def _record_decision(
        self, sensor_id: int, device_id: int, sensor_meta: SensorMetadata,
        severity: str, is_anomaly: bool, anomaly_score: float,
        predicted_value: float, trend: str, decision_context: Optional[DecisionContext]
    ) -> None:
        """Registra decisión en memoria."""
        if severity.lower() not in ("warning", "critical") and not is_anomaly:
            return
        
        try:
            actions = []
            if decision_context and decision_context.recommended_actions:
                actions = [a.action for a in decision_context.recommended_actions]
            
            record_ml_decision(
                self._conn,
                sensor_id=sensor_id,
                device_id=device_id,
                event_type=severity.lower(),
                event_code="ANOMALY_DETECTED" if is_anomaly else "PRED_THRESHOLD_BREACH",
                sensor_type=sensor_meta.sensor_type,
                severity=severity,
                trend=trend,
                anomaly_score=anomaly_score,
                predicted_value=predicted_value,
                actions_taken=actions,
            )
        except Exception as e:
            logger.warning("[ORCHESTRATOR] Failed to record decision: %s", str(e))
    
    def clear_caches(self) -> None:
        """Limpia los caches de los componentes."""
        self._correlator.clear_cache()


def enrich_prediction_with_context(
    conn: Connection,
    *,
    sensor_id: int,
    predicted_value: float,
    current_value: float,
    trend: str,
    confidence: float,
    anomaly_score: float,
    is_anomaly: bool,
    base_severity: str,
    risk_level: str,
    horizon_minutes: int = 10,
) -> EnrichedPrediction:
    """Función de conveniencia para enriquecer una predicción.
    
    Esta función es el punto de entrada principal para el batch runner.
    """
    orchestrator = PredictionOrchestrator(conn)
    return orchestrator.enrich_prediction(
        sensor_id=sensor_id,
        predicted_value=predicted_value,
        current_value=current_value,
        trend=trend,
        confidence=confidence,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        base_severity=base_severity,
        risk_level=risk_level,
        horizon_minutes=horizon_minutes,
    )
