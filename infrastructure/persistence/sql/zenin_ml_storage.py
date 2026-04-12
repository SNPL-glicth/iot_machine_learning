"""Storage adapter para esquema zenin_ml (Zenin canonical).

Escribe predicciones a zenin_ml.predictions usando:
- SeriesId: UUID (determinístico desde sensor_id + tenant_id)
- TenantId: UUID
- ModelId: UUID (determinístico desde engine_name + series_id)

Este adapter coexiste con SqlServerStorageAdapter (dbo.predictions legacy).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID, uuid5, NAMESPACE_OID

from sqlalchemy import text
from sqlalchemy.engine import Connection

from iot_machine_learning.domain.entities.prediction import Prediction

logger = logging.getLogger(__name__)

# Namespaces fijos para generación determinística de UUIDs
_SERIES_NAMESPACE = uuid5(NAMESPACE_OID, "zenin.series")
_MODEL_NAMESPACE = uuid5(NAMESPACE_OID, "zenin.model")
_DEFAULT_TENANT = UUID("00000000-0000-0000-0000-000000000001")  # Tenant por defecto


def _sensor_id_to_series_id(sensor_id: int, tenant_id: UUID) -> UUID:
    """Convierte sensor_id:int a series_id:UUID de forma determinística.
    
    Args:
        sensor_id: ID legacy del sensor
        tenant_id: UUID del tenant
    
    Returns:
        UUID determinístico para la serie
    """
    composite_key = f"{tenant_id}:{sensor_id}"
    return uuid5(_SERIES_NAMESPACE, composite_key)


def _engine_series_to_model_id(engine_name: str, series_id: UUID) -> UUID:
    """Convierte engine_name + series_id a model_id:UUID determinístico.
    
    Args:
        engine_name: Nombre del motor ML
        series_id: UUID de la serie
    
    Returns:
        UUID determinístico para el modelo
    """
    composite_key = f"{engine_name}:{series_id}"
    return uuid5(_MODEL_NAMESPACE, composite_key)


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convierte a float con guard contra None/NaN/Inf."""
    if value is None:
        return default
    try:
        import math
        f = float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


class ZeninMLStorageAdapter:
    """Adapter para escribir predicciones a zenin_ml.predictions.
    
    Características:
    - Usa UUIDs determinísticos (sensor_id → series_id)
    - Escribe a esquema zenin_ml (no dbo legacy)
    - Compatible con arquitectura Zenin canonical
    - Coexiste con SqlServerStorageAdapter legacy
    """
    
    def __init__(self, conn: Connection, tenant_id: Optional[UUID] = None):
        """Inicializa adapter con conexión SQL.
        
        Args:
            conn: Conexión SQLAlchemy
            tenant_id: UUID del tenant (usa default si None)
        """
        self._conn = conn
        self._tenant_id = tenant_id or _DEFAULT_TENANT
    
    def save_prediction(
        self,
        prediction: Prediction,
        *,
        horizon_minutes_per_step: int = 10,
    ) -> UUID:
        """Persiste una predicción en zenin_ml.predictions.
        
        Args:
            prediction: Predicción del dominio
            horizon_minutes_per_step: Minutos por paso de horizonte
        
        Returns:
            UUID del registro insertado
        """
        # 1. Convertir sensor_id legacy → series_id UUID
        sensor_id = int(prediction.series_id)  # legacy series_id es string de int
        series_id = _sensor_id_to_series_id(sensor_id, self._tenant_id)
        
        # 2. Generar model_id determinístico
        model_id = _engine_series_to_model_id(prediction.engine_name, series_id)
        
        # 3. Calcular target timestamp
        target_ts = datetime.now(timezone.utc) + timedelta(
            minutes=prediction.horizon_steps * horizon_minutes_per_step
        )
        
        # 4. Extraer metadata
        meta = prediction.metadata or {}
        is_anomaly = bool(meta.get("is_anomaly", False))
        anomaly_score = _safe_float(meta.get("anomaly_score"), default=0.0) if meta.get("anomaly_score") is not None else None
        
        # 5. Determinar risk_level
        risk_level = str(meta.get("risk_level", "NONE"))
        if risk_level not in ("NONE", "LOW", "MEDIUM", "HIGH"):
            risk_level = "NONE"
        
        # 6. Construir explanation
        explanation = str(meta.get("explanation", "")) or None
        
        # 7. Serializar metadata completa como JSON
        explanation_json = json.dumps(meta, default=str) if meta else None
        metadata_json = json.dumps({
            "sensor_id": sensor_id,  # Trazabilidad legacy
            "horizon_steps": prediction.horizon_steps,
            "window_size": meta.get("window_size"),
            "regime": meta.get("regime"),
        }, default=str)
        
        # 8. Generar audit_trace_id si no existe
        audit_trace_id = prediction.audit_trace_id
        if audit_trace_id is None:
            audit_trace_id = uuid5(NAMESPACE_OID, f"pred:{series_id}:{datetime.now(timezone.utc).isoformat()}")
        
        # 9. Insertar en zenin_ml.predictions
        try:
            row = self._conn.execute(
                text(
                    """
                    INSERT INTO zenin_ml.predictions (
                        Id, TenantId, SeriesId, ModelId,
                        PredictedValue, ConfidenceScore, ConfidenceLevel,
                        Trend, HorizonSteps,
                        ConfidenceIntervalLower, ConfidenceIntervalUpper,
                        TargetTimestamp, PredictedAt,
                        IsAnomaly, AnomalyScore,
                        RiskLevel, Explanation, ExplanationJson,
                        EngineName, Metadata, AuditTraceId
                    )
                    OUTPUT INSERTED.Id
                    VALUES (
                        NEWID(), :tenant_id, :series_id, :model_id,
                        :predicted_value, :confidence_score, :confidence_level,
                        :trend, :horizon_steps,
                        :conf_lower, :conf_upper,
                        :target_timestamp, SYSDATETIMEOFFSET(),
                        :is_anomaly, :anomaly_score,
                        :risk_level, :explanation, :explanation_json,
                        :engine_name, :metadata, :audit_trace_id
                    )
                    """
                ),
                {
                    "tenant_id": str(self._tenant_id),
                    "series_id": str(series_id),
                    "model_id": str(model_id),
                    "predicted_value": float(prediction.predicted_value),
                    "confidence_score": _safe_float(prediction.confidence_score, 0.5),
                    "confidence_level": prediction.confidence_level.value if prediction.confidence_level else "medium",
                    "trend": prediction.trend or "stable",
                    "horizon_steps": prediction.horizon_steps,
                    "conf_lower": _safe_float(prediction.confidence_interval[0]) if prediction.confidence_interval else None,
                    "conf_upper": _safe_float(prediction.confidence_interval[1]) if prediction.confidence_interval else None,
                    "target_timestamp": target_ts,
                    "is_anomaly": is_anomaly,
                    "anomaly_score": anomaly_score,
                    "risk_level": risk_level,
                    "explanation": explanation,
                    "explanation_json": explanation_json,
                    "engine_name": prediction.engine_name,
                    "metadata": metadata_json,
                    "audit_trace_id": str(audit_trace_id),
                },
            ).fetchone()
            
            inserted_id = UUID(row[0]) if row else None
            
            logger.info(
                "zenin_ml_prediction_saved",
                extra={
                    "sensor_id": sensor_id,
                    "series_id": str(series_id),
                    "model_id": str(model_id),
                    "predicted_value": prediction.predicted_value,
                    "engine": prediction.engine_name,
                    "audit_trace_id": str(audit_trace_id),
                },
            )
            
            return inserted_id
        
        except Exception as exc:
            logger.error(
                "zenin_ml_prediction_save_failed",
                extra={
                    "sensor_id": sensor_id,
                    "series_id": str(series_id),
                    "error": str(exc),
                },
            )
            raise
