from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Connection

# Imports de infraestructura compartida (BD)
from iot_ingest_services.common.db import get_engine

# Imports internos de ML (ahora en iot_machine_learning)
from iot_machine_learning.ml_service.config.ml_config import GlobalMLConfig, RegressionConfig
from iot_machine_learning.ml_service.explain.explanation_builder import (
    PredictionExplanation,
    build_explanation_text,
)
from iot_machine_learning.ml_service.models.regression_model import Trend, compute_trend
from iot_machine_learning.ml_service.repository.sensor_repository import (
    SensorSeries,
    SensorMetadata,
    list_active_sensors,
    load_sensor_series,
    get_device_id_for_sensor,
    load_sensor_metadata,
)
from iot_machine_learning.ml_service.trainers.regression_trainer import (
    train_regression_for_sensor,
    predict_future_value,
    predict_future_value_clamped,
)
from iot_machine_learning.ml_service.trainers.isolation_trainer import IsolationForestTrainer

# FIX CRÍTICO: Importar validación de estado operacional
# El ML NO puede generar eventos si el sensor está en INITIALIZING o STALE
from iot_ingest_services.ingest_api.sensor_state import SensorStateManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerConfig:
    interval_seconds: float
    once: bool
    dedupe_minutes: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iter_sensors(conn: Connection) -> Iterable[int]:
    return list_active_sensors(conn)


# ---------------------------------------------------------------------------
# Helpers SQL: escribir en predictions y ml_events según el contrato ML
# ---------------------------------------------------------------------------


def _get_or_create_model_id(conn: Connection, sensor_id: int) -> int:
    row = conn.execute(
        text(
            """
            SELECT TOP 1 id
            FROM dbo.ml_models
            WHERE sensor_id = :sensor_id AND is_active = 1 AND model_type = 'sklearn'
            ORDER BY trained_at DESC
            """
        ),
        {"sensor_id": sensor_id},
    ).fetchone()

    if row:
        return int(row[0])

    created = conn.execute(
        text(
            """
            INSERT INTO dbo.ml_models (
              sensor_id, model_name, model_type, version, is_active, trained_at
            )
            OUTPUT INSERTED.id
            VALUES (
              :sensor_id, :model_name, :model_type, :version, 1, GETDATE()
            )
            """
        ),
        {
            "sensor_id": sensor_id,
            "model_name": "sklearn_regression_iforest",
            "model_type": "sklearn",
            "version": "1.0.0",
        },
    ).fetchone()

    if not created:
        raise RuntimeError("failed to create ml_models row")

    return int(created[0])


def _has_recent_delta_spike(conn: Connection, sensor_id: int, window_seconds: int = 30) -> bool:
    """Devuelve True si hay un DELTA_SPIKE activo/ack reciente para el sensor.

    Se usa para forzar coherencia mínima en predictions cuando el detector
    inmediato ya disparó un evento.
    """

    row = conn.execute(
        text(
            """
            SELECT TOP 1 1
            FROM dbo.ml_events
            WHERE sensor_id = :sensor_id
              AND event_code = 'DELTA_SPIKE'
              AND status IN ('active', 'acknowledged')
              AND created_at >= DATEADD(second, -:sec, GETDATE())
            ORDER BY created_at DESC
            """
        ),
        {"sensor_id": sensor_id, "sec": window_seconds},
    ).fetchone()
    return row is not None


def _insert_prediction_row(
    conn: Connection,
    *,
    model_id: int,
    sensor_id: int,
    device_id: int,
    explanation: PredictionExplanation,
    target_ts_utc: datetime,
    horizon_minutes: int,
    window_points: int,
) -> int:
    row = conn.execute(
        text(
            """
            INSERT INTO dbo.predictions (
              model_id,
              sensor_id,
              device_id,
              predicted_value,
              confidence,
              predicted_at,
              target_timestamp,
              horizon_minutes,
              window_points,
              trend,
              is_anomaly,
              anomaly_score,
              risk_level,
              severity,
              explanation,
              status
            )
            OUTPUT INSERTED.id
            VALUES (
              :model_id,
              :sensor_id,
              :device_id,
              :predicted_value,
              :confidence,
              GETDATE(),
              :target_timestamp,
              :horizon_minutes,
              :window_points,
              :trend,
              :is_anomaly,
              :anomaly_score,
              :risk_level,
              :severity,
              :explanation,
              'active'
            )
            """
        ),
        {
            "model_id": model_id,
            "sensor_id": sensor_id,
            "device_id": device_id,
            "predicted_value": explanation.predicted_value,
            "confidence": explanation.confidence,
            "target_timestamp": target_ts_utc.replace(tzinfo=None),
            "horizon_minutes": horizon_minutes,
            "window_points": window_points,
            "trend": explanation.trend,
            "is_anomaly": 1 if explanation.anomaly else 0,
            "anomaly_score": explanation.anomaly_score,
            "risk_level": explanation.risk_level,
            "severity": explanation.severity,
            "explanation": explanation.explanation,
        },
    ).fetchone()

    if not row:
        raise RuntimeError("failed to insert prediction")

    prediction_id = int(row[0])

    # 1) Violación de umbral físico (PRED_THRESHOLD_BREACH):
    #    se registra como evento en ml_events y la vista de umbrales físicos
    #    reflejará la severidad crítica. Aquí solo dejamos trazabilidad.
    _insert_threshold_event_if_needed(
        conn,
        sensor_id=sensor_id,
        device_id=device_id,
        prediction_id=prediction_id,
        predicted_value=float(explanation.predicted_value),
        dedupe_minutes=5,
    )

    # 2) Eventos DELTA_SPIKE / slope fuerte
    if _has_recent_delta_spike(conn, sensor_id=sensor_id, window_seconds=30):
        conn.execute(
            text(
                """
                UPDATE dbo.predictions
                SET
                  anomaly_score = CASE
                    WHEN anomaly_score IS NULL OR anomaly_score < 0.3 THEN 0.3
                    ELSE anomaly_score
                  END,
                  severity = CASE
                    WHEN UPPER(severity) = 'CRITICAL' THEN severity
                    WHEN severity IS NULL OR UPPER(severity) NOT IN ('WARNING','CRITICAL') THEN 'WARNING'
                    ELSE severity
                  END
                WHERE id = :id
                """
            ),
            {"id": prediction_id},
        )

    return prediction_id


def _should_dedupe_threshold_event(
    conn: Connection, *, sensor_id: int, event_code: str, dedupe_minutes: int
) -> bool:
    row = conn.execute(
        text(
            """
            SELECT TOP 1 1
            FROM dbo.ml_events
            WHERE sensor_id = :sensor_id
              AND event_code = :event_code
              AND status IN ('active', 'acknowledged')
              AND created_at >= DATEADD(minute, -:mins, GETDATE())
            ORDER BY created_at DESC
            """
        ),
        {"sensor_id": sensor_id, "event_code": event_code, "mins": dedupe_minutes},
    ).fetchone()
    return row is not None


def _can_sensor_emit_events(conn: Connection, sensor_id: int) -> bool:
    """Verifica si el sensor puede emitir eventos ML.
    
    REGLA DE DOMINIO:
    - Sensores en INITIALIZING o STALE NO pueden generar eventos
    - Solo sensores en NORMAL, WARNING o ALERT pueden generar eventos
    """
    try:
        state_manager = SensorStateManager(conn)
        can_generate, reason = state_manager.can_generate_events(sensor_id)
        if not can_generate:
            logger.debug(
                "[ML_BATCH_BLOCKED] sensor_id=%s cannot emit events: %s",
                sensor_id, reason
            )
        return can_generate
    except Exception as e:
        logger.warning(
            "[ML_BATCH_STATE_CHECK_ERROR] sensor_id=%s error=%s, allowing events",
            sensor_id, str(e)
        )
        return True


def _is_value_within_user_thresholds(
    conn: Connection,
    sensor_id: int,
    value: float,
) -> bool:
    """Verifica si el valor está dentro de los umbrales WARNING del usuario.
    
    REGLA DE DOMINIO CRÍTICA:
    Si el valor está dentro de [warning_min, warning_max], el ML NO puede
    generar eventos de anomalía. El usuario definió ese rango como "normal".
    
    Returns:
        True si el valor está dentro del rango (ML no debe alertar)
        False si el valor está fuera del rango o no hay umbrales configurados
    """
    row = conn.execute(
        text(
            """
            SELECT 
                threshold_value_min,
                threshold_value_max
            FROM dbo.alert_thresholds
            WHERE sensor_id = :sensor_id
              AND is_active = 1
              AND severity = 'warning'
              AND condition_type = 'out_of_range'
            ORDER BY id ASC
            """
        ),
        {"sensor_id": sensor_id},
    ).fetchone()
    
    if not row:
        # Sin umbrales configurados, ML puede operar libremente
        return False
    
    warning_min = float(row[0]) if row[0] is not None else None
    warning_max = float(row[1]) if row[1] is not None else None
    
    # Si no hay límites de warning, no podemos determinar si está dentro
    if warning_min is None and warning_max is None:
        return False
    
    # Verificar si está dentro del rango
    if warning_min is not None and value < warning_min:
        return False  # Fuera del rango (por debajo)
    if warning_max is not None and value > warning_max:
        return False  # Fuera del rango (por arriba)
    
    # Valor dentro del rango WARNING del usuario
    return True


def _insert_threshold_event_if_needed(
    conn: Connection,
    *,
    sensor_id: int,
    device_id: int,
    prediction_id: int,
    predicted_value: float,
    dedupe_minutes: int,
) -> None:
    # =========================================================================
    # FIX CRÍTICO: Verificar estado operacional del sensor ANTES de generar evento
    # Sensores en INITIALIZING o STALE NO pueden generar eventos
    # =========================================================================
    if not _can_sensor_emit_events(conn, sensor_id):
        return
    
    thr = conn.execute(
        text(
            """
            SELECT TOP 1
              id, condition_type, threshold_value_min, threshold_value_max, severity, name
            FROM dbo.alert_thresholds
            WHERE sensor_id = :sensor_id AND is_active = 1
            ORDER BY id ASC
            """
        ),
        {"sensor_id": sensor_id},
    ).fetchone()

    if not thr:
        return

    threshold_id, cond, vmin, vmax, severity, thr_name = thr

    vmin_f = float(vmin) if vmin is not None else None
    vmax_f = float(vmax) if vmax is not None else None
    violated = False

    if cond == "greater_than" and vmin_f is not None and predicted_value > vmin_f:
        violated = True
    elif cond == "less_than" and vmin_f is not None and predicted_value < vmin_f:
        violated = True
    elif cond == "out_of_range" and vmin_f is not None and vmax_f is not None:
        violated = predicted_value < vmin_f or predicted_value > vmax_f
    elif cond == "equal_to" and vmin_f is not None and predicted_value == vmin_f:
        violated = True

    if not violated:
        return

    event_code = "PRED_THRESHOLD_BREACH"
    if _should_dedupe_threshold_event(
        conn, sensor_id=sensor_id, event_code=event_code, dedupe_minutes=dedupe_minutes
    ):
        return

    sev = str(severity)
    if sev == "critical":
        event_type = "critical"
    elif sev == "warning":
        event_type = "warning"
    else:
        event_type = "notice"

    title = f"Predicción viola umbral: {thr_name}"
    message = f"predicted_value={predicted_value} threshold_id={int(threshold_id)}"

    payload = (
        "{"  # JSON simple para trazabilidad
        f"\"threshold_id\": {int(threshold_id)}, "
        f"\"condition_type\": \"{cond}\", "
        f"\"threshold_value_min\": { 'null' if vmin is None else float(vmin) }, "
        f"\"threshold_value_max\": { 'null' if vmax is None else float(vmax) }, "
        f"\"predicted_value\": {predicted_value}"
        "}"
    )

    conn.execute(
        text(
            """
            INSERT INTO dbo.ml_events (
              device_id, sensor_id, prediction_id,
              event_type, event_code, title, message,
              status, created_at, payload
            )
            VALUES (
              :device_id, :sensor_id, :prediction_id,
              :event_type, :event_code, :title, :message,
              'active', GETDATE(), :payload
            )
            """
        ),
        {
            "device_id": device_id,
            "sensor_id": sensor_id,
            "prediction_id": prediction_id,
            "event_type": event_type,
            "event_code": event_code,
            "title": title,
            "message": message,
            "payload": payload,
        },
    )


def _insert_anomaly_event(
    conn: Connection,
    *,
    sensor_id: int,
    device_id: int,
    prediction_id: int,
    explanation: PredictionExplanation,
) -> None:
    if not explanation.anomaly:
        return
    
    # =========================================================================
    # FIX CRÍTICO: Verificar si el valor predicho está dentro del rango del usuario
    # Si está dentro del rango WARNING, NO generar evento de anomalía
    # =========================================================================
    if _is_value_within_user_thresholds(conn, sensor_id, explanation.predicted_value):
        logger.debug(
            "[ML_ANOMALY_SUPPRESSED] sensor_id=%s predicted_value=%.4f within user thresholds, "
            "suppressing ANOMALY_DETECTED event",
            sensor_id, explanation.predicted_value
        )
        return
    
    # =========================================================================
    # FIX CRÍTICO: Verificar estado operacional del sensor
    # =========================================================================
    if not _can_sensor_emit_events(conn, sensor_id):
        return

    event_code = "ANOMALY_DETECTED"
    event_type = "warning"  # o "critical" según tu política

    title = "Posible anomalía detectada por ML"
    message = (
        f"severidad={explanation.severity} "
        f"action_required={explanation.action_required} "
        f"trend={explanation.trend}"
    )

    safe_expl = explanation.explanation.replace("\"", "'")
    safe_action = explanation.recommended_action.replace("\"", "'")
    payload = (
        "{"  # JSON simple
        f"\"severity\": \"{explanation.severity}\", "
        f"\"action_required\": {str(explanation.action_required).lower()}, "
        f"\"anomaly_score\": {explanation.anomaly_score:.4f}, "
        f"\"trend\": \"{explanation.trend}\", "
        f"\"predicted_value\": {explanation.predicted_value:.5f}, "
        f"\"confidence\": {explanation.confidence:.4f}, "
        f"\"recommended_action\": \"{safe_action}\", "
        f"\"explanation\": \"{safe_expl}\""
        "}"
    )

    conn.execute(
        text(
            """
            INSERT INTO dbo.ml_events (
              device_id, sensor_id, prediction_id,
              event_type, event_code, title, message,
              status, created_at, payload
            )
            VALUES (
              :device_id, :sensor_id, :prediction_id,
              :event_type, :event_code, :title, :message,
              'active', GETDATE(), :payload
            )
            """
        ),
        {
            "device_id": device_id,
            "sensor_id": sensor_id,
            "prediction_id": prediction_id,
            "event_type": event_type,
            "event_code": event_code,
            "title": title,
            "message": message,
            "payload": payload,
        },
    )


# ---------------------------------------------------------------------------
# Lógica de ML por sensor
# ---------------------------------------------------------------------------


def _get_user_defined_range(conn: Connection, sensor_id: int) -> tuple[float, float] | None:
    """Obtiene el rango definido por el usuario desde alert_thresholds.
    
    REGLA DE DOMINIO CRÍTICA:
    Los umbrales del usuario tienen PRIORIDAD sobre cualquier heurística del ML.
    Si el usuario definió un rango, ese es el rango válido.
    
    Returns:
        (min, max) si hay umbrales configurados, None si no hay
    """
    row = conn.execute(
        text(
            """
            SELECT 
                threshold_value_min,
                threshold_value_max
            FROM dbo.alert_thresholds
            WHERE sensor_id = :sensor_id
              AND is_active = 1
              AND condition_type = 'out_of_range'
            ORDER BY 
                CASE severity WHEN 'warning' THEN 0 ELSE 1 END,
                id ASC
            """
        ),
        {"sensor_id": sensor_id},
    ).fetchone()
    
    if not row:
        return None
    
    min_val = float(row[0]) if row[0] is not None else None
    max_val = float(row[1]) if row[1] is not None else None
    
    if min_val is None and max_val is None:
        return None
    
    # Si solo hay un límite, usar un rango muy amplio para el otro
    if min_val is None:
        min_val = float('-inf')
    if max_val is None:
        max_val = float('inf')
    
    return (min_val, max_val)


def _derive_recommended_range(sensor_meta: SensorMetadata) -> tuple[float, float] | None:
    """Heurística de rango recomendado por tipo de sensor.

    NOTA: Esta función es un FALLBACK cuando no hay umbrales del usuario.
    Los umbrales del usuario siempre tienen prioridad.
    """

    t = sensor_meta.sensor_type
    if t == "temperature":
        return 15.0, 35.0
    if t == "humidity":
        return 30.0, 70.0
    if t == "air_quality":  # CO2 ppm
        return 400.0, 1000.0
    if t in {"power", "voltage"}:
        # Rango ficticio, depende del dominio real.
        return 0.0, 100000.0
    return None


def _compute_risk_level(sensor_meta: SensorMetadata, predicted_value: float) -> str:
    """Clasifica el nivel de riesgo físico (umbral) sin considerar anomalía estadística.

    Devuelve: 'LOW' | 'MEDIUM' | 'HIGH' | 'NONE'.
    """

    rng = _derive_recommended_range(sensor_meta)
    if rng is None:
        return "NONE"

    min_ok, max_ok = rng
    if min_ok <= predicted_value <= max_ok:
        return "LOW"

    margin = 0.1 * (max_ok - min_ok)
    if predicted_value < min_ok - margin or predicted_value > max_ok + margin:
        return "HIGH"

    return "MEDIUM"


def compute_severity(*, is_anomaly: bool, risk_level: str, out_of_physical_range: bool) -> str:
    """Combina anomalía estadística + riesgo físico en una severidad única.

    Prioridad de reglas:
    1) Si hay violación de umbral físico (fuera de rango) => CRITICAL.
    2) Si hay anomalía y riesgo físico alto => CRITICAL.
    3) Si hay anomalía o riesgo alto => WARNING.
    4) Resto de casos => INFO.
    """

    rl = (risk_level or "").upper()

    if out_of_physical_range:
        return "critical"
    if is_anomaly and rl == "HIGH":
        return "critical"
    if is_anomaly or rl == "HIGH":
        return "warning"
    return "info"


def _classify_severity(
    *,
    sensor_meta: SensorMetadata,
    predicted_value: float,
    trend: Trend,
    anomaly: bool,
    anomaly_score: float,
    confidence: float,
    horizon_minutes: int,
    user_defined_range: tuple[float, float] | None = None,
) -> tuple[str, str, bool, str]:
    """Devuelve (risk_level, severity, action_required, recommended_action).

    - risk_level: riesgo físico por umbral.
    - severity: combinación de anomalía + riesgo (compute_severity).
    
    FIX CRÍTICO: Si se proporciona user_defined_range, se usa en lugar de
    las heurísticas del ML. Los umbrales del usuario tienen PRIORIDAD.
    """

    risk_level = _compute_risk_level(sensor_meta, predicted_value)

    # FIX CRÍTICO: Usar umbrales del usuario si están disponibles
    # Los umbrales del usuario tienen PRIORIDAD sobre las heurísticas del ML
    rng = user_defined_range if user_defined_range is not None else _derive_recommended_range(sensor_meta)
    out_of_range = False
    if rng is not None:
        min_ok, max_ok = rng
        out_of_range = predicted_value < min_ok or predicted_value > max_ok

    severity = compute_severity(
        is_anomaly=anomaly,
        risk_level=risk_level,
        out_of_physical_range=out_of_range,
    )

    action_required = False
    recommended_action = "Sin acción requerida por ahora. Seguir monitoreando el sensor."

    # Ajustar acción según combinación de riesgo + anomalía, evitando contradicciones.
    rl = risk_level.upper()

    if severity == "info":
        # Solo mensajes neutrales.
        if rl in {"MEDIUM", "HIGH"}:
            recommended_action = (
                f"La proyección se acerca a los límites operativos en {sensor_meta.location}. "
                "Supervisar el comportamiento en las próximas horas."
            )
        else:
            recommended_action = (
                "Valores dentro o cerca del rango esperado. No se requiere acción inmediata."
            )
        action_required = False
        return risk_level, severity, action_required, recommended_action

    # A partir de aquí, severidad es warning o critical.
    action_required = True

    if severity == "critical":
        recommended_action = (
            f"Condición crítica detectada en {sensor_meta.location}. "
            "Revisar inmediatamente el equipo, la instalación y las condiciones ambientales."
        )
    else:  # warning
        if rl == "HIGH":
            recommended_action = (
                f"Riesgo elevado detectado en {sensor_meta.location}. "
                "Programar una revisión prioritaria en las próximas horas."
            )
        else:
            recommended_action = (
                f"Comportamiento inusual detectado en {sensor_meta.location}. "
                "Supervisar de cerca y considerar una inspección programada."
            )

    return risk_level, severity, action_required, recommended_action


def _build_explanation(
    *,
    sensor_meta: SensorMetadata,
    predicted_value: float,
    trend: Trend,
    anomaly: bool,
    anomaly_score: float,
    confidence: float,
    horizon_minutes: int,
    user_defined_range: tuple[float, float] | None = None,
) -> PredictionExplanation:
    # FIX CRÍTICO: Pasar umbrales del usuario a _classify_severity
    risk_level, severity, action_required, recommended_action = _classify_severity(
        sensor_meta=sensor_meta,
        predicted_value=predicted_value,
        trend=trend,
        anomaly=anomaly,
        anomaly_score=anomaly_score,
        confidence=confidence,
        horizon_minutes=horizon_minutes,
        user_defined_range=user_defined_range,
    )

    # Regla adicional: evitar contradicciones obvias
    if severity == "critical" and anomaly_score <= 0:
        anomaly_score = 0.5

    # short_message + recommended_action + details (JSON estructurado)
    if severity == "critical":
        short_message = (
            f"Riesgo crítico previsto en {sensor_meta.sensor_type} en {sensor_meta.location}."
        )
    elif severity == "warning":
        short_message = (
            f"Comportamiento inusual previsto en {sensor_meta.sensor_type} en {sensor_meta.location}."
        )
    else:
        short_message = (
            f"Predicción estable para {sensor_meta.sensor_type} en {sensor_meta.location}."
        )

    details = {
        "predicted_value": float(predicted_value),
        "trend": trend,
        "anomaly_score": float(anomaly_score),
        "confidence": float(confidence),
        "horizon_minutes": int(horizon_minutes),
        "risk_level": risk_level,
        "sensor_type": sensor_meta.sensor_type,
        "location": sensor_meta.location,
    }

    explanation_payload = {
        "source": "ml_baseline",
        "severity": severity.upper(),
        "short_message": short_message,
        "recommended_action": recommended_action,
        "details": details,
    }
    explanation_json = json.dumps(explanation_payload, ensure_ascii=False)

    return PredictionExplanation(
        sensor_id=sensor_meta.sensor_id,
        predicted_value=predicted_value,
        trend=trend,
        anomaly=anomaly,
        anomaly_score=anomaly_score,
        confidence=confidence,
        explanation=explanation_json,
        risk_level=risk_level,
        severity=severity,
        action_required=action_required,
        recommended_action=recommended_action,
    )


def _process_sensor(
    conn: Connection,
    sensor_id: int,
    ml_cfg: GlobalMLConfig,
    iso_trainer: IsolationForestTrainer,
    dedupe_minutes: int,
) -> None:
    series: SensorSeries = load_sensor_series(
        conn, sensor_id, limit_points=ml_cfg.regression.window_points
    )
    sensor_meta = load_sensor_metadata(conn, sensor_id)
    n_points = len(series.values)
    if not series.values:
        logger.info("[ML-RUNNER] sensor=%s sin datos; se omite (0 lecturas)", sensor_id)
        return

    logger.info(
        "[ML-RUNNER] sensor=%s: %s lecturas cargadas para entrenamiento (window=%s)",
        sensor_id,
        n_points,
        ml_cfg.regression.window_points,
    )

    reg_cfg: RegressionConfig = ml_cfg.regression

    # 1) Entrenar regresión con ventana deslizante
    reg_model, last_minutes = train_regression_for_sensor(conn, sensor_id, reg_cfg)

    if reg_model is None or last_minutes is None:
        # Sensores sin datos suficientes: fallback a promedio simple
        last_values = series.values[-min(5, len(series.values)) :]
        predicted_value = float(sum(last_values) / len(last_values))
        trend: Trend = "stable"
        confidence = reg_cfg.min_confidence
        anomaly = False
        anomaly_score = 0.0
        window_points_effective = len(series.values)
    else:
        # FIX 2: Predicción N minutos adelante con clamp para evitar valores extremos
        series_min = float(min(series.values))
        series_max = float(max(series.values))
        last_value = float(series.values[-1])
        
        predicted_value = predict_future_value_clamped(
            reg_model,
            last_minutes,
            last_value=last_value,
            series_min=series_min,
            series_max=series_max,
            max_change_ratio=0.5,  # Máximo 50% de cambio
        )
        trend = compute_trend(reg_model.coef_)

        # Residuales históricos para IsolationForest
        t0 = series.timestamps[0]
        xs: list[list[float]] = []
        for ts in series.timestamps:
            minutes = (ts - t0).total_seconds() / 60.0
            xs.append([minutes])
        X = np.asarray(xs, dtype=float)
        y = np.asarray(series.values, dtype=float)
        y_hat_hist = reg_model.intercept_ + reg_model.coef_ * X.ravel()
        residuals = y - y_hat_hist

        window_points_effective = len(series.values)

        # Confianza basada en R^2 + nº de puntos
        conf_r2 = max(0.0, min(1.0, reg_model.r2))
        raw_conf = min(1.0, window_points_effective / reg_cfg.window_points)
        confidence = max(
            reg_cfg.min_confidence,
            min(reg_cfg.max_confidence, 0.5 * (conf_r2 + raw_conf)),
        )

        # 2) Anomalía con IsolationForest (si hay suficientes datos)
        anomaly = False
        anomaly_score = 0.0
        model = iso_trainer.fit_for_sensor(sensor_id, residuals)
        if model is not None:
            last_residual = float(residuals[-1])
            anomaly_score, anomaly = iso_trainer.score_new_point(sensor_id, last_residual)

    # =========================================================================
    # FIX CRÍTICO: Subordinar anomalía ML a umbrales del usuario
    # Si el valor predicho está dentro del rango WARNING del usuario,
    # el ML NO puede marcar como anomalía. El usuario definió ese rango
    # como "normal" y el ML debe respetarlo.
    # =========================================================================
    if anomaly and _is_value_within_user_thresholds(conn, sensor_id, predicted_value):
        logger.debug(
            "[ML_ANOMALY_SUPPRESSED] sensor_id=%s predicted_value=%.4f within user thresholds, "
            "suppressing anomaly flag",
            sensor_id, predicted_value
        )
        anomaly = False
        # Mantener anomaly_score para trazabilidad, pero la anomalía no genera eventos

    # 3) Construir explicación contextual por tipo de sensor
    # FIX CRÍTICO: Obtener umbrales del usuario para subordinar ML a ellos
    user_range = _get_user_defined_range(conn, sensor_id)
    
    explanation = _build_explanation(
        sensor_meta=sensor_meta,
        predicted_value=predicted_value,
        trend=trend,
        anomaly=anomaly,
        anomaly_score=anomaly_score,
        confidence=confidence,
        horizon_minutes=reg_cfg.horizon_minutes,
        user_defined_range=user_range,
    )

    # 4) Persistir en BD
    device_id = get_device_id_for_sensor(conn, sensor_id)
    model_id = _get_or_create_model_id(conn, sensor_id)
    target_ts = _utc_now() + timedelta(minutes=reg_cfg.horizon_minutes)

    prediction_id = _insert_prediction_row(
        conn,
        model_id=model_id,
        sensor_id=sensor_id,
        device_id=device_id,
        explanation=explanation,
        target_ts_utc=target_ts,
        horizon_minutes=reg_cfg.horizon_minutes,
        window_points=window_points_effective,
    )

    logger.info(
        "[ML-RUNNER] prediction creada id=%s sensor=%s device=%s pred=%.4f conf=%.2f puntos=%s",
        prediction_id,
        sensor_id,
        device_id,
        explanation.predicted_value,
        explanation.confidence,
        window_points_effective,
    )

    # 5) Eventos: umbral y anomalía
    _insert_threshold_event_if_needed(
        conn,
        sensor_id=sensor_id,
        device_id=device_id,
        prediction_id=prediction_id,
        predicted_value=predicted_value,
        dedupe_minutes=dedupe_minutes,
    )

    _insert_anomaly_event(
        conn,
        sensor_id=sensor_id,
        device_id=device_id,
        prediction_id=prediction_id,
        explanation=explanation,
    )

    logger.info(
        "[ML-RUNNER] OK sensor=%s pred=%.3f trend=%s anomaly=%s score=%.4f conf=%.2f",
        sensor_id,
        predicted_value,
        trend,
        explanation.anomaly,
        explanation.anomaly_score,
        explanation.confidence,
    )


# ---------------------------------------------------------------------------
# Loop principal del batch
# ---------------------------------------------------------------------------


def run_once(ml_cfg: GlobalMLConfig, dedupe_minutes: int) -> None:
    engine = get_engine()
    iso_trainer = IsolationForestTrainer(ml_cfg.anomaly)

    with engine.begin() as conn:  # una transacción por iteración
        for sensor_id in _iter_sensors(conn):
            try:
                _process_sensor(conn, sensor_id, ml_cfg, iso_trainer, dedupe_minutes)
            except Exception:
                # Loggear error sin romper el batch
                logger.exception("Error procesando sensor_id=%s", sensor_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML batch runner (sklearn regression + IsolationForest)"
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=60.0,
        help="Intervalo entre ejecuciones (segundos). Ignorado si se usa --once.",
    )
    parser.add_argument(
        "--dedupe-minutes",
        type=int,
        default=10,
        help="Minutos para deduplicar eventos de cruce de umbral.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Ejecutar solo una vez y salir.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    logger.info("[ML-RUNNER] Iniciando ML batch runner (sklearn + IsolationForest)")

    ml_cfg = GlobalMLConfig()
    cfg = RunnerConfig(
        interval_seconds=args.interval_seconds,
        once=bool(args.once),
        dedupe_minutes=args.dedupe_minutes,
    )

    while True:
        logger.info("Inicio iteración ML batch runner")
        run_once(ml_cfg, dedupe_minutes=cfg.dedupe_minutes)
        logger.info("Fin iteración ML batch runner")

        if cfg.once:
            break

        time.sleep(cfg.interval_seconds)


if __name__ == "__main__":
    main()
