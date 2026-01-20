from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated
import logging

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.engine import Connection

# Imports de infraestructura compartida (BD)
from iot_ingest_services.common.db import get_engine

# Imports internos de ML (ahora en iot_machine_learning)
from iot_machine_learning.ml.baseline import BaselineConfig, predict_moving_average
from iot_machine_learning.ml.metadata import BASELINE_MOVING_AVERAGE


logger = logging.getLogger(__name__)

app = FastAPI(title="IoT ML Service", version="0.1.0")


# ---------------------------------------------------------------------------
# Utilidades compartidas (adaptadas de jobs/ml_batch_runner.py)
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_db_conn() -> Connection:
    """Dependencia FastAPI para obtener una conexión SQLAlchemy.

    Usa el mismo engine que iot_ingest_services.common.db.
    """

    engine = get_engine()
    with engine.begin() as conn:  # type: ignore[call-arg]
        yield conn


def _get_or_create_active_model_id(conn: Connection, sensor_id: int) -> int:
    row = conn.execute(
        text(
            """
            SELECT TOP 1 id
            FROM dbo.ml_models
            WHERE sensor_id = :sensor_id AND is_active = 1
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
            INSERT INTO dbo.ml_models (sensor_id, model_name, model_type, version, is_active, trained_at)
            OUTPUT INSERTED.id
            VALUES (:sensor_id, :model_name, :model_type, :version, 1, GETDATE())
            """
        ),
        {
            "sensor_id": sensor_id,
            "model_name": BASELINE_MOVING_AVERAGE.name,
            "model_type": BASELINE_MOVING_AVERAGE.model_type,
            "version": BASELINE_MOVING_AVERAGE.version,
        },
    ).fetchone()

    if not created:
        raise RuntimeError("failed to create ml_models row")

    return int(created[0])


def _insert_prediction(
    conn: Connection,
    *,
    model_id: int,
    sensor_id: int,
    predicted_value: float,
    confidence: float,
    target_ts_utc: datetime,
) -> int:
    row = conn.execute(
        text(
            """
            INSERT INTO dbo.predictions (
              model_id, sensor_id, predicted_value, confidence, predicted_at, target_timestamp
            )
            OUTPUT INSERTED.id
            VALUES (
              :model_id, :sensor_id, :predicted_value, :confidence, GETDATE(), :target_timestamp
            )
            """
        ),
        {
            "model_id": model_id,
            "sensor_id": sensor_id,
            "predicted_value": predicted_value,
            "confidence": confidence,
            "target_timestamp": target_ts_utc.replace(tzinfo=None),
        },
    ).fetchone()

    if not row:
        raise RuntimeError("failed to insert prediction")

    return int(row[0])


def _get_device_id_for_sensor(conn: Connection, sensor_id: int) -> int:
    row = conn.execute(
        text("SELECT device_id FROM dbo.sensors WHERE id = :sensor_id"),
        {"sensor_id": sensor_id},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="sensor_id not found")
    return int(row[0])


def _load_recent_values(conn: Connection, sensor_id: int, window: int) -> list[float]:
    # FIX FASE2: Evitar CAST AS float, mantener precisión DECIMAL(15,5)
    # FIX FASE4: Filtrar valores inválidos (NaN, Infinity, None)
    from iot_machine_learning.ml_service.utils.numeric_precision import safe_float, is_valid_sensor_value
    
    rows = conn.execute(
        text(
            """
            SELECT TOP (:limit) [value] AS v
            FROM dbo.sensor_readings
            WHERE sensor_id = :sensor_id
            ORDER BY [timestamp] DESC
            """
        ),
        {"sensor_id": sensor_id, "limit": window},
    ).fetchall()

    # Filtrar solo valores válidos (no None, no NaN, no Infinity)
    return [safe_float(r[0]) for r in rows if is_valid_sensor_value(r[0])]


def _should_dedupe_event(conn: Connection, *, sensor_id: int, event_code: str, dedupe_minutes: int) -> bool:
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


def _eval_pred_threshold_and_create_event(
    conn: Connection,
    *,
    sensor_id: int,
    device_id: int,
    prediction_id: int,
    predicted_value: float,
    dedupe_minutes: int,
) -> None:
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

    violated = False
    vmin_f = float(vmin) if vmin is not None else None
    vmax_f = float(vmax) if vmax is not None else None

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
    if _should_dedupe_event(conn, sensor_id=sensor_id, event_code=event_code, dedupe_minutes=dedupe_minutes):
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
        '{'
        f'"threshold_id": {int(threshold_id)}, '
        f'"condition_type": "{cond}", '
        f'"threshold_value_min": {"null" if vmin is None else float(vmin)}, '
        f'"threshold_value_max": {"null" if vmax is None else float(vmax)}, '
        f'"predicted_value": {predicted_value}'
        '}'
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
# Esquemas Pydantic
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    sensor_id: int = Field(..., gt=0)
    horizon_minutes: int = Field(10, gt=0, le=1440)
    window: int = Field(60, gt=0, le=1000)
    dedupe_minutes: int = Field(10, gt=0, le=1440)


class PredictResponse(BaseModel):
    sensor_id: int
    model_id: int
    prediction_id: int
    predicted_value: float
    confidence: float
    target_timestamp: datetime
    horizon_minutes: int
    window: int


DbConnDep = Annotated[Connection, Depends(get_db_conn)]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    logger.info("[ML-SERVICE] Health check solicitado")
    return {"status": "ok"}


@app.post("/ml/predict", response_model=PredictResponse)
async def ml_predict(payload: PredictRequest, conn: DbConnDep) -> PredictResponse:
    logger.info(
        "[ML-SERVICE] /ml/predict sensor_id=%s horizon=%s window=%s dedupe=%s",
        payload.sensor_id,
        payload.horizon_minutes,
        payload.window,
        payload.dedupe_minutes,
    )

    values = _load_recent_values(conn, sensor_id=payload.sensor_id, window=payload.window)
    if not values:
        logger.warning("[ML-SERVICE] sensor_id=%s sin lecturas recientes", payload.sensor_id)
        raise HTTPException(status_code=400, detail="No hay lecturas recientes para ese sensor")

    baseline_cfg = BaselineConfig(window=payload.window)
    predicted_value, confidence = predict_moving_average(values, baseline_cfg)

    model_id = _get_or_create_active_model_id(conn, payload.sensor_id)
    target_ts = _utc_now() + timedelta(minutes=payload.horizon_minutes)

    prediction_id = _insert_prediction(
        conn,
        model_id=model_id,
        sensor_id=payload.sensor_id,
        predicted_value=predicted_value,
        confidence=confidence,
        target_ts_utc=target_ts,
    )

    device_id = _get_device_id_for_sensor(conn, payload.sensor_id)
    _eval_pred_threshold_and_create_event(
        conn,
        sensor_id=payload.sensor_id,
        device_id=device_id,
        prediction_id=prediction_id,
        predicted_value=predicted_value,
        dedupe_minutes=payload.dedupe_minutes,
    )

    return PredictResponse(
        sensor_id=payload.sensor_id,
        model_id=model_id,
        prediction_id=prediction_id,
        predicted_value=predicted_value,
        confidence=confidence,
        target_timestamp=target_ts,
        horizon_minutes=payload.horizon_minutes,
        window=payload.window,
    )
