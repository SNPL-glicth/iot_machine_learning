"""Prediction deviation checker for the ML stream runner.

Extracted from SimpleMlOnlineProcessor._check_prediction_deviation to keep
the runner class focused on the main reading-processing loop.

Checks whether the actual sensor reading deviates significantly from the
most recent ML prediction, and emits a PREDICTION_DEVIATION event if so.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import text

from iot_ingest_services.common.db import get_engine
from iot_machine_learning.ml_service.config.ml_config import OnlineBehaviorConfig

logger = logging.getLogger(__name__)


def check_prediction_deviation(
    *,
    sensor_id: int,
    reading_value: float,
    reading_timestamp: float,
    sensor_type: str,
    cfg: OnlineBehaviorConfig,
    threshold_validator,
    event_persister,
    explanation_builder,
    analysis,
) -> None:
    """Verifica desviación entre predicción y valor real.

    Emits a PREDICTION_DEVIATION event when the absolute or relative error
    between the stored prediction and the actual reading exceeds the
    configured thresholds.

    Args:
        sensor_id: Sensor identifier.
        reading_value: Actual observed value.
        reading_timestamp: Unix timestamp of the reading.
        sensor_type: Sensor type string.
        cfg: Online behavior configuration.
        threshold_validator: ThresholdValidator instance.
        event_persister: MLEventPersister instance.
        explanation_builder: ExplanationBuilder instance.
        analysis: OnlineAnalysis result from the current cycle.
    """
    if threshold_validator.is_value_within_warning_range(sensor_id, reading_value):
        return

    engine = get_engine()
    reading_dt = datetime.fromtimestamp(reading_timestamp, tz=timezone.utc)

    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT TOP 1 id, [predicted_value], target_timestamp
                FROM dbo.predictions
                WHERE sensor_id = :sensor_id
                  AND ABS(DATEDIFF(second, target_timestamp, :ts)) <= :tol
                ORDER BY ABS(DATEDIFF(second, target_timestamp, :ts)) ASC
                """
            ),
            {
                "sensor_id": sensor_id,
                "ts": reading_dt.replace(tzinfo=None),
                "tol": cfg.prediction_time_tolerance_seconds,
            },
        ).fetchone()

        if not row:
            return

        prediction_id, predicted_value, _target_ts = row
        predicted_value_f = float(predicted_value) if predicted_value is not None else 0.0

        error_abs = abs(reading_value - predicted_value_f)
        denom = max(abs(predicted_value_f), 1e-6)
        error_rel = error_abs / denom

        if (
            error_abs < cfg.prediction_error_absolute
            and error_rel < cfg.prediction_error_relative
        ):
            return

        if event_persister.should_dedupe_prediction_deviation(
            conn,
            sensor_id=sensor_id,
            dedupe_minutes=cfg.dedupe_minutes_prediction_deviation,
        ):
            return

    severity_label = "WARN" if error_rel < 0.5 else "CRITICAL"
    event_type = explanation_builder.map_severity_to_event_type(severity_label)

    explanation = (
        "Desviación significativa entre la predicción del modelo y el valor real. "
        f"real={reading_value:.4f} predicho={predicted_value_f:.4f} "
        f"error_abs={error_abs:.4f} error_rel={error_rel:.2%}."
    )
    recommended_action = (
        "Revisar la configuración del modelo y las condiciones del sensor. "
        "Si la desviación persiste, considerar recalibrar o reentrenar el modelo."
    )

    event_persister.insert_ml_event(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        severity_label=severity_label,
        event_type=event_type,
        event_code="PREDICTION_DEVIATION",
        title="ML online: PREDICTION_DEVIATION",
        explanation=explanation,
        recommended_action=recommended_action,
        analysis=analysis,
        ts_utc=reading_timestamp,
        prediction_id=int(prediction_id),
        extra_payload={
            "prediction_id": int(prediction_id),
            "predicted_value": predicted_value_f,
            "error_abs": error_abs,
            "error_rel": error_rel,
        },
    )
