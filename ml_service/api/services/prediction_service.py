"""Prediction service for ML API.

Handles prediction logic, model management, and event creation.
Extracted from main.py for modularity.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.engine import Connection

from iot_machine_learning.ml.baseline import BaselineConfig, predict_moving_average
from iot_machine_learning.ml.metadata import BASELINE_MOVING_AVERAGE
from iot_machine_learning.ml_service.utils.numeric_precision import safe_float, is_valid_sensor_value

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for generating predictions.
    
    Responsibilities:
    - Load recent sensor values
    - Generate predictions using ML models
    - Insert predictions into database
    - Evaluate thresholds and create events
    """
    
    def __init__(self, conn: Connection):
        self._conn = conn
    
    def predict(
        self,
        *,
        sensor_id: int,
        horizon_minutes: int = 10,
        window: int = 60,
        dedupe_minutes: int = 10,
    ) -> dict:
        """Generate a prediction for a sensor.
        
        Args:
            sensor_id: ID of the sensor
            horizon_minutes: Prediction horizon in minutes
            window: Number of recent values to use
            dedupe_minutes: Minutes for event deduplication
            
        Returns:
            dict with prediction details
            
        Raises:
            ValueError: If no recent readings available
        """
        # Load recent values
        values = self._load_recent_values(sensor_id, window)
        if not values:
            raise ValueError(f"No recent readings for sensor {sensor_id}")
        
        # Generate prediction
        baseline_cfg = BaselineConfig(window=window)
        predicted_value, confidence = predict_moving_average(values, baseline_cfg)
        
        # Get or create model
        model_id = self._get_or_create_active_model_id(sensor_id)
        
        # Get device ID
        device_id = self._get_device_id_for_sensor(sensor_id)
        
        # Calculate target timestamp
        target_ts = self._utc_now() + timedelta(minutes=horizon_minutes)
        
        # Insert prediction
        prediction_id = self._insert_prediction(
            model_id=model_id,
            sensor_id=sensor_id,
            device_id=device_id,
            predicted_value=predicted_value,
            confidence=confidence,
            target_ts_utc=target_ts,
        )
        
        # Evaluate thresholds and create events if needed
        self._eval_pred_threshold_and_create_event(
            sensor_id=sensor_id,
            device_id=device_id,
            prediction_id=prediction_id,
            predicted_value=predicted_value,
            dedupe_minutes=dedupe_minutes,
        )
        
        return {
            "sensor_id": sensor_id,
            "model_id": model_id,
            "prediction_id": prediction_id,
            "predicted_value": predicted_value,
            "confidence": confidence,
            "target_timestamp": target_ts,
            "horizon_minutes": horizon_minutes,
            "window": window,
        }
    
    def _utc_now(self) -> datetime:
        return datetime.now(timezone.utc)
    
    def _load_recent_values(self, sensor_id: int, window: int) -> list[float]:
        """Load recent sensor values."""
        rows = self._conn.execute(
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
        
        return [safe_float(r[0]) for r in rows if is_valid_sensor_value(r[0])]
    
    def _get_or_create_active_model_id(self, sensor_id: int) -> int:
        """Get or create an active ML model for the sensor."""
        row = self._conn.execute(
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
        
        created = self._conn.execute(
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
            raise RuntimeError("Failed to create ml_models row")
        
        return int(created[0])
    
    def _get_device_id_for_sensor(self, sensor_id: int) -> int:
        """Get device ID for a sensor."""
        row = self._conn.execute(
            text("SELECT device_id FROM dbo.sensors WHERE id = :sensor_id"),
            {"sensor_id": sensor_id},
        ).fetchone()
        
        if not row:
            raise ValueError(f"Sensor {sensor_id} not found")
        
        return int(row[0])
    
    def _insert_prediction(
        self,
        *,
        model_id: int,
        sensor_id: int,
        device_id: int,
        predicted_value: float,
        confidence: float,
        target_ts_utc: datetime,
    ) -> int:
        """Insert a prediction into the database."""
        row = self._conn.execute(
            text(
                """
                INSERT INTO dbo.predictions (
                  model_id, sensor_id, device_id, predicted_value, confidence, predicted_at, target_timestamp
                )
                OUTPUT INSERTED.id
                VALUES (
                  :model_id, :sensor_id, :device_id, :predicted_value, :confidence, GETDATE(), :target_timestamp
                )
                """
            ),
            {
                "model_id": model_id,
                "sensor_id": sensor_id,
                "device_id": device_id,
                "predicted_value": predicted_value,
                "confidence": confidence,
                "target_timestamp": target_ts_utc.replace(tzinfo=None),
            },
        ).fetchone()
        
        if not row:
            raise RuntimeError("Failed to insert prediction")
        
        return int(row[0])
    
    def _eval_pred_threshold_and_create_event(
        self,
        *,
        sensor_id: int,
        device_id: int,
        prediction_id: int,
        predicted_value: float,
        dedupe_minutes: int,
    ) -> None:
        """Evaluate prediction against thresholds and create event if violated."""
        # Check if value is within warning range
        if self._is_value_within_warning_range(sensor_id, predicted_value):
            return
        
        # Get threshold
        thr = self._conn.execute(
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
        
        # Check if threshold is violated
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
        
        # Check deduplication
        event_code = "PRED_THRESHOLD_BREACH"
        if self._should_dedupe_event(sensor_id, event_code, dedupe_minutes):
            return
        
        # Determine event type
        sev = str(severity)
        if sev == "critical":
            event_type = "critical"
        elif sev == "warning":
            event_type = "warning"
        else:
            event_type = "notice"
        
        # Create event
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
        
        self._conn.execute(
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
    
    def _is_value_within_warning_range(self, sensor_id: int, value: float) -> bool:
        """Check if value is within user-defined warning range."""
        row = self._conn.execute(
            text(
                """
                SELECT threshold_value_min, threshold_value_max
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
            return False
        
        warning_min = float(row[0]) if row[0] is not None else None
        warning_max = float(row[1]) if row[1] is not None else None
        
        if warning_min is None and warning_max is None:
            return False
        
        if warning_min is not None and value < warning_min:
            return False
        if warning_max is not None and value > warning_max:
            return False
        
        return True
    
    def _should_dedupe_event(self, sensor_id: int, event_code: str, dedupe_minutes: int) -> bool:
        """Check if event should be deduplicated."""
        row = self._conn.execute(
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
