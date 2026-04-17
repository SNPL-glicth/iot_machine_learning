"""Storage adapter que lee del legacy (dbo) pero escribe SOLO a zenin_ml.predictions.

Este adapter es para migración completa a zenin_db:
- Lee datos de sensores desde dbo (legacy) - SqlServerStorageAdapter
- Escribe predicciones SOLO a zenin_ml.predictions - ZeninMLStorageAdapter
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from sqlalchemy.engine import Connection

from iot_machine_learning.domain.entities.anomaly import AnomalyResult
from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from iot_machine_learning.domain.ports.storage_port import StoragePort

from .storage import SqlServerStorageAdapter
from .zenin_ml_storage import ZeninMLStorageAdapter
from .zenin_db_connection import ZeninDbConnection

logger = logging.getLogger(__name__)


class ZeninMLOnlyStorageAdapter(StoragePort):
    """Adapter de migración: lee legacy, escribe solo a zenin_ml.
    
    Características:
    - Lecturas (load_sensor_window, etc.) desde dbo (legacy)
    - Escritura de predicciones SOLO a zenin_ml.predictions
    - No escribe más a dbo.predictions
    """
    
    def __init__(self, conn: Connection, tenant_id: Optional[str] = None):
        """Inicializa adapter.
        
        Args:
            conn: Conexión SQLAlchemy (para lecturas legacy desde iot_monitoring_system)
            tenant_id: UUID del tenant (opcional)
        """
        # Reader usa la conexión legacy (iot_monitoring_system)
        self._reader = SqlServerStorageAdapter(conn)
        self._conn = conn
        # Writer se inicializa lazy cuando se necesita
        self._writer = None
        self._tenant_id = tenant_id
    
    # === READ operations: delegate to legacy (dbo) ===
    
    def load_sensor_window(self, sensor_id: int, limit: int = 500) -> SensorWindow:
        return self._reader.load_sensor_window(sensor_id, limit)
    
    def list_active_sensor_ids(self) -> List[int]:
        return self._reader.list_active_sensor_ids()
    
    def get_sensor_metadata(self, sensor_id: int) -> Dict[str, object]:
        return self._reader.get_sensor_metadata(sensor_id)
    
    def get_device_id_for_sensor(self, sensor_id: int) -> int:
        return self._reader.get_device_id_for_sensor(sensor_id)
    
    def get_latest_prediction(self, sensor_id: int) -> Optional[Prediction]:
        return self._reader.get_latest_prediction(sensor_id)
    
    def save_anomaly_event(
        self,
        anomaly: AnomalyResult,
        prediction_id: Optional[int] = None,
    ) -> int:
        return self._reader.save_anomaly_event(anomaly, prediction_id)
    
    def record_prediction_error(
        self,
        series_id: str,
        engine_name: str,
        predicted_value: float,
        actual_value: float,
    ) -> None:
        return self._reader.record_prediction_error(
            series_id, engine_name, predicted_value, actual_value
        )
    
    def get_rolling_performance(
        self,
        series_id: str,
        engine_name: str,
        window_size: int = 50,
    ) -> Optional[Dict[str, float]]:
        return self._reader.get_rolling_performance(series_id, engine_name, window_size)
    
    def compute_confidence_interval(
        self,
        series_id: str,
        engine_name: str,
        predicted_value: float,
        confidence_level: float = 0.95,
    ) -> Optional[Dict[str, float]]:
        return self._reader.compute_confidence_interval(
            series_id, engine_name, predicted_value, confidence_level
        )
    
    def record_contextual_error(
        self,
        series_id: str,
        engine_name: str,
        predicted_value: float,
        actual_value: float,
        error: float,
        penalty: float,
        regime: str,
        noise_ratio: float,
        volatility: float,
        time_of_day: int,
        consecutive_failures: int,
        is_critical_zone: bool,
        context_key: str,
    ) -> None:
        return self._reader.record_contextual_error(
            series_id, engine_name, predicted_value, actual_value,
            error, penalty, regime, noise_ratio, volatility,
            time_of_day, consecutive_failures, is_critical_zone, context_key
        )
    
    def get_contextual_performance(
        self,
        series_id: str,
        engine_name: str,
        context_key: str,
        window_size: int = 50,
    ) -> Optional[Dict[str, float]]:
        return self._reader.get_contextual_performance(
            series_id, engine_name, context_key, window_size
        )
    
    def update_engine_health(
        self,
        series_id: str,
        engine_name: str,
        consecutive_failures: int,
        consecutive_successes: int,
        total_predictions: int,
        total_errors: int,
        last_error: float,
        failure_rate: float,
        is_inhibited: bool,
        inhibition_reason: Optional[str] = None,
        last_success_time: Optional[str] = None,
        last_failure_time: Optional[str] = None,
    ) -> None:
        return self._reader.update_engine_health(
            series_id, engine_name, consecutive_failures, consecutive_successes,
            total_predictions, total_errors, last_error, failure_rate,
            is_inhibited, inhibition_reason, last_success_time, last_failure_time
        )
    
    def get_engine_health(
        self,
        series_id: str,
        engine_name: str,
    ) -> Optional[Dict[str, any]]:
        return self._reader.get_engine_health(series_id, engine_name)
    
    def _get_writer(self):
        """Lazy initialization del writer con conexión a zenin_db."""
        if self._writer is None:
            # Crear conexión a zenin_db
            zenin_conn = ZeninDbConnection.get_engine().connect()
            self._writer = ZeninMLStorageAdapter(zenin_conn, tenant_id=self._tenant_id)
            self._zenin_conn = zenin_conn
        return self._writer
    
    # === WRITE operation: ONLY to zenin_ml ===
    
    def save_prediction(
        self,
        prediction: Prediction,
        *,
        horizon_minutes_per_step: int = 10,
    ) -> int:
        """Guarda predicción SOLO en zenin_ml.predictions (no en dbo).
        
        Returns:
            0 (placeholder, ya que zenin_ml usa UUID no int)
        """
        try:
            writer = self._get_writer()
            zenin_id = writer.save_prediction(
                prediction,
                horizon_minutes_per_step=horizon_minutes_per_step,
            )
            # Hacer commit explícito en la conexión de zenin_db
            if hasattr(self, '_zenin_conn'):
                self._zenin_conn.commit()
            logger.info(
                "zenin_ml_only_prediction_saved",
                extra={
                    "zenin_id": str(zenin_id),
                    "series_id": prediction.series_id,
                    "engine": prediction.engine_name,
                },
            )
            # Return 0 as placeholder (legacy code expects int)
            return 0
        except Exception as exc:
            logger.error(
                "zenin_ml_only_prediction_failed",
                extra={
                    "series_id": prediction.series_id,
                    "error": str(exc),
                },
            )
            # Rollback en caso de error
            if hasattr(self, '_zenin_conn'):
                try:
                    self._zenin_conn.rollback()
                except:
                    pass
            raise
