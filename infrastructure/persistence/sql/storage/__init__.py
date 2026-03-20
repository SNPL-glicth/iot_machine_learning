"""SQL Server Storage Adapter - Modularized implementation of StoragePort."""

from __future__ import annotations

from typing import Dict, List, Optional

from sqlalchemy.engine import Connection

from iot_machine_learning.domain.entities.anomaly import AnomalyResult
from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from iot_machine_learning.domain.ports.storage_port import StoragePort

from .base_queries import BaseQueries
from .prediction_queries import PredictionQueries
from .anomaly_queries import AnomalyQueries
from .performance_queries import PerformanceQueries
from .plasticity_queries import PlasticityQueries


class SqlServerStorageAdapter(StoragePort):
    """Facade that combines all SQL storage modules.
    
    Maintains 100% backward compatibility with original SqlServerStorageAdapter.
    Delegates to specialized query modules for better organization.
    """
    
    def __init__(self, conn: Connection) -> None:
        self._conn = conn
        self._base = BaseQueries(conn)
        self._predictions = PredictionQueries(conn)
        self._anomalies = AnomalyQueries(conn)
        self._performance = PerformanceQueries(conn)
        self._plasticity = PlasticityQueries(conn)
    
    # Delegate to base queries
    def load_sensor_window(self, sensor_id: int, limit: int = 500) -> SensorWindow:
        return self._base.load_sensor_window(sensor_id, limit)
    
    def list_active_sensor_ids(self) -> List[int]:
        return self._base.list_active_sensor_ids()
    
    def get_sensor_metadata(self, sensor_id: int) -> Dict[str, object]:
        return self._base.get_sensor_metadata(sensor_id)
    
    def get_device_id_for_sensor(self, sensor_id: int) -> int:
        return self._base.get_device_id_for_sensor(sensor_id)
    
    # Delegate to prediction queries
    def save_prediction(
        self,
        prediction: Prediction,
        *,
        horizon_minutes_per_step: int = 10,
    ) -> int:
        return self._predictions.save_prediction(
            prediction,
            self.get_device_id_for_sensor,
            horizon_minutes_per_step=horizon_minutes_per_step,
        )
    
    def get_latest_prediction(self, sensor_id: int) -> Optional[Prediction]:
        return self._predictions.get_latest_prediction(sensor_id)
    
    # Delegate to anomaly queries
    def save_anomaly_event(
        self,
        anomaly: AnomalyResult,
        prediction_id: Optional[int] = None,
    ) -> int:
        return self._anomalies.save_anomaly_event(
            anomaly,
            self.get_device_id_for_sensor,
            prediction_id,
        )
    
    # Delegate to performance queries
    def record_prediction_error(
        self,
        series_id: str,
        engine_name: str,
        predicted_value: float,
        actual_value: float,
    ) -> None:
        return self._performance.record_prediction_error(
            series_id, engine_name, predicted_value, actual_value
        )
    
    def get_rolling_performance(
        self,
        series_id: str,
        engine_name: str,
        window_size: int = 50,
    ) -> Optional[Dict[str, float]]:
        return self._performance.get_rolling_performance(
            series_id, engine_name, window_size
        )
    
    def compute_confidence_interval(
        self,
        series_id: str,
        engine_name: str,
        predicted_value: float,
        confidence_level: float = 0.95,
    ) -> Optional[Dict[str, float]]:
        return self._performance.compute_confidence_interval(
            series_id, engine_name, predicted_value, confidence_level
        )
    
    # Delegate to plasticity queries
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
        return self._plasticity.record_contextual_error(
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
        return self._plasticity.get_contextual_performance(
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
        return self._plasticity.update_engine_health(
            series_id, engine_name, consecutive_failures, consecutive_successes,
            total_predictions, total_errors, last_error, failure_rate,
            is_inhibited, inhibition_reason, last_success_time, last_failure_time
        )
    
    def get_engine_health(
        self,
        series_id: str,
        engine_name: str,
    ) -> Optional[Dict[str, any]]:
        return self._plasticity.get_engine_health(series_id, engine_name)


__all__ = ['SqlServerStorageAdapter']
