"""Dual-write storage adapter: escribe a dbo.predictions + zenin_ml.predictions.

Estrategia de migración gradual:
1. Escribe a ambas tablas (legacy + zenin_ml)
2. Valida que zenin_ml funciona correctamente
3. Eventualmente deprecar dbo.predictions

Este adapter es un decorator sobre SqlServerStorageAdapter.
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

logger = logging.getLogger(__name__)


class DualWriteStorageAdapter(StoragePort):
    """Storage adapter que escribe a dbo.predictions (legacy) + zenin_ml.predictions (nuevo).
    
    Características:
    - Escribe a ambas tablas en paralelo
    - Legacy (dbo) es source of truth (si falla, no continúa)
    - Zenin ML es fail-safe (si falla, solo logea warning)
    - Delega todas las demás operaciones a SqlServerStorageAdapter
    """
    
    def __init__(self, conn: Connection, enable_zenin_ml: bool = True):
        """Inicializa dual-write adapter.
        
        Args:
            conn: Conexión SQLAlchemy
            enable_zenin_ml: Si False, solo escribe a legacy (para rollback)
        """
        self._legacy = SqlServerStorageAdapter(conn)
        self._zenin_ml = ZeninMLStorageAdapter(conn) if enable_zenin_ml else None
        self._enable_zenin_ml = enable_zenin_ml
    
    def save_prediction(
        self,
        prediction: Prediction,
        *,
        horizon_minutes_per_step: int = 10,
    ) -> int:
        """Guarda predicción en ambas tablas (dual-write).
        
        Args:
            prediction: Predicción del dominio
            horizon_minutes_per_step: Minutos por paso de horizonte
        
        Returns:
            ID del registro legacy (dbo.predictions)
        """
        # 1. Escribir a legacy (source of truth)
        legacy_id = self._legacy.save_prediction(
            prediction,
            horizon_minutes_per_step=horizon_minutes_per_step,
        )
        
        # 2. Escribir a zenin_ml (fail-safe)
        if self._zenin_ml is not None:
            try:
                zenin_id = self._zenin_ml.save_prediction(
                    prediction,
                    horizon_minutes_per_step=horizon_minutes_per_step,
                )
                logger.debug(
                    "dual_write_success",
                    extra={
                        "legacy_id": legacy_id,
                        "zenin_id": str(zenin_id),
                        "series_id": prediction.series_id,
                    },
                )
            except Exception as exc:
                logger.warning(
                    "zenin_ml_write_failed_failsafe",
                    extra={
                        "legacy_id": legacy_id,
                        "series_id": prediction.series_id,
                        "error": str(exc),
                    },
                )
        
        return legacy_id
    
    # Delegar todas las demás operaciones a legacy
    def load_sensor_window(self, sensor_id: int, limit: int = 500) -> SensorWindow:
        return self._legacy.load_sensor_window(sensor_id, limit)
    
    def list_active_sensor_ids(self) -> List[int]:
        return self._legacy.list_active_sensor_ids()
    
    def get_sensor_metadata(self, sensor_id: int) -> Dict[str, object]:
        return self._legacy.get_sensor_metadata(sensor_id)
    
    def get_device_id_for_sensor(self, sensor_id: int) -> int:
        return self._legacy.get_device_id_for_sensor(sensor_id)
    
    def get_latest_prediction(self, sensor_id: int) -> Optional[Prediction]:
        return self._legacy.get_latest_prediction(sensor_id)
    
    def save_anomaly_event(
        self,
        anomaly: AnomalyResult,
        prediction_id: Optional[int] = None,
    ) -> int:
        return self._legacy.save_anomaly_event(anomaly, prediction_id)
    
    def record_prediction_error(
        self,
        series_id: str,
        engine_name: str,
        predicted_value: float,
        actual_value: float,
    ) -> None:
        return self._legacy.record_prediction_error(
            series_id, engine_name, predicted_value, actual_value
        )
    
    def get_rolling_performance(
        self,
        series_id: str,
        engine_name: str,
        window_size: int = 50,
    ) -> Optional[Dict[str, float]]:
        return self._legacy.get_rolling_performance(series_id, engine_name, window_size)
    
    def compute_confidence_interval(
        self,
        series_id: str,
        engine_name: str,
        predicted_value: float,
        window_size: int = 50,
    ) -> tuple[float, float]:
        return self._legacy.compute_confidence_interval(
            series_id, engine_name, predicted_value, window_size
        )
    
    def _get_or_create_model_id(self, sensor_id: int, engine_name: str) -> int:
        return self._legacy._get_or_create_model_id(sensor_id, engine_name)
    
    def record_plasticity_event(
        self,
        series_id: str,
        event_type: str,
        metadata: Dict[str, object],
    ) -> None:
        return self._legacy.record_plasticity_event(series_id, event_type, metadata)
    
    def get_plasticity_history(
        self,
        series_id: str,
        limit: int = 100,
    ) -> List[Dict[str, object]]:
        return self._legacy.get_plasticity_history(series_id, limit)
