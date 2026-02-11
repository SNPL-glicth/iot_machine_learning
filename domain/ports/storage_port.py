"""Port de almacenamiento — contrato para persistencia.

Desacopla el dominio de SQL Server, Redis o cualquier otro backend.
Los adaptadores implementan este port en la capa de infraestructura.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..entities.anomaly import AnomalyResult
from ..entities.prediction import Prediction
from ..entities.sensor_reading import SensorReading, SensorWindow


class StoragePort(ABC):
    """Contrato para persistencia de datos ML.

    Operaciones agrupadas por entidad.  Cada implementación concreta
    (SQL Server, Redis, archivo) decide cómo mapear a su backend.
    """

    # --- Lecturas de sensores ---

    @abstractmethod
    def load_sensor_window(
        self,
        sensor_id: int,
        limit: int = 500,
    ) -> SensorWindow:
        """Carga las últimas ``limit`` lecturas de un sensor.

        Args:
            sensor_id: ID del sensor.
            limit: Máximo de lecturas a cargar.

        Returns:
            ``SensorWindow`` con lecturas ordenadas cronológicamente.
        """
        ...

    @abstractmethod
    def list_active_sensor_ids(self) -> List[int]:
        """Retorna IDs de todos los sensores activos.

        Returns:
            Lista de sensor_ids con ``is_active = 1``.
        """
        ...

    # --- Predicciones ---

    @abstractmethod
    def save_prediction(self, prediction: Prediction) -> int:
        """Persiste una predicción.

        Args:
            prediction: Predicción del dominio.

        Returns:
            ID de la predicción persistida.
        """
        ...

    @abstractmethod
    def get_latest_prediction(self, sensor_id: int) -> Optional[Prediction]:
        """Obtiene la última predicción de un sensor.

        Args:
            sensor_id: ID del sensor.

        Returns:
            ``Prediction`` o ``None`` si no existe.
        """
        ...

    # --- Anomalías ---

    @abstractmethod
    def save_anomaly_event(
        self,
        anomaly: AnomalyResult,
        prediction_id: Optional[int] = None,
    ) -> int:
        """Persiste un evento de anomalía.

        Args:
            anomaly: Resultado de detección.
            prediction_id: ID de la predicción asociada (si existe).

        Returns:
            ID del evento persistido.
        """
        ...

    # --- Metadata de sensores ---

    @abstractmethod
    def get_sensor_metadata(self, sensor_id: int) -> Dict[str, object]:
        """Obtiene metadata de un sensor (tipo, ubicación, device_id, etc.).

        Args:
            sensor_id: ID del sensor.

        Returns:
            Dict con metadata del sensor.
        """
        ...

    @abstractmethod
    def get_device_id_for_sensor(self, sensor_id: int) -> int:
        """Obtiene el device_id al que pertenece un sensor.

        Args:
            sensor_id: ID del sensor.

        Returns:
            device_id.
        """
        ...
