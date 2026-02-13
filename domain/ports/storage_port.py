"""Port de almacenamiento — contrato para persistencia.

Desacopla el dominio de SQL Server, Redis o cualquier otro backend.
Los adaptadores implementan este port en la capa de infraestructura.

Dual interface:
- Métodos ``sensor_id: int`` (legacy IoT) — abstractos, deben implementarse.
- Métodos ``series_id: str`` (agnósticos) — default bridge a legacy.
  Implementaciones nuevas pueden override para soporte nativo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..entities.anomaly import AnomalyResult
from ..entities.prediction import Prediction
from ..entities.sensor_reading import SensorReading, SensorWindow
from ..entities.time_series import TimeSeries
from ..validators.input_guard import safe_series_id_to_int


class StoragePort(ABC):
    """Contrato para persistencia de datos ML.

    Operaciones agrupadas por entidad.  Cada implementación concreta
    (SQL Server, Redis, archivo) decide cómo mapear a su backend.

    Dual interface:
        - ``load_sensor_window(sensor_id: int)`` — legacy IoT.
        - ``load_series_window(series_id: str)`` — agnóstico.
          Default bridge: convierte ``series_id`` a ``int`` y delega.
    """

    # ── Legacy IoT interface (sensor_id: int) ─────────────────────────

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

    # ── Agnostic interface (series_id: str) ───────────────────────────
    # Default bridges convert series_id → sensor_id and delegate.
    # Implementations may override for native series_id support.

    def load_series_window(
        self,
        series_id: str,
        limit: int = 500,
    ) -> TimeSeries:
        """Carga las últimas ``limit`` lecturas de una serie.

        Default bridge: convierte a ``int`` y delega a ``load_sensor_window``,
        luego convierte ``SensorWindow`` → ``TimeSeries``.

        Args:
            series_id: Identificador de la serie (agnóstico).
            limit: Máximo de lecturas a cargar.

        Returns:
            ``TimeSeries`` con puntos ordenados cronológicamente.
        """
        sensor_id = safe_series_id_to_int(series_id)
        window = self.load_sensor_window(sensor_id, limit)
        return window.to_time_series()

    def list_active_series_ids(self) -> List[str]:
        """Retorna IDs de todas las series activas.

        Default bridge: convierte ``List[int]`` → ``List[str]``.

        Returns:
            Lista de series_ids activos.
        """
        return [str(sid) for sid in self.list_active_sensor_ids()]

    def get_latest_prediction_for_series(
        self, series_id: str
    ) -> Optional[Prediction]:
        """Obtiene la última predicción de una serie.

        Default bridge: convierte a ``int`` y delega.

        Args:
            series_id: Identificador de la serie.

        Returns:
            ``Prediction`` o ``None`` si no existe.
        """
        sensor_id = safe_series_id_to_int(series_id)
        return self.get_latest_prediction(sensor_id)

    def get_series_metadata(self, series_id: str) -> Dict[str, object]:
        """Obtiene metadata de una serie.

        Default bridge: convierte a ``int`` y delega.

        Args:
            series_id: Identificador de la serie.

        Returns:
            Dict con metadata.
        """
        sensor_id = safe_series_id_to_int(series_id)
        return self.get_sensor_metadata(sensor_id)
