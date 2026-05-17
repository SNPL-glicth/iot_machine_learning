"""RolloutPredictionPortBridge — bridge con rollout gradual por sensor_id.

Envuelve un MoEPredictionEngine y decide basado en series_id si:
- Usa el engine MoE (sensor en grupo de tratamiento)
- Delega al fallback directamente (sensor en grupo de control)

Determinista: mismo series_id → mismo grupo siempre.
"""

from __future__ import annotations

from typing import Optional

from iot_machine_learning.domain.ports.prediction_port import PredictionPort
from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow

from .rollout_decider import RolloutDecider


class RolloutPredictionPortBridge(PredictionPort):
    """Bridge que aplica rollout gradual antes de delegar al engine MoE.

    Args:
        engine: MoEPredictionEngine ( PredictionPort vía as_port() ).
        fallback: Engine fallback para grupo de control.
        decider: RolloutDecider configurado.
    """

    def __init__(
        self,
        engine: PredictionPort,
        fallback: PredictionPort,
        decider: RolloutDecider,
    ) -> None:
        self._engine = engine
        self._fallback = fallback
        self._decider = decider

    @property
    def name(self) -> str:
        return f"moe_rollout({self._decider.percent}%)"

    def predict(self, window: SensorWindow) -> Prediction:
        series_id = getattr(window, "series_id", "unknown")
        if self._decider.is_enabled(series_id):
            return self._engine.predict(window)
        return self._fallback.predict(window)

    def can_handle(self, n_points: int) -> bool:
        return self._engine.can_handle(n_points) or self._fallback.can_handle(n_points)
