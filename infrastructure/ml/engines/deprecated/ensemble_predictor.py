# DEPRECATED: Superseded by WeightedFusion + InhibitionGate
# Kept for reference only. Do not use in production.
# Will be removed in next major version.

"""EnsembleWeightedPredictor — thin shell.

Logic moved to ensemble_prediction.py and ensemble_weights.py
to keep files under 180 lines.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Optional

from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.ports.prediction_port import PredictionPort

from .ensemble_prediction import run_ensemble_prediction
from .ensemble_weights import update_ensemble_weights

logger = logging.getLogger(__name__)

_MAX_ERROR_HISTORY: int = 100


class EnsembleWeightedPredictor(PredictionPort):
    """Ensemble que combina múltiples engines con pesos dinámicos."""

    def __init__(
        self,
        engines: List[PredictionPort],
        initial_weights: Optional[List[float]] = None,
        adapt_weights: bool = True,
        series_id: Optional[str] = None,
        domain_type: str = "sensor",
        auto_load_weights: bool = True,
    ) -> None:
        if not engines:
            raise ValueError("Se requiere al menos un engine para ensemble")

        self._engines = engines
        self._adapt_weights = adapt_weights
        self._update_count: int = 0
        self._series_id = series_id
        self._domain_type = domain_type

        self._weights_repo = None
        if series_id:
            try:
                from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.ensemble_weights_repository import (
                    EnsembleWeightsRepository,
                )
                self._weights_repo = EnsembleWeightsRepository()
            except Exception as exc:
                logger.warning(
                    "ensemble_weights_repo_init_failed",
                    extra={"error": str(exc)},
                )

        loaded_weights = None
        if auto_load_weights and self._weights_repo and series_id:
            loaded_weights = self._weights_repo.load_weights(series_id)

        if loaded_weights:
            self._weights = [
                loaded_weights.get(engine.name, 1.0 / len(engines))
                for engine in engines
            ]
            total = sum(self._weights)
            self._weights = [w / total for w in self._weights]
        elif initial_weights is not None:
            if len(initial_weights) != len(engines):
                raise ValueError(
                    f"Pesos ({len(initial_weights)}) no coinciden con "
                    f"engines ({len(engines)})"
                )
            total = sum(initial_weights)
            self._weights = [w / total for w in initial_weights]
        else:
            n = len(engines)
            self._weights = [1.0 / n] * n

        self._engine_errors: Dict[str, Deque[float]] = {
            engine.name: deque(maxlen=_MAX_ERROR_HISTORY)
            for engine in engines
        }

    @property
    def name(self) -> str:
        return "ensemble_weighted"

    def can_handle(self, n_points: int) -> bool:
        return any(e.can_handle(n_points) for e in self._engines)

    def predict(self, window) -> Prediction:
        return run_ensemble_prediction(self, window)

    def update_weights(
        self, actual_value: float, predictions: List[Optional[Prediction]]
    ) -> None:
        update_ensemble_weights(self, actual_value, predictions)

    def supports_confidence_interval(self) -> bool:
        return False

    @property
    def current_weights(self) -> Dict[str, float]:
        return {
            self._engines[i].name: self._weights[i]
            for i in range(len(self._engines))
        }
