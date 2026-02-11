"""Factory para motores de predicción UTSAE con registro dinámico.

Responsabilidad ÚNICA: registrar y crear motores por nombre.
Lógica de selección por feature flags en application/use_cases/select_engine.py.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

from iot_machine_learning.infrastructure.ml.interfaces import PredictionEngine, PredictionResult

logger = logging.getLogger(__name__)


class BaselineMovingAverageEngine(PredictionEngine):
    """Motor baseline: media móvil simple (fallback)."""

    @property
    def name(self) -> str:
        return "baseline_moving_average"

    def can_handle(self, n_points: int) -> bool:
        return n_points >= 1

    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        if not values:
            raise ValueError("values no puede estar vacía")

        from iot_machine_learning.infrastructure.ml.engines.baseline_engine import (
            BaselineConfig,
            predict_moving_average,
        )

        cfg = BaselineConfig(window=len(values))
        predicted, confidence = predict_moving_average(values, cfg)

        return PredictionResult(
            predicted_value=predicted,
            confidence=confidence,
            trend="stable",
            metadata={
                "window": len(values),
                "fallback": None,
                "clamped": False,
            },
        )


class EngineFactory:
    """Factory + Registry para motores de predicción.

    Responsabilidad ÚNICA: registrar y crear motores por nombre.
    NO contiene lógica de selección — eso está en select_engine.py.
    """

    _registry: Dict[str, Type[PredictionEngine]] = {}

    @classmethod
    def register(cls, name: str, engine_class: Type[PredictionEngine]) -> None:
        if not (isinstance(engine_class, type) and issubclass(engine_class, PredictionEngine)):
            raise TypeError(
                f"{engine_class} no es subclase de PredictionEngine"
            )
        cls._registry[name] = engine_class
        logger.debug(
            "engine_registered",
            extra={"engine_name": name, "class": engine_class.__name__},
        )

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._registry.pop(name, None)

    @classmethod
    def list_engines(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, **kwargs: object) -> PredictionEngine:
        engine_class = cls._registry.get(name)

        if engine_class is None:
            logger.warning(
                "engine_not_found_fallback",
                extra={"requested": name, "fallback": "baseline_moving_average"},
            )
            return BaselineMovingAverageEngine()

        try:
            engine = engine_class(**kwargs)  # type: ignore[call-arg]
            logger.debug(
                "engine_created",
                extra={"engine_name": name, "kwargs": str(kwargs)},
            )
            return engine
        except Exception:
            logger.exception(
                "engine_creation_failed_fallback",
                extra={"requested": name, "fallback": "baseline_moving_average"},
            )
            return BaselineMovingAverageEngine()

    @classmethod
    def get_engine_for_sensor(
        cls,
        sensor_id: int,
        flags: object,
    ) -> PredictionEngine:
        """Selecciona engine según feature flags.

        .. deprecated::
            Usar ``application.use_cases.select_engine.select_engine_for_sensor``
            + ``EngineFactory.create()`` en su lugar.
        """
        from iot_machine_learning.application.use_cases.select_engine import (
            select_engine_for_sensor,
        )
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags

        if not isinstance(flags, FeatureFlags):
            logger.warning(
                "invalid_flags_type_fallback",
                extra={"type": type(flags).__name__},
            )
            return BaselineMovingAverageEngine()

        selection = select_engine_for_sensor(sensor_id, flags)
        return cls.create(selection["engine_name"], **selection["kwargs"])


# --- Auto-registro del baseline al importar ---
EngineFactory.register("baseline_moving_average", BaselineMovingAverageEngine)
