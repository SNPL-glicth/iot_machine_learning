"""Factory para motores de predicción UTSAE con registro dinámico.

Responsabilidad ÚNICA: registrar y crear motores por nombre.
Lógica de selección por feature flags en application/use_cases/select_engine.py.

.. versionchanged:: 2.0
    Added ``register_engine`` decorator for auto-registration (ROB-1).
    Added ``discover_engines`` for plugin discovery.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Type

from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionEnginePortBridge,
    PredictionResult,
)
from iot_machine_learning.infrastructure.ml.engines.seasonal.engine import SeasonalPredictorEngine

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

        from iot_machine_learning.infrastructure.ml.engines.baseline.engine import (
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
    def create_as_port(cls, name: str, **kwargs: object) -> PredictionEnginePortBridge:
        """Create an engine and wrap it as ``PredictionPort``.

        Convenience method that combines ``create()`` + ``as_port()``.
        """
        engine = cls.create(name, **kwargs)
        return engine.as_port()

    @classmethod
    def get_engine_for_sensor(
        cls,
        sensor_id: int,
        flags: object,
    ) -> PredictionEngine:
        """Selecciona engine según feature flags.

        .. deprecated:: 2.0
            Use ``application.use_cases.select_engine.select_engine_for_sensor``
            + ``EngineFactory.create()`` at the application layer instead.
            This method will be removed in a future version.
        """
        warnings.warn(
            "EngineFactory.get_engine_for_sensor() is deprecated. "
            "Use select_engine_for_sensor() + EngineFactory.create() "
            "at the application layer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Fallback to baseline — no longer imports from application layer
        return BaselineMovingAverageEngine()


def register_engine(name: str):
    """Class decorator: auto-register a ``PredictionEngine`` with ``EngineFactory``.

    Usage::

        @register_engine("my_engine")
        class MyEngine(PredictionEngine):
            ...

    The engine is registered at class definition time — no need to
    touch ``EngineFactory`` or ``engines/__init__.py``.

    Args:
        name: Registry name for the engine.
    """
    def decorator(cls: Type[PredictionEngine]) -> Type[PredictionEngine]:
        EngineFactory.register(name, cls)
        return cls
    return decorator


def discover_engines(package_path: str) -> List[str]:
    """Import all modules in a package to trigger ``@register_engine`` decorators.

    This enables plugin discovery: drop a new ``.py`` file with a
    ``@register_engine`` decorated class into the engines package,
    and it will be auto-registered.

    Args:
        package_path: Dotted package path (e.g.
            ``"iot_machine_learning.infrastructure.ml.engines"``).

    Returns:
        List of newly discovered engine names.
    """
    import importlib
    import pkgutil

    before = set(EngineFactory.list_engines())
    try:
        pkg = importlib.import_module(package_path)
        if hasattr(pkg, "__path__"):
            for _importer, modname, _ispkg in pkgutil.iter_modules(pkg.__path__):
                try:
                    importlib.import_module(f"{package_path}.{modname}")
                except Exception as exc:
                    logger.debug(
                        "engine_discovery_skip",
                        extra={"module": modname, "error": str(exc)},
                    )
    except Exception as exc:
        logger.warning(
            "engine_discovery_failed",
            extra={"package": package_path, "error": str(exc)},
        )
    after = set(EngineFactory.list_engines())
    return sorted(after - before)


# --- Auto-registro de engines al importar ---
EngineFactory.register("baseline_moving_average", BaselineMovingAverageEngine)
EngineFactory.register("seasonal_fft", SeasonalPredictorEngine)
