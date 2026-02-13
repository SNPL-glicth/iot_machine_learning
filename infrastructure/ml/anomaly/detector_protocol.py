"""Protocolo interno para sub-detectores del ensemble de anomalías.

Cada sub-detector tiene UNA responsabilidad: producir un voto [0, 1]
para un valor dado, basándose en su propia lógica de entrenamiento.

No confundir con ``AnomalyDetectionPort`` (port de dominio que retorna
``AnomalyResult``).  Este protocolo es interno a la capa de infraestructura.

.. versionchanged:: 2.0
    Added ``DetectorRegistry`` and ``register_detector`` decorator
    for auto-registration of sub-detectors (ROB-2).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class SubDetector(ABC):
    """Contrato para un sub-detector individual del ensemble.

    Un sub-detector = una responsabilidad de detección.
    """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Nombre del método (para votos y logging)."""
        ...

    @abstractmethod
    def train(self, values: List[float], **kwargs: object) -> None:
        """Entrena el sub-detector con datos históricos.

        Args:
            values: Serie temporal de entrenamiento.
            **kwargs: Parámetros adicionales (timestamps, etc.).
        """
        ...

    @abstractmethod
    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        """Produce un voto [0, 1] para un valor.

        Args:
            value: Valor a evaluar.
            **kwargs: Contexto adicional (window, temporal_features, etc.).

        Returns:
            Voto en [0, 1], o ``None`` si no puede votar.
        """
        ...

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """``True`` si el sub-detector fue entrenado."""
        ...


class DetectorRegistry:
    """Registry for sub-detector classes.

    Enables auto-discovery of sub-detectors via the
    ``@register_detector`` decorator.  New detectors self-register
    at class definition time.

    Usage::

        # List all registered detector names
        DetectorRegistry.list_detectors()

        # Create all registered detectors with a config
        detectors = DetectorRegistry.create_all(config)
    """

    _registry: Dict[str, Callable[..., SubDetector]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable[..., SubDetector],
    ) -> None:
        """Register a detector factory by name."""
        cls._registry[name] = factory
        logger.debug(
            "detector_registered",
            extra={"detector_name": name},
        )

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._registry.pop(name, None)

    @classmethod
    def list_detectors(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def create_all(
        cls,
        config: object,
    ) -> List[SubDetector]:
        """Instantiate all registered detectors.

        Each factory receives the config object and returns a
        ``SubDetector`` instance.

        Args:
            config: Configuration object (typically ``AnomalyDetectorConfig``).

        Returns:
            List of instantiated sub-detectors.
        """
        detectors: List[SubDetector] = []
        for name, factory in cls._registry.items():
            try:
                detectors.append(factory(config))
            except Exception as exc:
                logger.warning(
                    "detector_creation_failed",
                    extra={"detector": name, "error": str(exc)},
                )
        return detectors


def register_detector(name: str):
    """Class decorator: auto-register a ``SubDetector`` factory.

    The decorated class must accept an ``AnomalyDetectorConfig`` as
    its first positional argument (a factory callable).

    Usage::

        @register_detector("my_detector")
        def _create_my_detector(config):
            return MyDetector(threshold=config.z_vote_upper)

    Or with a class that has a ``from_config`` classmethod::

        @register_detector("my_detector")
        class MyDetector(SubDetector):
            @classmethod
            def from_config(cls, config):
                return cls(threshold=config.z_vote_upper)

    Args:
        name: Registry name for the detector.
    """
    def decorator(factory_or_cls):
        if callable(factory_or_cls):
            DetectorRegistry.register(name, factory_or_cls)
        return factory_or_cls
    return decorator
