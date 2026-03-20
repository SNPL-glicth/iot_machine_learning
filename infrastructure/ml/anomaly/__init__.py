"""Detectores de anomalías — implementaciones concretas de AnomalyDetectionPort.

Arquitectura modular (reorganizada 2026-03-20):
- ``core/`` → VotingAnomalyDetector, SubDetector, AnomalyDetectorConfig
- ``scoring/`` → Funciones de scoring puras, estadísticas de entrenamiento
- ``voting/`` → VotingStrategy, construcción de contexto de votos
- ``factory/`` → create_default_detectors
- ``narration/`` → build_anomaly_explanation
- ``detectors/`` → Sub-detectores individuales (ZScore, IQR, IF, LOF, Temporal)

Public API (backward compatible):
    from infrastructure.ml.anomaly import (
        VotingAnomalyDetector,
        AnomalyDetectorConfig,
        SubDetector,
        DetectorRegistry,
        register_detector,
        VotingStrategy,
        create_default_detectors,
    )
"""

from .core import (
    VotingAnomalyDetector,
    AnomalyDetectorConfig,
    SubDetector,
    DetectorRegistry,
    register_detector,
)
from .voting import VotingStrategy
from .factory import create_default_detectors

__all__ = [
    "VotingAnomalyDetector",
    "AnomalyDetectorConfig",
    "SubDetector",
    "DetectorRegistry",
    "register_detector",
    "VotingStrategy",
    "create_default_detectors",
]
