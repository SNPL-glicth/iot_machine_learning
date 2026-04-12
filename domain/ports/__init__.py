"""Ports (interfaces) del dominio UTSAE.

Definen contratos que la capa de infraestructura debe implementar.
Dependencias hacia adentro: Infrastructure implementa estos ports,
Domain los consume sin conocer la implementación.
"""

from .prediction_port import PredictionPort
from .anomaly_detection_port import AnomalyDetectionPort
from .pattern_detection_port import PatternDetectionPort
from .storage_port import StoragePort
from .audit_port import AuditPort
from .cognitive_memory_port import CognitiveMemoryPort
from .sliding_window_port import ISlidingWindowStore, WindowConfig
from .plasticity_port import PlasticityPort
from .experiment_tracker_port import ExperimentTrackerPort, NullExperimentTracker
from .confidence_calibrator_port import (
    CalibrationStats,
    CalibratedScore,
    ConfidenceCalibratorPort,
    NullCalibrator,
)
from .recent_anomaly_tracker_port import (
    RecentAnomalyTrackerPort,
    NullAnomalyTracker,
)

__all__ = [
    "PredictionPort",
    "AnomalyDetectionPort",
    "PatternDetectionPort",
    "StoragePort",
    "AuditPort",
    "CognitiveMemoryPort",
    "ISlidingWindowStore",
    "WindowConfig",
    "PlasticityPort",
    "ExperimentTrackerPort",
    "NullExperimentTracker",
    "ConfidenceCalibratorPort",
    "CalibrationStats",
    "CalibratedScore",
    "NullCalibrator",
    "RecentAnomalyTrackerPort",
    "NullAnomalyTracker",
]
