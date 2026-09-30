"""Core anomaly detection components.

Components:
    - VotingAnomalyDetector: Main ensemble orchestrator
    - SubDetector: Protocol for sub-detectors
    - DetectorRegistry: Auto-discovery registry
    - AnomalyDetectorConfig: Configuration dataclass
"""

from .config import AnomalyDetectorConfig
from .detector import VotingAnomalyDetector
from .protocol import DetectorRegistry, SubDetector, register_detector

__all__ = [
    "VotingAnomalyDetector",
    "SubDetector",
    "DetectorRegistry",
    "register_detector",
    "AnomalyDetectorConfig",
]
