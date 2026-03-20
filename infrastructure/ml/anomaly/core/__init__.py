"""Core anomaly detection components.

Components:
    - VotingAnomalyDetector: Main ensemble orchestrator
    - SubDetector: Protocol for sub-detectors
    - DetectorRegistry: Auto-discovery registry
    - AnomalyDetectorConfig: Configuration dataclass
"""

from .detector import VotingAnomalyDetector
from .protocol import SubDetector, DetectorRegistry, register_detector
from .config import AnomalyDetectorConfig

__all__ = [
    "VotingAnomalyDetector",
    "SubDetector",
    "DetectorRegistry",
    "register_detector",
    "AnomalyDetectorConfig",
]
