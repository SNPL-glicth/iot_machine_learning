"""Detectores de anomalías — implementaciones concretas de AnomalyDetectionPort.

Arquitectura modular:
- ``anomaly_config.py`` → AnomalyDetectorConfig (configuración centralizada)
- ``detector_protocol.py`` → SubDetector (contrato para sub-detectores)
- ``detectors/`` → Sub-detectores individuales (ZScore, IQR, IF, LOF, Temporal)
- ``voting_strategy.py`` → VotingStrategy (combinación de votos)
- ``voting_anomaly_detector.py`` → VotingAnomalyDetector (orquestador)
- ``anomaly_narrator.py`` → build_anomaly_explanation (narrativa)
- ``training_stats.py`` → TrainingStats + compute_training_stats
- ``scoring_functions.py`` → Funciones de scoring puras
- ``temporal_stats.py`` → TemporalTrainingStats + compute_temporal_training_stats
"""

from .anomaly_config import AnomalyDetectorConfig
from .voting_anomaly_detector import VotingAnomalyDetector, create_default_detectors
from .voting_strategy import VotingStrategy
from .detector_protocol import DetectorRegistry, SubDetector, register_detector
