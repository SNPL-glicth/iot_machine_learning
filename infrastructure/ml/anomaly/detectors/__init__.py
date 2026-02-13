"""Sub-detectores individuales para el ensemble de anomalías.

Cada detector tiene UNA responsabilidad: producir un voto [0, 1].

Módulos:
- ``z_score_detector.py`` — Detección por Z-score de magnitud.
- ``iqr_detector.py`` — Detección por rango intercuartílico.
- ``isolation_forest_detector.py`` — IsolationForest (sklearn).
- ``lof_detector.py`` — LocalOutlierFactor (sklearn).
- ``temporal_z_detector.py`` — Z-score de velocidad y aceleración.
"""

from .z_score_detector import ZScoreDetector
from .iqr_detector import IQRDetector
from .isolation_forest_detector import IsolationForestDetector
from .lof_detector import LOFDetector
from .temporal_z_detector import VelocityZDetector, AccelerationZDetector

__all__ = [
    "ZScoreDetector",
    "IQRDetector",
    "IsolationForestDetector",
    "LOFDetector",
    "VelocityZDetector",
    "AccelerationZDetector",
]
