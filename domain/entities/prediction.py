"""Re-export facade — backward compatibility.

Canonical location: ``domain.entities.results.prediction``
"""

from .results.prediction import Prediction, PredictionConfidence

__all__ = ["Prediction", "PredictionConfidence"]
