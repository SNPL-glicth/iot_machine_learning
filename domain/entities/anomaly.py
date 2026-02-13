"""Re-export facade — backward compatibility.

Canonical location: ``domain.entities.results.anomaly``
"""

from .results.anomaly import AnomalyResult, AnomalySeverity

__all__ = ["AnomalyResult", "AnomalySeverity"]
