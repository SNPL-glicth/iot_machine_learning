"""RUL (Remaining Useful Life) module for ZENIN anomaly pipeline.

Estimates time-to-failure from deterioration signals and
produces human-readable narratives for operators.
"""

from .estimator import RULEstimator
from .models import RULEstimate
from .narrator import RULNarrator

__all__ = ["RULEstimator", "RULEstimate", "RULNarrator"]
