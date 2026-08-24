"""Statistical inference — Bayesian methods.

Bayesian Inference:
- BayesianUpdater: conjugate prior-posterior updates
- NaiveBayesClassifier: online multi-class classification
- ProbabilityCalibrator: Platt scaling for score calibration
"""

from .bayesian import (
    BayesianUpdater,
    NaiveBayesClassifier,
    ProbabilityCalibrator,
    Prior,
    Posterior,
)

__all__ = [
    "BayesianUpdater",
    "NaiveBayesClassifier",
    "ProbabilityCalibrator",
    "Prior",
    "Posterior",
]