"""Statistical inference — MLE + Bayesian methods.

Maximum Likelihood Estimation:
- MaximumLikelihoodEstimator: fit distributions to data
- Supported: Gaussian, Poisson, Beta, Exponential

Bayesian Inference:
- BayesianUpdater: conjugate prior-posterior updates
- NaiveBayesClassifier: online multi-class classification
- ProbabilityCalibrator: Platt scaling for score calibration
"""

from .mle import MaximumLikelihoodEstimator, MLEResult
from .bayesian import (
    BayesianUpdater,
    NaiveBayesClassifier,
    ProbabilityCalibrator,
    Prior,
    Posterior,
)

__all__ = [
    "MaximumLikelihoodEstimator",
    "MLEResult",
    "BayesianUpdater",
    "NaiveBayesClassifier",
    "ProbabilityCalibrator",
    "Prior",
    "Posterior",
]
