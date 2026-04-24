"""Plasticity base stub — re-exports from bayesian_weight_tracker.

DEPRECATED: Use bayesian_weight_tracker.base directly.
"""

from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.base import (
    BayesianWeightTracker as PlasticityTracker,
)

__all__ = ["PlasticityTracker"]
