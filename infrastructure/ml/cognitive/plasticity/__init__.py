"""Plasticity module stub — re-exports from bayesian_weight_tracker.

Backward compatibility: old plasticity imports still work.
"""

from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.base import (
    BayesianWeightTracker as PlasticityTracker,
)

__all__ = ["PlasticityTracker"]
