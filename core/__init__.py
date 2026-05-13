"""Core numerical constants and validation.

Centralized registry for numerical stability and statistical thresholds.
Organized into submodules: parameters, statistical, ensemble, drift, tuning.
"""

# Re-export from parameters submodule for backward compatibility
from core.parameters.numerical_constants import EPSILON, STAT_THRESHOLDS, CONFIDENCE
from core.parameters.parameter_validator import ParameterValidator

# Re-export all submodules for direct access
from core import parameters
from core import statistical
from core import ensemble
from core import drift
from core import tuning

__all__ = [
    "EPSILON",
    "STAT_THRESHOLDS",
    "CONFIDENCE",
    "ParameterValidator",
    "parameters",
    "statistical",
    "ensemble",
    "drift",
    "tuning",
]
