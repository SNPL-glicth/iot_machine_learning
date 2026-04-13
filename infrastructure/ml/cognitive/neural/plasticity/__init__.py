"""Advanced neural plasticity mechanisms.

Biological learning mechanisms for neural networks:
- Metaplasticity: BCM sliding threshold
- Homeostatic regulation: Activity-dependent weight scaling

NOTE: Neuromodulation moved to _experimental/neural/ — not production ready
"""

from .metaplasticity import MetaplasticityController
from .homeostatic import HomeostaticRegulator

# NOTE: Neuromodulation moved to _experimental/ — not currently available
try:
    from iot_machine_learning.infrastructure.ml._experimental.neural.neuromodulation import NeuromodulationSignal
except ImportError:
    NeuromodulationSignal = None  # type: ignore[assignment,misc]

__all__ = [
    "MetaplasticityController",
    "HomeostaticRegulator",
    "NeuromodulationSignal",
]
