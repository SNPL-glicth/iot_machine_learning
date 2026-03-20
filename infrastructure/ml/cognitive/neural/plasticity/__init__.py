"""Advanced neural plasticity mechanisms.

Biological learning mechanisms for neural networks:
- Metaplasticity: BCM sliding threshold
- Neuromodulation: Dopamine-like reward signal
- Homeostatic regulation: Activity-dependent weight scaling
"""

from .metaplasticity import MetaplasticityController
from .neuromodulation import NeuromodulationSignal
from .homeostatic import HomeostaticRegulator

__all__ = [
    "MetaplasticityController",
    "NeuromodulationSignal",
    "HomeostaticRegulator",
]
