"""Neural subsystem — Hybrid Spiking + Classical Neural Network.

Combines Spiking Neural Network (SNN) with classical feedforward layers
for neuromorphic analysis with online learning.

Components:
    - HybridNeuralEngine: Main orchestrator
    - NeuralResult: Analysis output with energy metrics
    - SNN: Leaky-Integrate-Fire neurons with spike encoding
    - Classical: Dense feedforward layer
    - Competition: Arbiter for engine selection
"""

from .types import NeuralResult, NeuronState, SpikePattern

# NOTE: hybrid_engine was moved to cognitive/neural/ and is not available here.
# The __init__.py intentionally does NOT re-export it to avoid import errors.

__all__ = [
    # "HybridNeuralEngine",  # module moved to cognitive/neural/
    "NeuralResult",
    "NeuronState",
    "SpikePattern",
]
