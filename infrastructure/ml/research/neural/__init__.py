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

from .hybrid_engine import HybridNeuralEngine
from .types import NeuralResult, NeuronState, SpikePattern

__all__ = [
    "HybridNeuralEngine",
    "NeuralResult",
    "NeuronState",
    "SpikePattern",
]
