"""Spiking Neural Network (SNN) layer — biologically realistic.

Implements full Leaky-Integrate-Fire (LIF) neuron model with:
- Realistic membrane dynamics and parameters
- Spike-frequency adaptation
- STDP online learning
- Rate + temporal coding

Components:
    - LeakyIntegrateFireNeuron: Biologically realistic LIF neuron
    - NeuronParameters: Cortical pyramidal neuron parameters
    - SpikeEncoder: Hybrid rate + temporal coding
    - SpikeDecoder: Spike pattern decoder
    - SNNLayer: Network with STDP learning
    - SynapticKernel: Exponential synaptic currents
    - MembraneDynamics: Euler and RK4 integration
"""

from .neuron import LeakyIntegrateFireNeuron, NeuronParameters, SpikeEvent
from .spike_encoder import SpikeEncoder
from .spike_decoder import SpikeDecoder
from .network import SNNLayer, STDPLearning
from .membrane import SynapticKernel, MembraneDynamics

__all__ = [
    "LeakyIntegrateFireNeuron",
    "NeuronParameters",
    "SpikeEvent",
    "SpikeEncoder",
    "SpikeDecoder",
    "SNNLayer",
    "STDPLearning",
    "SynapticKernel",
    "MembraneDynamics",
]
