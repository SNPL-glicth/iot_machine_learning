"""Data types for neural engine subsystem.

Pure value objects for neural analysis results and internal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..universal.analysis.types import InputType


@dataclass(frozen=True)
class SpikePattern:
    """Spike pattern from a neuron or layer.
    
    Attributes:
        neuron_id: Identifier of the neuron
        spike_times: List of spike times in milliseconds
        firing_rate: Average firing rate (spikes/ms)
        total_spikes: Total number of spikes
        duration_ms: Duration of the pattern
    """
    neuron_id: str
    spike_times: List[float]
    firing_rate: float
    total_spikes: int
    duration_ms: float
    
    def to_dict(self) -> dict:
        return {
            "neuron_id": self.neuron_id,
            "spike_times": [round(t, 3) for t in self.spike_times],
            "firing_rate": round(self.firing_rate, 6),
            "total_spikes": self.total_spikes,
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass(frozen=True)
class NeuronState:
    """Internal state of a single neuron.
    
    Attributes:
        membrane_potential: Current voltage (like electron charge)
        last_spike_time: Time of last spike (ms)
        refractory_remaining: Remaining refractory period (ms)
        total_spikes: Cumulative spike count
        energy_consumed: Energy consumed (spike_count × spike_energy_cost)
    """
    membrane_potential: float
    last_spike_time: float
    refractory_remaining: float
    total_spikes: int
    energy_consumed: float
    
    def to_dict(self) -> dict:
        return {
            "membrane_potential": round(self.membrane_potential, 6),
            "last_spike_time": round(self.last_spike_time, 3),
            "refractory_remaining": round(self.refractory_remaining, 3),
            "total_spikes": self.total_spikes,
            "energy_consumed": round(self.energy_consumed, 9),
        }


@dataclass(frozen=True)
class NeuralResult:
    """Output of HybridNeuralEngine.
    
    Combines SNN and classical feedforward outputs with neuromorphic metrics.
    
    Attributes:
        severity: Classified severity level
        confidence: Overall confidence [0, 1]
        spike_patterns: Per-neuron spike patterns
        firing_rates: Per-neuron average firing rates
        energy_consumed: Total energy (sum of spike counts × spike_energy_cost)
        active_neurons: Number of neurons that fired
        silent_neurons: Number of neurons below threshold
        domain: Detected/assigned domain
        input_type: Input type (TEXT, NUMERIC, etc.)
        snn_output: Raw SNN layer output
        classical_output: Raw classical layer output
        hybrid_weight_snn: Weight given to SNN in hybrid fusion
        hybrid_weight_classical: Weight given to classical in hybrid fusion
        monte_carlo: Optional Monte Carlo uncertainty result
    """
    severity: str
    confidence: float
    spike_patterns: Dict[str, SpikePattern]
    firing_rates: Dict[str, float]
    energy_consumed: float
    active_neurons: int
    silent_neurons: int
    domain: str
    input_type: InputType
    snn_output: float = 0.0
    classical_output: float = 0.0
    hybrid_weight_snn: float = 0.5
    hybrid_weight_classical: float = 0.5
    monte_carlo: Optional[object] = None
    
    def to_dict(self) -> dict:
        """Serialize for API responses."""
        result = {
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "spike_patterns": {
                k: v.to_dict() for k, v in self.spike_patterns.items()
            },
            "firing_rates": {
                k: round(v, 6) for k, v in self.firing_rates.items()
            },
            "energy_consumed": round(self.energy_consumed, 9),
            "active_neurons": self.active_neurons,
            "silent_neurons": self.silent_neurons,
            "domain": self.domain,
            "input_type": self.input_type.value,
            "snn_output": round(self.snn_output, 6),
            "classical_output": round(self.classical_output, 6),
            "hybrid_weight_snn": round(self.hybrid_weight_snn, 4),
            "hybrid_weight_classical": round(self.hybrid_weight_classical, 4),
        }
        
        if self.monte_carlo is not None:
            result["monte_carlo"] = self.monte_carlo.to_dict()
        
        return result
    
    @property
    def energy_efficiency(self) -> float:
        """Energy efficiency metric (0-1, higher is better).
        
        Returns fraction of neurons that remained silent (didn't consume energy).
        """
        total = self.active_neurons + self.silent_neurons
        if total == 0:
            return 1.0
        return self.silent_neurons / total
