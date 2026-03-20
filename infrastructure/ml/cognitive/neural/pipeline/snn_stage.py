"""SNN stage — spiking neural network forward pass."""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..snn import SNNLayer
from ..types import SpikePattern


class SNNStage:
    """Processes spike trains through spiking neural network.
    
    Args:
        snn_layer: SNNLayer instance
        duration_ms: Simulation duration in milliseconds
    """
    
    def __init__(
        self,
        snn_layer: SNNLayer,
        duration_ms: float = 100.0,
    ) -> None:
        self.snn_layer = snn_layer
        self.duration_ms = duration_ms
    
    def process(
        self,
        spike_trains: Dict[str, List[float]],
    ) -> Tuple[Dict[str, SpikePattern], Dict[str, float]]:
        """Run SNN forward pass.
        
        Args:
            spike_trains: Dict of {analyzer_name: [spike_times]}
            
        Returns:
            Tuple of (output_patterns, firing_rates)
        """
        # Forward pass through network
        output_patterns = self.snn_layer.forward(
            spike_trains,
            self.duration_ms,
        )
        
        # Extract firing rates
        firing_rates = {
            neuron_id: pattern.firing_rate
            for neuron_id, pattern in output_patterns.items()
        }
        
        return output_patterns, firing_rates
    
    def get_energy_metrics(self) -> Tuple[float, int, int]:
        """Get energy consumption metrics.
        
        Returns:
            Tuple of (energy_consumed, active_neurons, silent_neurons)
        """
        energy = self.snn_layer.get_total_energy()
        active = self.snn_layer.get_active_neuron_count()
        silent = self.snn_layer.get_silent_neuron_count()
        
        return energy, active, silent
