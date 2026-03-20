"""SNN network with Spike-Timing Dependent Plasticity (STDP).

STDP learning rule:
    ΔW = A+ * exp(-Δt/τ+) if pre before post (potentiation)
    ΔW = -A- * exp(-Δt/τ-) if post before pre (depression)

This implements Hebbian learning for spiking networks.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from .neuron import LeakyIntegrateFireNeuron, NeuronParameters
from .membrane import SynapticKernel
from ..types import SpikePattern


class STDPLearning:
    """Spike-Timing Dependent Plasticity learning rule.
    
    Args:
        A_plus: Potentiation amplitude (weight increase)
        A_minus: Depression amplitude (weight decrease)
        tau_plus: Potentiation time constant (ms)
        tau_minus: Depression time constant (ms)
        w_min: Minimum weight
        w_max: Maximum weight
    """
    
    def __init__(
        self,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
    ) -> None:
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max
    
    def update_weight(
        self,
        weight: float,
        pre_spike_times: List[float],
        post_spike_times: List[float],
    ) -> float:
        """Apply STDP rule for pre-post spike pairs.
        
        Args:
            weight: Current synaptic weight
            pre_spike_times: Presynaptic spike times
            post_spike_times: Postsynaptic spike times
            
        Returns:
            Updated weight
        """
        if not pre_spike_times or not post_spike_times:
            return weight
        
        delta_w = 0.0
        
        # For each post spike, find nearest pre spike
        for t_post in post_spike_times:
            for t_pre in pre_spike_times:
                dt = t_post - t_pre
                
                if dt > 0:
                    # Pre before post → potentiation
                    delta_w += self.A_plus * np.exp(-dt / self.tau_plus)
                elif dt < 0:
                    # Post before pre → depression
                    delta_w -= self.A_minus * np.exp(dt / self.tau_minus)
        
        # Update weight with bounds
        new_weight = weight + delta_w
        return np.clip(new_weight, self.w_min, self.w_max)


class SNNLayer:
    """SNN network layer with STDP online learning.
    
    Architecture: input(N) → hidden(H) → output(O)
    Fully connected with online STDP weight updates.
    
    Args:
        n_input: Number of input neurons
        n_hidden: Number of hidden neurons
        n_output: Number of output neurons
        neuron_config: Neuron parameters (uses defaults if None)
        dt: Simulation timestep (ms)
        enable_stdp: Enable online STDP learning
    """
    
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        neuron_config: Optional[NeuronParameters] = None,
        dt: float = 0.1,
        enable_stdp: bool = False,
    ) -> None:
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.dt = dt
        self.enable_stdp = enable_stdp
        self.config = neuron_config or NeuronParameters()
        
        # Create neurons (using biologically realistic LIF)
        self.input_neurons = [
            LeakyIntegrateFireNeuron(f"input_{i}", self.config)
            for i in range(n_input)
        ]
        self.hidden_neurons = [
            LeakyIntegrateFireNeuron(f"hidden_{i}", self.config)
            for i in range(n_hidden)
        ]
        self.output_neurons = [
            LeakyIntegrateFireNeuron(f"output_{i}", self.config)
            for i in range(n_output)
        ]
        
        # Initialize weights (Xavier initialization)
        rng = np.random.RandomState(42)
        limit_ih = np.sqrt(6.0 / (n_input + n_hidden))
        limit_ho = np.sqrt(6.0 / (n_hidden + n_output))
        
        self.w_input_hidden = rng.uniform(0.1, limit_ih, (n_input, n_hidden))
        self.w_hidden_output = rng.uniform(0.1, limit_ho, (n_hidden, n_output))
        
        # Synaptic kernel for current computation
        self.synapse = SynapticKernel()
        
        # STDP learner
        self.stdp = STDPLearning() if enable_stdp else None
    
    def forward(
        self,
        input_spike_trains: Dict[str, List[float]],
        duration_ms: float,
    ) -> Dict[str, SpikePattern]:
        """Run forward pass through network with STDP learning.
        
        Args:
            input_spike_trains: Dict of {input_name: [spike_times]}
            duration_ms: Simulation duration (ms)
            
        Returns:
            Dict of output spike patterns
        """
        # Reset all neurons
        for neuron in self.input_neurons + self.hidden_neurons + self.output_neurons:
            neuron.reset()
        
        # Map input spike trains to neuron indices
        input_spikes_by_neuron = self._map_input_spikes(input_spike_trains)
        
        # Simulate network dynamics
        n_steps = int(duration_ms / self.dt)
        
        for step in range(n_steps):
            current_time = step * self.dt
            
            # Step 1: Drive input neurons with spike trains
            for i, neuron in enumerate(self.input_neurons):
                spikes = input_spikes_by_neuron.get(i, [])
                I_syn = 10.0 if any(abs(t - current_time) < self.dt for t in spikes) else 0.0
                neuron.integrate(I_syn, self.dt, current_time)
            
            # Step 2: Propagate input → hidden
            for j, hidden_neuron in enumerate(self.hidden_neurons):
                I_syn = self._compute_synaptic_current(
                    self.input_neurons, self.w_input_hidden[:, j], current_time
                )
                hidden_neuron.integrate(I_syn, self.dt, current_time)
            
            # Step 3: Propagate hidden → output
            for k, output_neuron in enumerate(self.output_neurons):
                I_syn = self._compute_synaptic_current(
                    self.hidden_neurons, self.w_hidden_output[:, k], current_time
                )
                output_neuron.integrate(I_syn, self.dt, current_time)
        
        # Apply STDP learning after forward pass
        if self.enable_stdp and self.stdp:
            self._apply_stdp_learning()
        
        # Collect output patterns
        output_patterns = {}
        for neuron in self.output_neurons:
            pattern = SpikePattern(
                neuron_id=neuron.neuron_id,
                spike_times=neuron.spike_times.copy(),
                firing_rate=neuron.get_firing_rate(duration_ms),
                total_spikes=neuron.total_spikes,
                duration_ms=duration_ms,
            )
            output_patterns[neuron.neuron_id] = pattern
        
        return output_patterns
    
    def _compute_synaptic_current(
        self,
        presynaptic_neurons: List[LeakyIntegrateFireNeuron],
        weights: np.ndarray,
        current_time: float,
    ) -> float:
        """Compute synaptic current from presynaptic neurons."""
        I_total = 0.0
        
        for i, neuron in enumerate(presynaptic_neurons):
            if neuron.total_spikes > 0 and current_time - neuron.t_last_spike < self.dt:
                I_total += weights[i] * 50.0  # Synaptic strength
        
        return I_total
    
    def _apply_stdp_learning(self) -> None:
        """Apply STDP to all synaptic weights."""
        # Update input → hidden weights
        for i, input_neuron in enumerate(self.input_neurons):
            for j, hidden_neuron in enumerate(self.hidden_neurons):
                self.w_input_hidden[i, j] = self.stdp.update_weight(
                    self.w_input_hidden[i, j],
                    input_neuron.spike_times,
                    hidden_neuron.spike_times,
                )
        
        # Update hidden → output weights
        for j, hidden_neuron in enumerate(self.hidden_neurons):
            for k, output_neuron in enumerate(self.output_neurons):
                self.w_hidden_output[j, k] = self.stdp.update_weight(
                    self.w_hidden_output[j, k],
                    hidden_neuron.spike_times,
                    output_neuron.spike_times,
                )
    
    def _map_input_spikes(
        self,
        input_spike_trains: Dict[str, List[float]],
    ) -> Dict[int, List[float]]:
        """Map input spike trains to neuron indices."""
        result = {}
        for idx, (analyzer_name, spike_times) in enumerate(input_spike_trains.items()):
            if idx < self.n_input:
                result[idx] = spike_times
        return result
    
    def get_total_energy(self) -> float:
        """Compute total energy consumed by all neurons."""
        total = 0.0
        for neuron in self.input_neurons + self.hidden_neurons + self.output_neurons:
            total += neuron.energy_consumed
        return total
    
    def get_active_neuron_count(self) -> int:
        """Count neurons that fired at least once."""
        active = 0
        for neuron in self.input_neurons + self.hidden_neurons + self.output_neurons:
            if neuron.is_active:
                active += 1
        return active
    
    def get_silent_neuron_count(self) -> int:
        """Count neurons that never fired."""
        total = self.n_input + self.n_hidden + self.n_output
        return total - self.get_active_neuron_count()
