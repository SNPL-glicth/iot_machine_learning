"""Biologically realistic Leaky-Integrate-Fire neuron model.

Membrane potential differential equation:
    C_m * dV/dt = -(V - V_rest)/R_m + I_syn(t) + I_noise(t)
    Equivalent: dV/dt = -(V - V_rest)/τ_m + I_syn/C_m + I_noise

Includes:
- Absolute and relative refractory periods
- Spike-frequency adaptation
- Membrane noise
- Realistic biological parameters
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .membrane import compute_spike_energy


@dataclass
class NeuronParameters:
    """Biologically realistic neuron parameters (cortical pyramidal neuron).
    
    Attributes:
        V_rest: Resting membrane potential (mV)
        V_threshold: Spike threshold (mV)
        V_reset: Post-spike hyperpolarization (mV)
        V_spike: Spike peak amplitude (mV)
        tau_m: Membrane time constant (ms) = R_m * C_m
        tau_ref: Absolute refractory period (ms)
        tau_rel: Relative refractory period (ms)
        C_m: Membrane capacitance (pF)
        R_m: Membrane resistance (MΩ)
        noise_std: Membrane noise amplitude (mV)
        adaptation_tau: Spike-frequency adaptation time constant (ms)
        adaptation_increment: Adaptation current increment per spike (pA)
    """
    V_rest: float = -70.0
    V_threshold: float = -55.0
    V_reset: float = -80.0
    V_spike: float = 40.0
    tau_m: float = 20.0
    tau_ref: float = 2.0
    tau_rel: float = 5.0
    C_m: float = 281.0
    R_m: float = 70.0
    noise_std: float = 0.5
    adaptation_tau: float = 144.0
    adaptation_increment: float = 4.0


@dataclass
class SpikeEvent:
    """Spike event with energetics.
    
    Attributes:
        timestamp: Time of spike (ms)
        amplitude: Spike peak amplitude (mV)
        energy_cost: Energy consumed (J)
    """
    timestamp: float
    amplitude: float
    energy_cost: float


class LeakyIntegrateFireNeuron:
    """Biologically realistic LIF neuron with adaptation and noise.
    
    Args:
        neuron_id: Unique identifier
        params: Neuron parameters (uses biological defaults if None)
    """
    
    def __init__(
        self,
        neuron_id: str,
        params: Optional[NeuronParameters] = None,
    ) -> None:
        self.neuron_id = neuron_id
        self.params = params or NeuronParameters()
        
        # State variables
        self.V = self.params.V_rest
        self.I_adapt = 0.0
        self.t_last_spike = -float('inf')
        self.refractory_remaining = 0.0
        self.total_spikes = 0
        self.spike_times: List[float] = []
        self.energy_consumed = 0.0
    
    def integrate(
        self,
        I_syn: float,
        dt: float,
        current_time: float,
    ) -> Optional[SpikeEvent]:
        """Euler integration of membrane dynamics.
        
        Args:
            I_syn: Synaptic input current (pA)
            dt: Time step (ms)
            current_time: Current simulation time (ms)
            
        Returns:
            SpikeEvent if neuron fired, None otherwise
        """
        # 1. Check absolute refractory period
        if self.refractory_remaining > 0:
            self.refractory_remaining = max(0.0, self.refractory_remaining - dt)
            return None
        
        # 2. Apply relative refractory period (increased threshold)
        effective_threshold = self._get_effective_threshold(current_time)
        
        # 3. Euler integration of membrane potential
        # dV/dt = -(V - V_rest)/τ_m + I_total/C_m + noise
        I_total = I_syn - self.I_adapt
        noise = np.random.normal(0, self.params.noise_std)
        
        dV_dt = (
            -(self.V - self.params.V_rest) / self.params.tau_m +
            I_total / self.params.C_m +
            noise / dt
        )
        
        self.V += dV_dt * dt
        
        # 4. Decay adaptation current
        dI_adapt_dt = -self.I_adapt / self.params.adaptation_tau
        self.I_adapt += dI_adapt_dt * dt
        
        # 5. Spike detection
        if self.V >= effective_threshold:
            return self._fire_spike(current_time)
        
        return None
    
    def _fire_spike(self, current_time: float) -> SpikeEvent:
        """Handle spike generation and reset.
        
        Args:
            current_time: Time of spike (ms)
            
        Returns:
            SpikeEvent with spike details
        """
        # Record spike
        self.total_spikes += 1
        self.spike_times.append(current_time)
        self.t_last_spike = current_time
        
        # Reset membrane potential
        spike_amplitude = self.params.V_spike
        self.V = self.params.V_reset
        
        # Enter refractory period
        self.refractory_remaining = self.params.tau_ref
        
        # Increment adaptation current
        self.I_adapt += self.params.adaptation_increment
        
        # Compute energy cost
        energy = compute_spike_energy(1)
        self.energy_consumed += energy
        
        return SpikeEvent(
            timestamp=current_time,
            amplitude=spike_amplitude,
            energy_cost=energy,
        )
    
    def _get_effective_threshold(self, current_time: float) -> float:
        """Compute effective threshold (higher during relative refractory).
        
        Args:
            current_time: Current time (ms)
            
        Returns:
            Effective spike threshold (mV)
        """
        time_since_spike = current_time - self.t_last_spike
        
        # During relative refractory, threshold is elevated
        if time_since_spike < self.params.tau_rel:
            # Exponential recovery toward normal threshold
            elevation = 10.0 * np.exp(-time_since_spike / self.params.tau_rel)
            return self.params.V_threshold + elevation
        
        return self.params.V_threshold
    
    def reset(self) -> None:
        """Reset neuron to initial state."""
        self.V = self.params.V_rest
        self.I_adapt = 0.0
        self.t_last_spike = -float('inf')
        self.refractory_remaining = 0.0
        self.total_spikes = 0
        self.spike_times = []
        self.energy_consumed = 0.0
    
    @property
    def is_active(self) -> bool:
        """True if neuron has fired at least once."""
        return self.total_spikes > 0
    
    def get_firing_rate(self, duration_ms: float) -> float:
        """Compute average firing rate over duration.
        
        Args:
            duration_ms: Total simulation duration (ms)
            
        Returns:
            Firing rate in spikes/ms
        """
        if duration_ms <= 0:
            return 0.0
        return self.total_spikes / duration_ms
