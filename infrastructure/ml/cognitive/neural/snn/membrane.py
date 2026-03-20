"""Membrane dynamics and synaptic kernels for biologically realistic SNN.

Synaptic current with exponential decay:
    I_syn(t) = w * I_0 * exp(-(t - t_spike)/τ_syn)

Membrane integration methods:
- Euler: Fast, less accurate
- Runge-Kutta 4: Slower, more accurate for large dt
"""

from __future__ import annotations

import numpy as np
from typing import List


# Physical constants
SPIKE_ENERGY_COST = 1e-12  # 1 picojoule per spike


class SynapticKernel:
    """Exponential synaptic current kernel.
    
    Models post-synaptic current (PSC) as exponential decay after spike.
    
    Args:
        tau_syn_exc: Excitatory synapse decay constant (ms)
        tau_syn_inh: Inhibitory synapse decay constant (ms)
        I_0: Base synaptic current amplitude (pA)
    """
    
    def __init__(
        self,
        tau_syn_exc: float = 5.0,
        tau_syn_inh: float = 10.0,
        I_0: float = 1.0,
    ) -> None:
        self.tau_syn_exc = tau_syn_exc
        self.tau_syn_inh = tau_syn_inh
        self.I_0 = I_0
    
    def compute_current(
        self,
        spike_times: List[float],
        weights: List[float],
        current_time: float,
        synapse_type: str = "excitatory",
    ) -> float:
        """Compute total synaptic current from incoming spikes.
        
        Args:
            spike_times: Times of incoming spikes (ms)
            weights: Synaptic weights for each spike
            current_time: Current time (ms)
            synapse_type: "excitatory" or "inhibitory"
            
        Returns:
            Total synaptic current (pA)
        """
        if not spike_times:
            return 0.0
        
        # Select decay constant
        tau_syn = (
            self.tau_syn_exc if synapse_type == "excitatory"
            else self.tau_syn_inh
        )
        
        # Sum exponential kernels for all spikes
        I_total = 0.0
        
        for t_spike, weight in zip(spike_times, weights):
            delta_t = current_time - t_spike
            
            # Only count spikes that have occurred
            if delta_t >= 0:
                kernel = np.exp(-delta_t / tau_syn)
                I_total += weight * self.I_0 * kernel
        
        return I_total


class MembraneDynamics:
    """Membrane potential integration methods.
    
    Provides Euler and Runge-Kutta 4 integration for:
        dV/dt = -(V - V_rest)/τ_m + I_syn/C_m + I_noise
    
    Args:
        use_rk4: Use RK4 integration when dt > threshold_dt
        threshold_dt: Time step threshold for RK4 (ms)
    """
    
    def __init__(
        self,
        use_rk4: bool = False,
        threshold_dt: float = 0.5,
    ) -> None:
        self.use_rk4 = use_rk4
        self.threshold_dt = threshold_dt
    
    def euler_step(
        self,
        V: float,
        I_syn: float,
        I_adapt: float,
        V_rest: float,
        tau_m: float,
        C_m: float,
        noise_std: float,
        dt: float,
    ) -> float:
        """Euler integration of membrane potential.
        
        Args:
            V: Current membrane potential (mV)
            I_syn: Synaptic input current (pA)
            I_adapt: Adaptation current (pA)
            V_rest: Resting potential (mV)
            tau_m: Membrane time constant (ms)
            C_m: Membrane capacitance (pF)
            noise_std: Noise amplitude (mV)
            dt: Time step (ms)
            
        Returns:
            New membrane potential (mV)
        """
        I_total = I_syn - I_adapt
        noise = np.random.normal(0, noise_std)
        
        dV_dt = (
            -(V - V_rest) / tau_m +
            I_total / C_m +
            noise / dt
        )
        
        return V + dV_dt * dt
    
    def runge_kutta_step(
        self,
        V: float,
        I_syn: float,
        I_adapt: float,
        V_rest: float,
        tau_m: float,
        C_m: float,
        dt: float,
    ) -> float:
        """RK4 integration of membrane potential (no noise).
        
        Higher precision for larger time steps.
        
        Args:
            V: Current membrane potential (mV)
            I_syn: Synaptic input current (pA)
            I_adapt: Adaptation current (pA)
            V_rest: Resting potential (mV)
            tau_m: Membrane time constant (ms)
            C_m: Membrane capacitance (pF)
            dt: Time step (ms)
            
        Returns:
            New membrane potential (mV)
        """
        I_total = I_syn - I_adapt
        
        # Define dV/dt function
        def f(v):
            return -(v - V_rest) / tau_m + I_total / C_m
        
        # RK4 steps
        k1 = f(V)
        k2 = f(V + 0.5 * dt * k1)
        k3 = f(V + 0.5 * dt * k2)
        k4 = f(V + dt * k3)
        
        return V + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_membrane(
    v_current: float,
    i_input: float,
    tau: float,
    dt: float,
    v_rest: float = 0.0,
) -> float:
    """Legacy interface — simple leaky integration.
    
    Args:
        v_current: Current membrane potential
        i_input: Input current
        tau: Membrane time constant
        dt: Time step
        v_rest: Resting potential
        
    Returns:
        New membrane potential
    """
    decay = -(v_current - v_rest) / tau
    dv = dt * (decay + i_input)
    return v_current + dv


def apply_refractory_decay(refractory_remaining: float, dt: float) -> float:
    """Decay refractory period."""
    return max(0.0, refractory_remaining - dt)


def compute_spike_energy(spike_count: int) -> float:
    """Compute energy consumed by spikes."""
    return spike_count * SPIKE_ENERGY_COST


def reset_membrane(v_reset: float = 0.0) -> float:
    """Reset membrane potential."""
    return v_reset
