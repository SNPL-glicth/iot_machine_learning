"""Synthetic ground-truth environments for the Rosa Roja sandbox (RR-4).

Environments are deterministic cyclic transition systems over integer delta
vectors. Integer deltas guarantee that Module 2's state quantization
(round to 2 decimals) maps identical dynamics to identical graph keys, so the
ground truth is exactly recoverable from the engine's point of view.

Adversarial extensions (RR-4 expansion):
- JitteryPatternEnv: tempo jitter via Gaussian Δt
- NoisyStateEnv: Gaussian state noise on deltas
- FlappingRegimeEnv: rapid regime switching
"""

from __future__ import annotations

import numpy as np


class CyclicPatternEnv:
    """Deterministic system cycling through a known sequence of state deltas."""

    def __init__(self, cycle: list[list[float]], dim: int = 3):
        if len(cycle) < 2:
            raise ValueError("cycle needs at least 2 transitions")
        self.cycle = [np.asarray(c, dtype=float) for c in cycle]
        for c in self.cycle:
            if c.shape != (dim,):
                raise ValueError(f"each delta must have shape ({dim},)")
        self.dim = dim
        self.t = 0

    def reset(self) -> None:
        self.t = 0

    @property
    def period(self) -> int:
        return len(self.cycle)

    def peek(self) -> np.ndarray:
        """True next delta without advancing time."""
        return self.cycle[self.t % self.period].copy()

    def step(self) -> np.ndarray:
        delta = self.peek()
        self.t += 1
        return delta


class SwitchingPatternEnv:
    """
    Pattern A until `switch_step`, pattern B afterwards.

    Ground truth exposes `switched_at` and `current_pattern` so the benchmark
    can measure detection latency and recovery.
    """

    def __init__(
        self,
        pattern_a: list[list[float]],
        pattern_b: list[list[float]],
        switch_step: int,
        dim: int = 3,
    ):
        self._env_a = CyclicPatternEnv(pattern_a, dim)
        self._env_b = CyclicPatternEnv(pattern_b, dim)
        self.switch_step = switch_step
        self.dim = dim
        self.t = 0

    def reset(self) -> None:
        self.t = 0
        self._env_a.reset()
        self._env_b.reset()

    @property
    def switched_at(self) -> int | None:
        return self.switch_step if self.t >= self.switch_step else None

    @property
    def current_pattern(self) -> str:
        return "B" if self.t >= self.switch_step else "A"

    def peek(self) -> np.ndarray:
        return self._env_b.peek() if self.t >= self.switch_step else self._env_a.peek()

    def step(self) -> np.ndarray:
        delta = self.peek()
        self.t += 1
        if self.current_pattern == "A":
            self._env_a.t = self.t
        else:
            self._env_b.t = self.t - self.switch_step
        return delta


class JitteryPatternEnv:
    """
    Deterministic directions with stochastic time intervals.
    
    Δt ~ N(μ=1.0, σ) where σ is configurable. Tests tempo sensitivity.
    """
    
    def __init__(
        self, 
        cycle: list[list[float]], 
        dim: int = 3, 
        dt_mean: float = 1.0, 
        dt_std: float = 0.5,
        rng: np.random.Generator = None
    ):
        self._base = CyclicPatternEnv(cycle, dim)
        self.dt_mean = dt_mean
        self.dt_std = dt_std
        self._rng = rng or np.random.default_rng()
        
    def reset(self) -> None:
        self._base.reset()
        
    def peek(self) -> np.ndarray:
        """True next delta (directional, no time noise)."""
        return self._base.peek()
        
    def peek_dt(self) -> float:
        """True next time interval (includes jitter)."""
        dt = self._rng.normal(self.dt_mean, self.dt_std)
        return max(dt, 1e-3)
        
    def step(self) -> tuple[np.ndarray, float]:
        """Returns (delta, dt) with jittered time interval."""
        delta = self._base.peek()
        dt = self.peek_dt()
        self._base.t += 1
        return delta, dt


class NoisyStateEnv:
    """
    Cyclic pattern with Gaussian noise injected into state deltas.
    
    S_{t+1} = S_t + ΔS + ε, ε ~ N(0, σ² I)
    Tests Mahalanobis covariance adaptation without false resets.
    """
    
    def __init__(
        self, 
        cycle: list[list[float]], 
        noise_std: float = 0.1,
        dim: int = 3,
        rng: np.random.Generator = None
    ):
        self._base = CyclicPatternEnv(cycle, dim)
        self.noise_std = noise_std
        self._rng = rng or np.random.default_rng()
        
    def reset(self) -> None:
        self._base.reset()
        
    def peek(self) -> np.ndarray:
        return self._base.peek()
        
    def step(self) -> np.ndarray:
        delta = self._base.peek()
        noise = self._rng.normal(0.0, self.noise_std, self._base.dim)
        noisy_delta = delta + noise
        self._base.t += 1
        return noisy_delta


class FlappingRegimeEnv:
    """
    Rapid regime flapping: switches regimes every few steps.
    
    Faster than covariance adaptation window. Tests safe degradation to HOLD
    and exploration boost without infinite reset loops.
    """
    
    def __init__(
        self,
        pattern_a: list[list[float]],
        pattern_b: list[list[float]],
        flip_every: int = 5,
        dim: int = 3,
        rng: np.random.Generator = None
    ):
        self._env_a = CyclicPatternEnv(pattern_a, dim)
        self._env_b = CyclicPatternEnv(pattern_b, dim)
        self.flip_every = flip_every
        self.dim = dim
        self._rng = rng or np.random.default_rng()
        self.t = 0
        self._current_is_a = True
        
    def reset(self) -> None:
        self.t = 0
        self._current_is_a = True
        self._env_a.reset()
        self._env_b.reset()
        
    @property
    def current_pattern(self) -> str:
        return "A" if self._current_is_a else "B"
        
    def peek(self) -> np.ndarray:
        if self._current_is_a:
            return self._env_a.peek()
        return self._env_b.peek()
        
    def step(self) -> np.ndarray:
        delta = self.peek()
        self.t += 1
        # Update internal counters
        if self._current_is_a:
            self._env_a.t += 1
        else:
            self._env_b.t += 1
        # Flip regime
        if self.t % self.flip_every == 0:
            self._current_is_a = not self._current_is_a
        return delta


# Distinct-delta cycles: every state key is unique within a pattern so walks
# can build trajectories of length >= 4 before cycle detection fires.
PATTERN_A = [
    [2.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 3.0, 0.0],
]

# Pattern B lives mostly on the z-axis (zero variance under A's covariance),
# so post-switch z-axis deltas are extreme Mahalanobis outliers.
PATTERN_B = [
    [0.0, 0.0, 2.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 1.0],
    [3.0, 0.0, 0.0],
]
