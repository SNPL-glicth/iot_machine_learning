"""Fractal Chronometric Engine Adapter for Rosa Roja MoE Juror.

Computes rhythmic synchronization Lambda(t) between price velocity (dS/dt)
and microstructure order-flow reference rhythm (dR/dt) in pure shadow mode.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from domain.entities.rosa_roja.trajectory import Trajectory
from domain.ports.rosa_roja.expert_jury import ExpertJuryPort
from infrastructure.ml.engines.statistical.smoothing import holt_stable
from core.parameters.numerical_constants import EPSILON


class TemporalEngineAdapter(ExpertJuryPort):
    """
    Expert juror implementing Fractal Chronometric Rhythm Synchronization Λ(t).
    
    Formula:
        ∂S/∂t: Smoothed price velocity (derived from Holt-EMA trend or Kalman).
        ∂R/∂t: Reference rhythm velocity from order-flow microstructure periodicity.
        Λ(t) = exp(-|((∂S/∂t) / (∂R/∂t + ε)) - 1|)
    """

    def __init__(
        self,
        name: str = "fractal_chronometric",
        is_critical: bool = False,
        threshold: float = 0.5,
        weight: float = 0.0,
        window_size: int = 30,
        rhythm_ema_alpha: float = 0.2,
        epsilon: float = 1e-6,
    ):
        self.name = name
        self.is_critical = is_critical
        self.threshold = threshold
        self.weight = weight
        self.window_size = window_size
        self.rhythm_ema_alpha = rhythm_ema_alpha
        self.epsilon = epsilon

        # State buffers
        self._imbalances: deque[float] = deque(maxlen=window_size)
        self._price_history: deque[float] = deque(maxlen=window_size)
        self._smoothed_dR_dt: float = 0.01
        self._last_verdict: Dict[str, Any] = {
            "lambda_crono": 0.5,
            "dS_dt": 0.01,
            "dR_dt": 0.01,
            "ratio": 1.0,
            "dominant_period": 4.0,
            "confidence": 0.5,
        }

    def _estimate_reference_rhythm(self) -> tuple[float, float]:
        """
        Option A: Microstructure periodicity derived from zero-crossing rate
        of centered book imbalance / order flow.
        
        Returns:
            tuple: (tau_ref: dominant period in ticks, dR_dt_raw: reference velocity)
        """
        n = len(self._imbalances)
        if n < 6:
            return 4.0, 0.01

        arr = np.array(self._imbalances, dtype=np.float64)
        centered = arr - np.mean(arr)
        
        # Count zero crossings to estimate fundamental cycle period
        signs = np.sign(centered)
        signs[signs == 0] = 1.0
        crossings = np.sum(signs[1:] != signs[:-1])
        
        if crossings > 0:
            # Average half-cycle length is (n - 1) / crossings; full period is double
            tau_ref = max(2.0, (2.0 * (n - 1)) / float(crossings))
        else:
            tau_ref = float(n)

        # Reference amplitude from absolute imbalance
        amplitude = float(np.mean(np.abs(arr))) + self.epsilon
        dR_dt_raw = amplitude / tau_ref
        return tau_ref, dR_dt_raw

    def record_observation(
        self,
        current_price: float,
        book_imbalance: Optional[float] = None,
        price_velocity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Record observation and compute Λ(t) synchrony."""
        if math.isfinite(current_price):
            self._price_history.append(float(current_price))

        if book_imbalance is not None and math.isfinite(book_imbalance):
            self._imbalances.append(float(book_imbalance))

        # 1. Compute reference rhythm velocity ∂R/∂t
        tau_ref, dR_dt_raw = self._estimate_reference_rhythm()
        self._smoothed_dR_dt = (
            self.rhythm_ema_alpha * dR_dt_raw + (1.0 - self.rhythm_ema_alpha) * self._smoothed_dR_dt
        )
        dR_dt = max(self.epsilon, self._smoothed_dR_dt)

        # 2. Compute price velocity ∂S/∂t using Holt-EMA trend on price history
        if price_velocity is not None and math.isfinite(price_velocity):
            dS_dt = abs(float(price_velocity))
        elif len(self._price_history) >= 4:
            _, trend = holt_stable(list(self._price_history), alpha=0.3, beta=0.1)
            # Scale trend relative to price to obtain dimensionless fractional rate
            ref_price = self._price_history[-1] if self._price_history[-1] > 0 else 1.0
            dS_dt = abs(trend / ref_price)
        else:
            dS_dt = dR_dt  # Default to in-sync when insufficient history

        # 3. Compute Synchrony Index Λ(t)
        # Λ(t) = exp(-|((∂S/∂t) / (∂R/∂t + ε)) - 1|)
        ratio = dS_dt / (dR_dt + self.epsilon)
        discrepancy = abs(ratio - 1.0)
        # Cap discrepancy exponent at 10 to avoid numerical underflow
        lambda_crono = float(math.exp(-min(10.0, discrepancy)))
        lambda_crono = max(0.001, min(1.0, lambda_crono))

        self._last_verdict = {
            "lambda_crono": float(lambda_crono),
            "dS_dt": float(dS_dt),
            "dR_dt": float(dR_dt),
            "ratio": float(ratio),
            "dominant_period": float(tau_ref),
            "confidence": float(lambda_crono),
        }
        return self._last_verdict

    def evaluate_trajectory(self, trajectory: Trajectory) -> float:
        """ExpertJuryPort contract: evaluate trajectory rhythm synchrony."""
        try:
            if trajectory.movements:
                # Average trajectory velocity
                velocities = trajectory.velocities
                if len(velocities) > 0:
                    traj_vel = float(np.mean(velocities))
                    dR_dt = max(self.epsilon, self._smoothed_dR_dt)
                    ratio = traj_vel / dR_dt
                    sync = math.exp(-min(10.0, abs(ratio - 1.0)))
                    return float(max(0.001, min(1.0, sync)))
        except Exception:
            pass
        return float(self._last_verdict.get("confidence", 0.5))

    def update_learning(self, actual: float, predicted: float) -> None:
        """Update historical tracking."""
        pass

    def get_shadow_metrics(self) -> Dict[str, Any]:
        """Return latest shadow evaluation metrics."""
        return dict(self._last_verdict)
