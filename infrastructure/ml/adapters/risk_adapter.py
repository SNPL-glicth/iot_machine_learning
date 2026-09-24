"""Stochastic Risk Engine Adapter for Rosa Roja MoE Juror.

Computes tolerance tube R_t, calibration breach penalty Omega_t,
parametric CVaR, and binary capitulation veto in pure shadow mode.
"""

from __future__ import annotations

from collections import deque
import logging
import math
from typing import Any, Dict, Optional, Sequence
import numpy as np

from domain.entities.rosa_roja.trajectory import Trajectory
from domain.ports.rosa_roja.expert_jury import ExpertJuryPort
from core.parameters.numerical_constants import EPSILON

logger = logging.getLogger(__name__)


class RiskEngineAdapter(ExpertJuryPort):
    """
    Expert juror implementing Stochastic Risk Tube & CVaR Capitulation Veto.
    
    Operates strictly in logarithmic return space (dimensionless return units, not dollars).
    
    Formula:
        R_t = σ(t) · √(Δt) · exp(clip(Ω_t, 0, Ω_max))
        breach_t = 1 if |r_t - r_{t-1}^expected| > R_{t-1} else 0
        Ω_t = EMA(breach_t, alpha=0.1)
        CVaR_t ≈ 2.70 · R_t  (Student-t nu=4 tail parametric approximation; replaces Gaussian 2.063)
        veto_riesgo = 1 if CVaR_t <= L_max else 0

    NOTE ON EMPIRICAL TAIL CALIBRATION:
        Raw sample excess kurtosis observed in production logs is ~1,734. However,
        this extreme figure is largely an artifact of the low-frequency event feed where
        ~36% of consecutive ticks have identical mid-prices (zero returns), causing
        zero-variance instability in rolling short windows (w=25). A dynamic rolling
        estimation of kurtosis/nu is therefore noisy and unviable. Instead, a static robust
        Student-t calibration with nu=4 is used (cvar_multiplier = 2.70), providing a
        protective 31% margin of safety over the Gaussian approximation (2.0627) without
        introducing online variance noise. Do NOT use raw sample kurtosis as an automated
        calibration input without prior filtering.
    """

    def __init__(
        self,
        name: str = "stochastic_risk",
        is_critical: bool = False,
        threshold: float = 0.5,
        weight: float = 0.0,
        window_size: int = 25,
        omega_alpha: float = 0.1,
        omega_max: float = 2.0,
        cvar_multiplier: float = 2.70,  # Student-t (nu=4) 95% CVaR multiplier (calibrated; replaces Gaussian 2.0627)
        l_max: float = 0.02,  # 2.0% max tolerable risk move by default
        default_sigma: float = 0.001,
    ):
        self.name, self.is_critical, self.threshold = name, is_critical, threshold
        self.weight, self.window_size, self.omega_alpha = weight, window_size, omega_alpha
        self.omega_max, self.cvar_multiplier = omega_max, cvar_multiplier
        self.l_max, self.default_sigma = l_max, default_sigma

        # Internal state (in return space)
        self._returns: deque[float] = deque(maxlen=window_size)
        self._omega: float = 0.0
        self._last_expected_return: Optional[float] = None
        self._last_r_t: Optional[float] = None
        self._last_verdict: Dict[str, Any] = {
            "R_t": 0.0, "sigma_t": self.default_sigma, "delta_t": 1.0, "omega_t": 0.0,
            "cvar_t": 0.0, "l_max": self.l_max, "veto_riesgo": 1, "breach_detected": False, "confidence": 0.95,
        }

    def record_observation(
        self,
        return_signal: Optional[float] = None,
        delta_time: float = 1.0,
        log_return: Optional[float] = None,
        expected_return: Optional[float] = None,
        *,
        current_price: Optional[float] = None,
        expected_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Record real market observation and compute risk metrics in logarithmic return units."""
        dt = max(EPSILON.DIVISION, float(delta_time))
        effective_signal = return_signal if return_signal is not None else current_price
        if effective_signal is None:
            effective_signal = log_return if log_return is not None else 0.0

        if effective_signal is not None and abs(effective_signal) > 1.0:
            logger.warning(
                "RiskEngineAdapter: input signal magnitude |%.4f| > 1.0. "
                "Engine operates in logarithmic return space (~0.0001 - 0.05), possible absolute price mistakenly passed.",
                effective_signal,
            )

        effective_expected = expected_return if expected_return is not None else expected_price
        effective_log_return = log_return if log_return is not None else effective_signal
        if effective_log_return is not None and math.isfinite(effective_log_return):
            self._returns.append(float(effective_log_return))

        # Check calibration breach from previous step in return space
        breach_detected = False
        if self._last_expected_return is not None and self._last_r_t is not None and self._last_r_t > 0:
            if abs(effective_signal - self._last_expected_return) > self._last_r_t:
                breach_detected = True

        breach_val = 1.0 if breach_detected else 0.0
        self._omega = self.omega_alpha * breach_val + (1.0 - self.omega_alpha) * self._omega
        omega_clamped = min(self.omega_max, max(0.0, self._omega))

        # Dynamic realized volatility σ(t)
        sigma_t = float(np.std(self._returns)) if len(self._returns) >= 5 else self.default_sigma
        if sigma_t < EPSILON.COMPARISON:
            sigma_t = self.default_sigma

        # Tolerance tube R_t & parametric CVaR
        r_t = sigma_t * math.sqrt(dt) * math.exp(omega_clamped)
        cvar_t = self.cvar_multiplier * r_t
        veto_riesgo = 1 if cvar_t <= self.l_max else 0

        risk_ratio = cvar_t / (self.l_max + EPSILON.DIVISION)
        confidence = max(0.20, min(0.95, 1.0 - 0.5 * risk_ratio))

        self._last_r_t = float(r_t)
        self._last_expected_return = float(effective_expected if effective_expected is not None else effective_signal)

        self._last_verdict = {
            "R_t": float(r_t),
            "sigma_t": float(sigma_t),
            "delta_t": float(dt),
            "omega_t": float(omega_clamped),
            "cvar_t": float(cvar_t),
            "l_max": float(self.l_max),
            "veto_riesgo": int(veto_riesgo),
            "breach_detected": breach_detected,
            "confidence": float(confidence),
        }
        return self._last_verdict

    def evaluate_trajectory(self, trajectory: Trajectory) -> float:
        """ExpertJuryPort contract: evaluate trajectory confidence."""
        # If trajectory has delta_states, use feature 0 (log_return) to evaluate dispersion
        try:
            if trajectory.movements:
                traj_returns = [float(m.delta_state[0]) for m in trajectory.movements if len(m.delta_state) > 0]
                if traj_returns:
                    traj_std = float(np.std(traj_returns))
                    dt = sum(m.delta_time for m in trajectory.movements)
                    r_traj = traj_std * math.sqrt(max(1.0, dt)) * math.exp(min(self.omega_max, self._omega))
                    ratio = (self.cvar_multiplier * r_traj) / (self.l_max + EPSILON.DIVISION)
                    return float(max(0.20, min(0.95, 1.0 - 0.5 * ratio)))
        except Exception: pass
        return float(self._last_verdict.get("confidence", 0.5))

    def update_learning(self, actual: float, predicted: float) -> None:
        """Update calibration state with realized outcome."""
        if self._last_r_t is not None and self._last_r_t > 0:
            breach = 1.0 if abs(actual - predicted) > self._last_r_t else 0.0
            self._omega = self.omega_alpha * breach + (1.0 - self.omega_alpha) * self._omega

    def get_shadow_metrics(self) -> Dict[str, Any]:
        """Return latest shadow evaluation metrics."""
        return dict(self._last_verdict)
