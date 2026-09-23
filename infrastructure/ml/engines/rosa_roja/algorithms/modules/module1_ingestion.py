"""Module 1: Ingestion & Anti-Contamination Filter (Mahalanobis Distance).

Optimized with true O(1) incremental Welford covariance + diagonal inverse updates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
from typing import Any, Dict, List, Optional
import numpy as np
from numpy.linalg import inv

from iot_machine_learning.core.parameters.numerical_constants import DIV_BY_ZERO_EPSILON
from domain.entities.rosa_roja.movement import Movement, RhythmSignature
from domain.entities.rosa_roja.state_persistence import (
    STATE_SCHEMA_VERSION,
    movement_to_raw,
    movements_from_raw,
    pack_array,
    unpack_array,
)

logger = logging.getLogger(__name__)


DEFAULT_NOISE_THRESHOLD: float = 3.0
DEFAULT_HISTORY_WINDOW: int = 100
DEFAULT_MIN_SAMPLES_FOR_COV: int = 20
DEFAULT_RECOMPUTE_INTERVAL: int = 100
DEFAULT_REGULARIZATION_EPSILON: float = 1e-6
DEFAULT_MAX_CONDITION_NUMBER: float = 1e12
DEFAULT_PINV_RCOND: float = 1e-10


@dataclass
class MahalanobisFilter:
    """
    Mahalanobis Anti-Contamination Filter with O(1) incremental updates.
    
    Uses Welford's online algorithm for mean/covariance and
    diagonal inverse updates with periodic exact recomputation.
    
    Equation: D_t = D_{t-1} ∪ {(U_t, Y_t) · I(d_Mahalanobis(M_t) ≤ τ_noise)}
    """
    
    noise_threshold: float = DEFAULT_NOISE_THRESHOLD        # τ_noise (χ² quantile)
    history_window: int = DEFAULT_HISTORY_WINDOW           # Covariance estimation window
    min_samples_for_cov: int = DEFAULT_MIN_SAMPLES_FOR_COV       # Minimum samples before filtering active
    recompute_interval: int = DEFAULT_RECOMPUTE_INTERVAL       # Exact recompute interval for stability
    regularization_epsilon: float = DEFAULT_REGULARIZATION_EPSILON # Covariance regularization epsilon
    max_condition_number: float = DEFAULT_MAX_CONDITION_NUMBER  # Threshold for singular matrix condition
    pinv_rcond: float = DEFAULT_PINV_RCOND          # Pinv rcond for singular inversion
    division_epsilon: float = DIV_BY_ZERO_EPSILON # Epsilon to prevent division by zero
    
    def __post_init__(self):
        self._history: deque[Movement] = deque(maxlen=self.history_window)
        self._cov_inv: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._state_dim: Optional[int] = None
        # Welford's algorithm state
        self._n: int = 0
        self._M2: Optional[np.ndarray] = None  # Sum of squared differences
        
    def process_raw_step(self, delta_state: np.ndarray, delta_time: float) -> tuple[Movement, bool]:
        """
        Process a raw state transition and return (Movement, is_outlier).
        
        Returns:
            tuple: (Movement object, True if outlier/regime_change_alert else False)
        """
        timestamp = self._current_timestamp()
        
        # First sample initializes dimension
        if self._state_dim is None:
            self._state_dim = len(delta_state)
            # Initialize Welford state
            self._mean = delta_state.copy().astype(np.float64)
            self._M2 = np.zeros((self._state_dim, self._state_dim), dtype=np.float64)
            self._n = 0
            self._cov_inv = np.eye(self._state_dim, dtype=np.float64)
            self._reg_matrix = self.regularization_epsilon * np.eye(self._state_dim, dtype=np.float64)
        
        # Compute Mahalanobis distance if we have enough history
        mahal_dist = self._compute_mahalanobis(delta_state)
        
        # Create movement with current Mahalanobis distance
        prev_movement = self._history[-1] if self._history else None
        movement = Movement.from_raw(
            delta_state=delta_state,
            delta_time=delta_time,
            timestamp=timestamp,
            mahalanobis_dist=mahal_dist,
            prev_movement=prev_movement,
        )
        
        # Check if outlier - ensure Python bool return
        is_outlier = mahal_dist > self.noise_threshold
        
        if not is_outlier:
            # Accept into historical prior D_t
            self._history.append(movement)
            self._update_covariance_incremental(delta_state)
            self._prev_velocity = movement.velocity
            self._prev_direction = movement.direction
        
        return movement, is_outlier
    
    def _compute_mahalanobis(self, delta_state: np.ndarray) -> float:
        """Compute Mahalanobis distance from current mean."""
        if self._mean is None or self._cov_inv is None:
            return 0.0
        
        if self._n < self.min_samples_for_cov:
            return 0.0
        
        try:
            x = delta_state.astype(np.float64)
            diff = x - self._mean
            # d² = (x - μ)ᵀ Σ⁻¹ (x - μ)
            d2 = float(diff @ self._cov_inv @ diff)
            return float(np.sqrt(max(0.0, d2)))
        except Exception:
            return 0.0
    
    def _update_covariance_incremental(self, x: np.ndarray) -> None:
        """Incremental Welford covariance update with Sherman-Morrison rank-1 inverse.

        Welford's algorithm updates mean and outer-product scatter M2 online:
            n_new = n + 1
            δ = x - μ_old
            μ_new = μ_old + δ / n_new
            δ_2 = x - μ_new
            M2_new = M2_old + δ ⊗ δ_2

        Inverse covariance is updated via the Sherman-Morrison formula in O(d^2):
            Σ_n = c_n · Σ_{n-1} + u_n u_n^T
            where c_n = (n - 2) / (n - 1), u_n = [sqrt(n) / (n - 1)] · δ_2
            Σ_n^-1 = (1 / c_n) · [Σ_{n-1}^-1 - (z_0 z_0^T) / (c_n + u_n^T z_0)]
            with z_0 = Σ_{n-1}^-1 · u_n.

        Exact full inversion is anchored at n=2 and periodically every
        recompute_interval steps to prevent numerical eigenvalue drift.
        Active symmetrization is enforced after each step to prevent IEEE 754 drift.
        """
        # Welford's online algorithm
        self._n += 1
        if self._n == 1:
            # First sample: mean = x, M2 = 0, cov_inv = I
            self._mean = x.copy().astype(np.float64)
            return

        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += np.outer(delta, delta2)

        # Need at least 2 samples for covariance
        if self._n < 2:
            return

        # Re-anchor exact inversion during initial subspace warmup (n <= dim + 1),
        # uninitialized state, or periodic checkpoint
        dim_thresh = (self._state_dim + 1) if self._state_dim is not None else 10
        warmup_thresh = min(self.min_samples_for_cov, dim_thresh)

        if self._n <= warmup_thresh or self._cov_inv is None or self._n % self.recompute_interval == 0:
            cov = self._M2 / (self._n - 1)
            reg = getattr(self, "_reg_matrix", None)
            if reg is None or reg.shape != cov.shape:
                reg = self.regularization_epsilon * np.eye(cov.shape[0], dtype=np.float64)
                self._reg_matrix = reg
            cov_reg = cov + reg
            try:
                self._cov_inv = inv(cov_reg)
            except Exception:
                self._cov_inv = np.linalg.pinv(cov_reg, rcond=self.pinv_rcond)
            self._cov_inv = 0.5 * (self._cov_inv + self._cov_inv.T)
            return

        # Exact rank-1 update via Sherman-Morrison formula:
        c_n = (self._n - 2.0) / (self._n - 1.0)
        u_n = (math.sqrt(float(self._n)) / (self._n - 1.0)) * delta2

        z_0 = self._cov_inv @ u_n
        denom = c_n + float(np.dot(u_n, z_0))

        # Safe division guard: protect against near-singular colinear vectors
        if denom > self.division_epsilon:
            self._cov_inv = (self._cov_inv - np.outer(z_0, z_0) / denom) / c_n
            # Active symmetrization to prevent IEEE 754 drift
            self._cov_inv = 0.5 * (self._cov_inv + self._cov_inv.T)
        else:
            logger.warning(
                "Sherman-Morrison rank-1 update skipped due to near-singular denominator %.4e", denom
            )
            try:
                cov = self._M2 / (self._n - 1)
                cov_reg = cov + getattr(self, "_reg_matrix", self.regularization_epsilon * np.eye(cov.shape[0]))
                self._cov_inv = inv(cov_reg)
                self._cov_inv = 0.5 * (self._cov_inv + self._cov_inv.T)
            except Exception:
                pass
    
    def _update_covariance(self) -> None:
        """Legacy full recompute - kept for compatibility."""
        self._update_covariance_incremental(self._history[-1].delta_state)
    
    def _current_timestamp(self) -> float:
        import time
        return time.time()
    
    def get_history(self) -> list[Movement]:
        """Return current history for Module 2."""
        return list(self._history)
    
    def reset(self) -> None:
        """Reset filter state (e.g., after confirmed regime change)."""
        self._history.clear()
        self._cov_inv = None
        self._mean = None
        self._M2 = None
        self._n = 0
        self._state_dim = None
    
    def soft_reset(self, keep_last: int = 10) -> None:
        """
        Partial reset that preserves the most recent accepted movements.
        Used for automatic regime recovery: keeps covariance adaptive.
        """
        recent = list(self._history)[-keep_last:]
        self._history.clear()
        self._cov_inv = None
        self._mean = None
        self._M2 = None
        self._n = 0
        
        if recent:
            for m in recent:
                x = m.delta_state.astype(np.float64)
                if self._state_dim is None:
                    self._state_dim = len(x)
                    self._mean = x.copy()
                    self._M2 = np.zeros((self._state_dim, self._state_dim), dtype=np.float64)
                    self._n = 1
                    self._cov_inv = np.eye(self._state_dim, dtype=np.float64)
                else:
                    self._n += 1
                    if self._n == 1:
                        self._mean = x.copy()
                    else:
                        delta = x - self._mean
                        self._mean += delta / self._n
                        delta2 = x - self._mean
                        self._M2 += np.outer(delta, delta2)
            
            if self._M2 is not None and self._n >= self.min_samples_for_cov:
                cov = self._M2 / (self._n - 1)
                reg = self.regularization_epsilon * np.eye(cov.shape[0], dtype=np.float64)
                cov_reg = cov + reg
                cond = np.linalg.cond(cov_reg)
                if cond > self.max_condition_number or np.isnan(cond):
                    self._cov_inv = np.linalg.pinv(cov_reg, rcond=self.pinv_rcond)
                else:
                    self._cov_inv = inv(cov_reg)

    # ------------------------------------------------------------------
    # Persistence (StatePersistable contract)
    # ------------------------------------------------------------------

    def export_state(self) -> Dict[str, Any]:
        """Serialize Welford state and accepted history to a JSON-safe dict.

        Persisting (_n, _mean, _M2, _cov_inv) makes the filter warm on
        restore: outliers are detected from the first post-restore event,
        with no re-learning window.
        """
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "noise_threshold": self.noise_threshold,
            "n": self._n,
            "state_dim": self._state_dim,
            "mean": pack_array(self._mean),
            "M2": pack_array(self._M2),
            "cov_inv": pack_array(self._cov_inv),
            "history": [movement_to_raw(m) for m in self._history],
        }

    def import_state(self, payload: Dict[str, Any]) -> None:
        """Restore Welford state and history from an export_state payload."""
        if not isinstance(payload, dict) or "schema_version" not in payload:
            raise ValueError("MahalanobisFilter payload missing schema_version")
        if payload["schema_version"] != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported MahalanobisFilter schema: {payload['schema_version']}"
            )

        state_dim = payload.get("state_dim")
        n = int(payload.get("n", 0))
        mean = unpack_array(payload.get("mean"))
        M2 = unpack_array(payload.get("M2"))
        cov_inv = unpack_array(payload.get("cov_inv"))

        if n > 0 and (mean is None or M2 is None or state_dim is None):
            raise ValueError("MahalanobisFilter payload inconsistent: n>0 without Welford fields")
        if n == 0 and (mean is not None or cov_inv is not None):
            raise ValueError("MahalanobisFilter payload inconsistent: n=0 with covariance")

        history_raw = payload.get("history", [])
        if not isinstance(history_raw, list):
            raise ValueError("MahalanobisFilter payload 'history' must be a list")
        for raw in history_raw:
            if not isinstance(raw, dict) or "delta_state" not in raw:
                raise ValueError("Malformed movement entry in history")
        history = movements_from_raw(history_raw)

        self.reset()
        if history:
            self._history.extend(history)
            # Restore dimension from the movements themselves when absent
            # (cold snapshots taken before any accepted sample).
            if state_dim is None:
                state_dim = len(history[0].delta_state)

        self._state_dim = state_dim
        self._n = n
        self._mean = mean
        self._M2 = M2
        self._cov_inv = cov_inv