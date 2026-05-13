"""Phase Timer for pipeline timeout enforcement (SEVERO-4).

Distributes total pipeline budget across phases using configurable weights.
Raises TimeoutError if remaining time is insufficient for a phase.

Applies OCP: PipelineExecutor accepts PhaseTimer as injectable dependency.
"""

from __future__ import annotations

import time
from typing import Dict, Optional


class PhaseTimer:
    """Timer for enforcing phase-level timeouts (SEVERO-4).
    
    Distributes total pipeline budget across phases using weights.
    Each phase checks if remaining time is sufficient before starting.
    
    Attributes:
        _total_budget_ms: Total pipeline budget in milliseconds.
        _phase_weights: Dict mapping phase name → weight (0-1, sum=1.0).
        _start_time: Pipeline start time (monotonic).
        _current_phase: Currently executing phase name.
    
    Applies OCP: Weights are configurable, no hardcoded phase logic.
    """
    
    # Default phase weights (sum = 1.0)
    DEFAULT_PHASE_WEIGHTS: Dict[str, float] = {
        "sanitize": 0.05,
        "boundary_check": 0.02,
        "seasonal_decomposition": 0.03,
        "perceive": 0.05,
        "drift_detection": 0.05,
        "predict": 0.35,
        "adapt": 0.10,
        "inhibit": 0.10,
        "fuse": 0.10,
        "decision_arbiter": 0.02,
        "coherence_check": 0.02,
        "confidence_calibration": 0.01,
        "explain": 0.05,
        "action_guard": 0.02,
        "narrative": 0.03,
    }
    
    def __init__(
        self,
        total_budget_ms: float,
        phase_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize phase timer.
        
        Args:
            total_budget_ms: Total pipeline budget in milliseconds.
            phase_weights: Optional custom phase weights (default: DEFAULT_PHASE_WEIGHTS).
        
        Raises:
            ValueError: If total_budget_ms <= 0 or weights don't sum to ~1.0.
        """
        if total_budget_ms <= 0:
            raise ValueError(f"total_budget_ms must be > 0, got {total_budget_ms}")
        
        self._total_budget_ms = total_budget_ms
        self._phase_weights = phase_weights or self.DEFAULT_PHASE_WEIGHTS.copy()
        self._start_time: Optional[float] = None
        self._current_phase: Optional[str] = None
        
        # Validate weights sum to ~1.0
        weight_sum = sum(self._phase_weights.values())
        if not (0.99 <= weight_sum <= 1.01):
            raise ValueError(
                f"Phase weights must sum to ~1.0, got {weight_sum:.3f}. "
                f"Weights: {self._phase_weights}"
            )
    
    def start(self) -> None:
        """Start the timer.
        
        Should be called at the beginning of pipeline execution.
        """
        self._start_time = time.monotonic()
        self._current_phase = None
    
    def start_phase(self, phase_name: str) -> None:
        """Start a phase and check if sufficient time remains.
        
        Args:
            phase_name: Name of the phase starting.
        
        Raises:
            TimeoutError: If remaining time < phase_budget * 0.5.
        
        Applies SEVERO-4: Proactive timeout enforcement.
        """
        if self._start_time is None:
            raise RuntimeError("Timer not started. Call start() first.")
        
        # Calculate remaining time
        elapsed_ms = (time.monotonic() - self._start_time) * 1000
        remaining_ms = self._total_budget_ms - elapsed_ms
        
        # Get phase budget
        phase_weight = self._phase_weights.get(phase_name, 0.05)  # Default 5% if unknown
        phase_budget_ms = self._total_budget_ms * phase_weight
        
        # SEVERO-4: Raise if remaining < 50% of phase budget
        if remaining_ms < (phase_budget_ms * 0.5):
            raise TimeoutError(
                f"Insufficient time for phase '{phase_name}': "
                f"remaining={remaining_ms:.1f}ms, "
                f"phase_budget={phase_budget_ms:.1f}ms, "
                f"required_min={phase_budget_ms * 0.5:.1f}ms"
            )
        
        self._current_phase = phase_name
    
    def get_remaining_ms(self) -> float:
        """Get remaining time in milliseconds.
        
        Returns:
            Remaining time in milliseconds, or total budget if not started.
        """
        if self._start_time is None:
            return self._total_budget_ms
        
        elapsed_ms = (time.monotonic() - self._start_time) * 1000
        return max(0.0, self._total_budget_ms - elapsed_ms)
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds, or 0.0 if not started.
        """
        if self._start_time is None:
            return 0.0
        
        return (time.monotonic() - self._start_time) * 1000
    
    def get_phase_budget_ms(self, phase_name: str) -> float:
        """Get budget for a specific phase.
        
        Args:
            phase_name: Phase name.
        
        Returns:
            Phase budget in milliseconds.
        """
        weight = self._phase_weights.get(phase_name, 0.05)
        return self._total_budget_ms * weight
    
    def is_over_budget(self) -> bool:
        """Check if total budget has been exceeded.
        
        Returns:
            True if elapsed time > total budget.
        """
        return self.get_remaining_ms() <= 0.0
    
    def get_metrics(self) -> Dict[str, float]:
        """Get timer metrics.
        
        Returns:
            Dict with elapsed, remaining, and budget info.
        """
        return {
            "total_budget_ms": self._total_budget_ms,
            "elapsed_ms": self.get_elapsed_ms(),
            "remaining_ms": self.get_remaining_ms(),
            "current_phase": self._current_phase or "none",
            "is_over_budget": self.is_over_budget(),
        }
