"""Data structures for the cognitive orchestration layer.

Pure value objects — no I/O, no state, no side effects.

Hierarchy:
    SignalProfile    — structural features extracted from raw signal
    EnginePerception — one engine's prediction + diagnostic + confidence
    InhibitionState  — per-engine suppression state
    MetaDiagnostic   — full reasoning trace of the orchestrator
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

try:
    from prometheus_client import Histogram
except ImportError:
    Histogram = None

# Prometheus metric: pipeline phase duration
if Histogram is not None:
    PIPELINE_PHASE_DURATION = Histogram(
        "zenin_pipeline_phase_duration_seconds",
        "Duración de cada fase del pipeline cognitivo",
        ["phase_name"]
    )
else:
    PIPELINE_PHASE_DURATION = None

from iot_machine_learning.domain.entities.series.structural_analysis import (
    StructuralAnalysis,
)


@dataclass
class PipelineTimer:
    """Per-phase latency tracker for the cognitive pipeline.

    Records wall-clock milliseconds for each phase of
    Perceive → Predict → Inhibit → Adapt → Fuse → Explain.
    Provides budget checking: if total exceeds ``budget_ms``,
    ``is_over_budget`` returns True so the orchestrator can
    cut to fallback.

    Not frozen — phases are recorded incrementally.
    """

    budget_ms: float = 500.0
    perceive_ms: float = 0.0
    predict_ms: float = 0.0
    inhibit_ms: float = 0.0
    adapt_ms: float = 0.0
    fuse_ms: float = 0.0
    explain_ms: float = 0.0
    _start: float = field(default=0.0, repr=False)

    def start(self) -> None:
        """Mark the beginning of a phase measurement."""
        self._start = time.perf_counter()

    def stop(self, phase: str) -> float:
        """Stop timing and record elapsed ms for *phase*.

        Returns the elapsed milliseconds.
        """
        elapsed = (time.perf_counter() - self._start) * 1000.0
        if hasattr(self, f"{phase}_ms"):
            setattr(self, f"{phase}_ms", elapsed)
        if PIPELINE_PHASE_DURATION is not None:
            PIPELINE_PHASE_DURATION.labels(phase_name=phase).observe(elapsed / 1000.0)
        return elapsed

    @property
    def total_ms(self) -> float:
        return (self.perceive_ms + self.predict_ms + self.inhibit_ms
                + self.adapt_ms + self.fuse_ms + self.explain_ms)

    @property
    def is_over_budget(self) -> bool:
        return self.total_ms > self.budget_ms

    def to_dict(self) -> dict:
        return {
            "perceive_ms": round(self.perceive_ms, 3),
            "predict_ms": round(self.predict_ms, 3),
            "inhibit_ms": round(self.inhibit_ms, 3),
            "adapt_ms": round(self.adapt_ms, 3),
            "fuse_ms": round(self.fuse_ms, 3),
            "explain_ms": round(self.explain_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "budget_ms": self.budget_ms,
            "over_budget": self.is_over_budget,
        }


# ARCH-SEV-1: SignalProfile is now an alias to StructuralAnalysis
# This eliminates duplication and establishes StructuralAnalysis as source of truth
SignalProfile = StructuralAnalysis


def _deprecated_signal_profile_warning():
    """Emit deprecation warning for SignalProfile usage."""
    warnings.warn(
        "SignalProfile is deprecated and will be removed in version 3.0. "
        "Use StructuralAnalysis from domain.entities.series.structural_analysis instead. "
        "SignalProfile is now just an alias to StructuralAnalysis.",
        DeprecationWarning,
        stacklevel=3,
    )


# Monkey-patch __init__ to emit warning on instantiation
_original_structural_analysis_init = StructuralAnalysis.__init__


def _patched_init(self, *args, **kwargs):
    """Patched init that warns if called via SignalProfile alias."""
    import inspect
    frame = inspect.currentframe()
    if frame and frame.f_back:
        caller_locals = frame.f_back.f_locals
        # Check if caller is using the SignalProfile name
        if 'SignalProfile' in frame.f_back.f_code.co_names:
            _deprecated_signal_profile_warning()
    _original_structural_analysis_init(self, *args, **kwargs)


# Note: Actual patching commented out to avoid breaking existing code
# Uncomment when ready to enforce migration:
# StructuralAnalysis.__init__ = _patched_init


# Legacy to_dict method for backward compatibility
def _signal_profile_to_dict_legacy(self) -> dict:
    """Legacy to_dict for SignalProfile compatibility."""
    return {
        "n_points": self.n_points,
        "mean": round(self.mean, 6),
        "std": round(self.std, 6),
        "noise_ratio": round(self.noise_ratio, 6),
        "slope": round(self.slope, 6),
        "curvature": round(self.curvature, 6),
        "regime": self.regime,
            "dt": self.dt,
        }


@dataclass(frozen=True)
class EnginePerception:
    """One engine's partial perception of the signal.

    Produced by running a single engine.  The orchestrator collects
    perceptions from all engines and fuses them.

    Attributes:
        engine_name: Identifier of the engine that produced this.
        predicted_value: Raw prediction (before fusion).
        confidence: Engine's self-reported confidence [0, 1].
        trend: Detected trend direction.
        stability: Stability indicator (0 = stable, 1 = unstable).
        local_fit_error: RMS residual of the engine's local model.
        metadata: Engine-specific diagnostic data.
    """

    engine_name: str
    predicted_value: float
    confidence: float
    trend: Literal["up", "down", "stable"] = "stable"
    stability: float = 0.0
    local_fit_error: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "engine_name": self.engine_name,
            "predicted_value": round(self.predicted_value, 6),
            "confidence": round(self.confidence, 4),
            "trend": self.trend,
            "stability": round(self.stability, 4),
            "local_fit_error": round(self.local_fit_error, 6),
        }


@dataclass(frozen=True)
class InhibitionState:
    """Per-engine inhibition (weight suppression) state.

    Analogous to neural inhibition: an engine with high instability,
    high fit error, or high recent prediction error gets its weight
    suppressed toward zero temporarily.

    Attributes:
        engine_name: Which engine this applies to.
        base_weight: Weight before inhibition [0, 1].
        inhibited_weight: Weight after inhibition [0, 1].
        inhibition_reason: Why the weight was suppressed (or "none").
        suppression_factor: How much the weight was reduced (0 = no suppression).
    """

    engine_name: str
    base_weight: float
    inhibited_weight: float
    inhibition_reason: str = "none"
    suppression_factor: float = 0.0

    def to_dict(self) -> dict:
        return {
            "engine_name": self.engine_name,
            "base_weight": round(self.base_weight, 4),
            "inhibited_weight": round(self.inhibited_weight, 4),
            "inhibition_reason": self.inhibition_reason,
            "suppression_factor": round(self.suppression_factor, 4),
        }


@dataclass(frozen=True)
class MetaDiagnostic:
    """Full reasoning trace of the cognitive orchestrator.

    Documents *why* the orchestrator chose a particular prediction,
    which engines contributed, and how weights were determined.

    Attributes:
        signal_profile: Structural analysis of the input signal.
        perceptions: Each engine's partial perception.
        inhibition_states: Per-engine inhibition decisions.
        final_weights: Normalized weights used for fusion.
        selected_engine: Primary engine (highest weight).
        selection_reason: Human-readable explanation.
        fusion_method: "weighted_average" or "single_engine".
        fallback_reason: If fallback was used, why.
    """

    signal_profile: StructuralAnalysis
    perceptions: List[EnginePerception]
    inhibition_states: List[InhibitionState]
    final_weights: Dict[str, float]
    selected_engine: str
    selection_reason: str
    fusion_method: str = "weighted_average"
    fallback_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "signal_profile": self.signal_profile.to_dict(),
            "perceptions": [p.to_dict() for p in self.perceptions],
            "inhibition_states": [s.to_dict() for s in self.inhibition_states],
            "final_weights": {
                k: round(v, 4) for k, v in self.final_weights.items()
            },
            "selected_engine": self.selected_engine,
            "selection_reason": self.selection_reason,
            "fusion_method": self.fusion_method,
            "fallback_reason": self.fallback_reason,
        }
