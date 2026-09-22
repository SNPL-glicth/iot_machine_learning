"""Unified ML Metrics Taxonomy.

Canonical metric types for all ML models in the system.
Each metric is explicitly bounded and semantically precise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar('T', bound=float)


@dataclass(frozen=True, slots=True)
class BoundedMetric:
    """Base class for metrics bounded in [0.0, 1.0]."""
    value: float
    
    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"{self.__class__.__name__} must be in [0.0, 1.0], got {self.value}")
    
    def __float__(self) -> float:
        return self.value
    
    def __add__(self, other: 'BoundedMetric') -> float:
        return self.value + float(other)
    
    def __radd__(self, other: float) -> float:
        return other + self.value
    
    def __mul__(self, other: float) -> float:
        return self.value * other
    
    def __rmul__(self, other: float) -> float:
        return other * self.value
    
    def __lt__(self, other: 'BoundedMetric') -> bool:
        return self.value < float(other)
    
    def __le__(self, other: 'BoundedMetric') -> bool:
        return self.value <= float(other)
    
    def __gt__(self, other: 'BoundedMetric') -> bool:
        return self.value > float(other)
    
    def __ge__(self, other: 'BoundedMetric') -> bool:
        return self.value >= float(other)


@dataclass(frozen=True, slots=True)
class PredictionConfidence(BoundedMetric):
    """Individual expert model confidence in [0.0, 1.0].
    
    Semantics: The expert's self-assessed prediction quality.
    - Taylor: Fit quality of polynomial coefficients
    - Kalman: 1 / (1 + position_covariance)
    - Statistical: 1 - noise_ratio
    - Baseline: Heuristic based on window stability
    """
    pass


@dataclass(frozen=True, slots=True)
class TrajectoryCoherence(BoundedMetric):
    """Trajectory coherence score Φ_Ritmo in [0.0, 1.0].
    
    Semantics: Normalized information gain + rhythm consistency per unit length.
    Formula: (λ_t * ΔH + ρ) / |T|
    """
    pass


@dataclass(frozen=True, slots=True)
class MoEConfidence(BoundedMetric):
    """Mixture-of-Experts consensus confidence Φ_MoE in [0.0, 1.0].
    
    Semantics: Weighted jury score with variance penalty.
    Formula: (Σ w_e * Ψ_e) / (1 + γ * Var(Ψ))
    Multiplicative veto: If any critical expert Ψ_k < τ_k → 0.0
    """
    pass


@dataclass(frozen=True, slots=True)
class ExecutionConfidence(BoundedMetric):
    """Final composite confidence for action execution in [0.0, 1.0].
    
    Semantics: Post-veto MoE confidence driving ActionEnvelope magnitude.
    Equals MoEConfidence when no veto; 0.0 when vetoed.
    """
    pass


@dataclass(frozen=True, slots=True)
class DriftSeverity(BoundedMetric):
    """Normalized concept drift severity in [0.0, 1.0].
    
    Semantics: Max normalized drift score across all sensors.
    """
    pass


@dataclass(frozen=True, slots=True)
class TransitionEntropy(BoundedMetric):
    """Normalized posterior entropy H(Θ|D_t) in [0.0, 1.0].
    
    Semantics: Shannon entropy of transition posterior, normalized by log2(K).
    """
    pass


@dataclass(frozen=True, slots=True)
class ExplorationFactor(BoundedMetric):
    """Exploration factor λ_t in [0.0, 1.0].
    
    Semantics: min(H(Θ)/H_max, 1 - DriftScore). Modulates exploitation vs exploration.
    """
    pass


@dataclass(frozen=True, slots=True)
class ActionMagnitude(BoundedMetric):
    """Normalized action intensity in [0.0, 1.0].
    
    Semantics: Maps ExecutionConfidence → position sizing via confidence bands.
    """
    pass


# Diagnostic metrics (may exceed 1.0, documented bounds)
@dataclass(frozen=True, slots=True)
class PositionCovariance:
    """Kalman filter position covariance P[0,0]. Range: [0.0, ∞)."""
    value: float
    
    def __post_init__(self):
        if self.value < 0.0:
            raise ValueError(f"PositionCovariance must be >= 0.0, got {self.value}")


@dataclass(frozen=True, slots=True)
class PredictionVariance:
    """Inter-expert prediction variance Var(Ψ_e). Range: [0.0, ∞)."""
    value: float
    
    def __post_init__(self):
        if self.value < 0.0:
            raise ValueError(f"PredictionVariance must be >= 0.0, got {self.value}")


@dataclass(frozen=True, slots=True)
class AnomalySeverity:
    """Stored anomaly score from memory. Range: [0.0, 1.0]."""
    value: float
    
    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"AnomalySeverity must be in [0.0, 1.0], got {self.value}")


# Metric factory functions for backward compatibility
def as_prediction_confidence(value: float) -> PredictionConfidence:
    return PredictionConfidence(float(value))


def as_trajectory_coherence(value: float) -> TrajectoryCoherence:
    return TrajectoryCoherence(float(value))


def as_moe_confidence(value: float) -> MoEConfidence:
    return MoEConfidence(float(value))


def as_execution_confidence(value: float) -> ExecutionConfidence:
    return ExecutionConfidence(float(value))


def as_drift_severity(value: float) -> DriftSeverity:
    return DriftSeverity(float(value))


def as_transition_entropy(value: float) -> TransitionEntropy:
    return TransitionEntropy(float(value))


def as_exploration_factor(value: float) -> ExplorationFactor:
    return ExplorationFactor(float(value))


def as_action_magnitude(value: float) -> ActionMagnitude:
    return ActionMagnitude(float(value))