"""L2 Regularization for Bayesian Weight Tracker.

Prevents overfitting by pulling weights toward uniform distribution.
Implements ridge regularization: w_new = (1-λ)w_bayesian + λw_uniform
"""

from __future__ import annotations

from typing import Dict


def apply_l2_regularization(
    accuracies: Dict[str, float],
    engine_names: list[str],
    regularization_strength: float = 0.01,
) -> Dict[str, float]:
    """Apply L2 regularization toward uniform weights.
    
    Args:
        accuracies: Per-engine accuracy scores
        engine_names: List of engine names
        regularization_strength: λ ∈ [0, 1]. 0=no regularization, 1=full uniform
    
    Returns:
        Regularized accuracies
    """
    if regularization_strength <= 0.0 or len(engine_names) == 0:
        return accuracies
    
    # Clamp λ to [0, 1]
    lambda_ = max(0.0, min(1.0, regularization_strength))
    
    # Uniform target: all engines equal
    uniform_accuracy = 1.0 / len(engine_names)
    
    # Apply ridge: (1-λ)·bayesian + λ·uniform
    regularized = {}
    for engine in engine_names:
        bayesian_acc = accuracies.get(engine, uniform_accuracy)
        regularized[engine] = (
            (1.0 - lambda_) * bayesian_acc +
            lambda_ * uniform_accuracy
        )
    
    return regularized


def compute_regularization_strength(
    n_updates: int,
    base_strength: float = 0.01,
    decay_rate: float = 0.95,
    min_strength: float = 0.001,
    drift_score: float = 0.0,
) -> float:
    """Compute adaptive regularization strength.

    Statistical rationale:
    - Regularization should INCREASE when concept drift is detected,
      preventing overfitting to stale data.
    - Regularization should never decay to zero (always retain some
      shrinkage toward uniform weights for robustness).
    - With no drift (drift_score=0), strength equals base_strength.

    Formula: lambda = max(min_lambda, base_lambda * (1 + |drift_score|))

    Args:
        n_updates: Number of updates so far (retained for backward compat,
            not used in new formula).
        base_strength: Base λ (never goes below this when drift=0).
        decay_rate: Retained for backward compat signature (unused).
        min_strength: Hard floor — λ never decays below this value.
        drift_score: Normalized drift magnitude ≥ 0. Higher → more
            regularization to counter stale learned weights.

    Returns:
        Adaptive λ ≥ min_strength, increases monotonically with drift.
    """
    # Drift-aware: lambda increases with detected drift, never decays to zero.
    # When drift_score > 0, we increase regularization to prevent the tracker
    # from overfitting to a regime that may have fundamentally changed.
    adaptive_lambda = base_strength * (1.0 + abs(drift_score))
    return max(min_strength, adaptive_lambda)
