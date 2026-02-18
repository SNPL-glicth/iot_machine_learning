"""Contextual weight calculation from MAE history.

Pure computation: given a dict of MAEs per engine, returns normalized
inverse-MAE weights. No state, no I/O, no threading.

Extracted from ContextualPlasticityTracker to keep that class focused
on error recording and retrieval only.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def compute_inverse_mae_weights(
    maes: Dict[str, float],
    epsilon: float = 0.1,
) -> Dict[str, float]:
    """Compute normalized inverse-MAE weights.

    Formula: weight_i = 1 / (mae_i + epsilon), then normalize to sum=1.

    Args:
        maes: Dict mapping engine_name → MAE value (all >= 0).
        epsilon: Small constant to prevent division by zero.

    Returns:
        Normalized weight dict. Falls back to uniform if total < 1e-9.
    """
    if not maes:
        return {}

    raw: Dict[str, float] = {
        name: 1.0 / (mae + epsilon) for name, mae in maes.items()
    }
    total = sum(raw.values())
    if total < 1e-9:
        uniform = 1.0 / len(maes)
        return {name: uniform for name in maes}

    return {name: w / total for name, w in raw.items()}


def resolve_contextual_weights(
    maes: Dict[str, Optional[float]],
    engine_names: List[str],
    epsilon: float = 0.1,
) -> Optional[Dict[str, float]]:
    """Resolve contextual weights given a MAE map that may have None values.

    Returns None if any engine lacks sufficient data (None MAE), signalling
    the caller to fall back to base weights.

    Args:
        maes: Dict mapping engine_name → MAE or None (insufficient data).
        engine_names: Ordered list of engine names to include.
        epsilon: Forwarded to compute_inverse_mae_weights.

    Returns:
        Normalized weights or None if any engine has insufficient data.
    """
    complete: Dict[str, float] = {}
    for name in engine_names:
        mae = maes.get(name)
        if mae is None:
            return None
        complete[name] = mae

    return compute_inverse_mae_weights(complete, epsilon=epsilon)
