"""Weight calculation algorithms for plasticity.

Pure functions for computing regime-contextual weights.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def apply_ttl_decay(
    weights: Dict[str, float],
    regime_last_update: float,
    regime_ttl_seconds: float,
    now: float,
) -> Dict[str, float]:
    """Apply TTL decay to weights based on elapsed time.
    
    Exponential decay to equilibrium (uniform distribution).
    
    Args:
        weights: Current weights dict
        regime_last_update: Last update timestamp
        regime_ttl_seconds: TTL for full decay
        now: Current timestamp
    
    Returns:
        Decayed weights
    """
    elapsed = now - regime_last_update
    if elapsed <= 0:
        return weights
    
    n = len(weights)
    if n == 0:
        return weights
    
    w_eq = 1.0 / n
    decay_rate = 1.0 / regime_ttl_seconds
    decay_factor = math.exp(-decay_rate * elapsed)
    
    return {
        eng: w * decay_factor + w_eq * (1.0 - decay_factor)
        for eng, w in weights.items()
    }


def compute_weights_from_accuracy(
    engine_names: List[str],
    regime_data: Dict[str, float],
    min_weight: float,
) -> Dict[str, float]:
    """Compute normalized weights from accuracy data.
    
    Args:
        engine_names: List of engine names to compute weights for
        regime_data: Dict of {engine_name: accuracy}
        min_weight: Minimum weight floor
    
    Returns:
        Normalized weights summing to 1.0
    """
    if not engine_names:
        return {}
    
    n = len(engine_names)
    
    # No history - uniform
    if not regime_data:
        uniform = 1.0 / n
        return {name: uniform for name in engine_names}
    
    # Compute raw weights with floor
    raw = {
        name: max(min_weight, regime_data.get(name, min_weight))
        for name in engine_names
    }
    
    # Normalize
    total = sum(raw.values())
    if total < 1e-12:
        uniform = 1.0 / n
        return {name: uniform for name in engine_names}
    
    return {name: w / total for name, w in raw.items()}


def compute_weights_with_decay(
    engine_names: List[str],
    accuracy_data: Dict[str, float],
    regime_last_update: float,
    regime_ttl_seconds: float,
    min_weight: float,
    now: float,
) -> Dict[str, float]:
    """Compute weights with TTL decay applied.
    
    Convenience function combining compute_weights_from_accuracy + apply_ttl_decay.
    """
    weights = compute_weights_from_accuracy(engine_names, accuracy_data, min_weight)
    
    if regime_last_update > 0 and now - regime_last_update > regime_ttl_seconds:
        weights = apply_ttl_decay(weights, regime_last_update, regime_ttl_seconds, now)
    
    # Re-apply floor after decay
    return {eng: max(min_weight, w) for eng, w in weights.items()}
