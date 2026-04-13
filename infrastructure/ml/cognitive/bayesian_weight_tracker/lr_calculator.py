"""Learning rate calculation algorithms.

Pure calculation functions without state management.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from iot_machine_learning.domain.entities.series.structural_analysis import RegimeType

logger = logging.getLogger(__name__)


# Regime factors for learning rate adjustment (ISO 25010: configurable via flags in Phase 2)
REGIME_FACTORS: dict[RegimeType, float] = {
    RegimeType.STABLE: 1.0,
    RegimeType.TRENDING: 1.2,
    RegimeType.VOLATILE: 1.5,
    RegimeType.NOISY: 0.8,
    RegimeType.TRANSITIONAL: 1.1,
    RegimeType.UNKNOWN: 1.0,
}

# Noise threshold for penalty application (Phase 2: migrate to ML_BAYES_NOISE_THRESHOLD flag)
NOISE_PENALTY_THRESHOLD: float = 0.3


def get_regime_factor(regime: RegimeType) -> float:
    """Get learning rate adjustment factor for signal regime."""
    return REGIME_FACTORS.get(regime, 1.0)


def compute_learning_rate(
    error: float,
    context: PlasticityContext,
    base_lr: float,
    lr_min: float,
    lr_max: float,
    error_scale: float,
    noise_penalty: float,
    failure_boost: float,
    failure_threshold: int,
) -> float:
    """Compute adaptive learning rate.
    
    Algorithm:
    1. Start with base_lr
    2. Scale by log(1 + error/error_scale)
    3. Adjust for regime
    4. Reduce by noise_penalty if noise_ratio > 0.3
    5. Boost by failure_boost if consecutive_failures > threshold
    6. Clip to [lr_min, lr_max]
    
    Returns:
        Adaptive learning rate in [lr_min, lr_max]
    """
    if error < 0:
        raise ValueError(f"error must be >= 0, got {error}")
    
    # 1. Start with base
    lr = base_lr
    
    # 2. Scale by error magnitude
    error_factor = math.log1p(error / error_scale)
    lr *= (1.0 + error_factor)
    
    # 3. Adjust for regime
    lr *= get_regime_factor(context.regime)
    
    # 4. Noise penalty
    if context.noise_ratio > NOISE_PENALTY_THRESHOLD:
        lr *= noise_penalty
        logger.debug(
            "lr_noise_penalty",
            extra={"noise_ratio": context.noise_ratio, "lr": lr},
        )
    
    # 5. Failure boost
    if context.consecutive_failures >= failure_threshold:
        lr *= failure_boost
        logger.debug(
            "lr_failure_boost",
            extra={"failures": context.consecutive_failures, "lr": lr},
        )
    
    # 6. Safety limits
    lr = float(np.clip(lr, lr_min, lr_max))
    
    logger.debug(
        "lr_computed",
        extra={
            "error": error,
            "regime": context.regime.value,
            "final_lr": lr,
        },
    )
    
    return lr
