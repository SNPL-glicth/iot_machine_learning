"""Adaptive Learning Rate calculator for contextual plasticity.

Computes dynamic learning rates based on error magnitude, signal regime,
noise levels, and failure patterns. Implements logarithmic scaling with
safety limits to prevent instability.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np

from iot_machine_learning.domain.entities.plasticity.plasticity_context import PlasticityContext, RegimeType

logger = logging.getLogger(__name__)


class AdaptiveLearningRate:
    """Calculates adaptive learning rates for contextual plasticity.
    
    Learning rate adapts based on:
    - Error magnitude (logarithmic scaling)
    - Signal regime (STABLE vs VOLATILE vs NOISY)
    - Noise ratio (reduces lr when noisy)
    - Consecutive failures (increases lr to escape local minima)
    
    Safety limits: lr ∈ [0.001, 0.2]
    
    Attributes:
        base_lr: Base learning rate (default: 0.05)
        lr_min: Minimum learning rate (default: 0.001)
        lr_max: Maximum learning rate (default: 0.2)
        error_scale: Scaling factor for error magnitude (default: 10.0)
        noise_penalty: Penalty factor for high noise (default: 0.5)
        failure_boost: Boost factor for consecutive failures (default: 2.0)
        failure_threshold: Failures needed to trigger boost (default: 5)
    
    Examples:
        >>> calc = AdaptiveLearningRate()
        >>> ctx = PlasticityContext(
        ...     regime=RegimeType.STABLE,
        ...     noise_ratio=0.1,
        ...     volatility=0.2,
        ...     time_of_day=14,
        ...     consecutive_failures=0,
        ...     error_magnitude=5.0,
        ...     is_critical_zone=False,
        ... )
        >>> lr = calc.compute_adaptive_lr(error=5.0, context=ctx)
        >>> 0.001 <= lr <= 0.2
        True
    """
    
    def __init__(
        self,
        base_lr: float = 0.05,
        lr_min: float = 0.001,
        lr_max: float = 0.2,
        error_scale: float = 10.0,
        noise_penalty: float = 0.5,
        failure_boost: float = 2.0,
        failure_threshold: int = 5,
        momentum: float = 0.9,
        momentum_decay: float = 0.95,
    ) -> None:
        """Initialize adaptive learning rate calculator.
        
        Args:
            base_lr: Base learning rate
            lr_min: Minimum learning rate (safety limit)
            lr_max: Maximum learning rate (safety limit)
            error_scale: Scaling factor for error magnitude
            noise_penalty: Penalty factor for high noise (0.5 = 50% reduction)
            failure_boost: Boost factor for consecutive failures
            failure_threshold: Number of failures to trigger boost
            momentum_decay: Decay factor for velocity (default: 0.95)
        
        Raises:
            ValueError: If parameters are invalid
        """
        if not 0 < lr_min < lr_max:
            raise ValueError(f"Must have 0 < lr_min < lr_max, got {lr_min}, {lr_max}")
        if not 0 < base_lr <= lr_max:
            raise ValueError(f"base_lr must be in (0, lr_max], got {base_lr}")
        if error_scale <= 0:
            raise ValueError(f"error_scale must be > 0, got {error_scale}")
        if not 0 < noise_penalty <= 1:
            raise ValueError(f"noise_penalty must be in (0, 1], got {noise_penalty}")
        if failure_boost < 1:
            raise ValueError(f"failure_boost must be >= 1, got {failure_boost}")
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if not 0 < momentum_decay <= 1:
            raise ValueError(f"momentum_decay must be in (0, 1], got {momentum_decay}")
        
        self.base_lr = base_lr
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.error_scale = error_scale
        self.noise_penalty = noise_penalty
        self.failure_boost = failure_boost
        self.failure_threshold = failure_threshold
        self.momentum = momentum
        self.momentum_decay = momentum_decay
        self._velocity: OrderedDict[Tuple[str, str], float] = OrderedDict()
        self._max_velocity_entries = 10000
    
    def compute_adaptive_lr(
        self,
        error: float,
        context: PlasticityContext,
        engine_name: Optional[str] = None,
        series_id: Optional[str] = None,
    ) -> float:
        """Compute adaptive learning rate based on error and context.
        
        Algorithm:
        1. Start with base_lr
        2. Scale by log(1 + error/error_scale) for error magnitude
        3. Adjust for regime (STABLE=1.0, TRENDING=1.2, VOLATILE=1.5, NOISY=0.8)
        4. Reduce by noise_penalty if noise_ratio > 0.3
        5. Boost by failure_boost if consecutive_failures > threshold
        6. Apply momentum if engine_name provided
        7. Clip to [lr_min, lr_max]
        
        Args:
            error: Absolute prediction error
            context: Plasticity context with regime, noise, failures
            engine_name: Optional engine name for momentum tracking
        
        Returns:
            Adaptive learning rate in [lr_min, lr_max]
        
        Raises:
            ValueError: If error is negative
        
        Examples:
            >>> calc = AdaptiveLearningRate()
            >>> ctx = PlasticityContext.create_default(RegimeType.STABLE)
            >>> lr = calc.compute_adaptive_lr(10.0, ctx)
            >>> 0.001 <= lr <= 0.2
            True
        """
        if error < 0:
            raise ValueError(f"error must be >= 0, got {error}")
        
        # 1. Start with base learning rate
        lr = self.base_lr
        
        # 2. Scale logarithmically by error magnitude
        # log(1 + x) grows slowly, preventing lr explosion on large errors
        error_factor = math.log1p(error / self.error_scale)
        lr *= (1.0 + error_factor)
        
        # 3. Adjust for signal regime
        regime_factor = self._get_regime_factor(context.regime)
        lr *= regime_factor
        
        # 4. Penalize high noise (reduces lr by 50% if noise_ratio > 0.3)
        if context.noise_ratio > 0.3:
            lr *= self.noise_penalty
            logger.debug(
                "adaptive_lr_noise_penalty",
                extra={
                    "noise_ratio": context.noise_ratio,
                    "penalty": self.noise_penalty,
                    "lr_after_penalty": lr,
                },
            )
        
        # 5. Boost on consecutive failures (escape local minima)
        if context.consecutive_failures >= self.failure_threshold:
            lr *= self.failure_boost
            logger.debug(
                "adaptive_lr_failure_boost",
                extra={
                    "consecutive_failures": context.consecutive_failures,
                    "boost": self.failure_boost,
                    "lr_after_boost": lr,
                },
            )
        
        # 6. Apply momentum if engine_name provided
        if engine_name is not None and self.momentum > 0:
            key = (series_id, engine_name) if series_id else (engine_name, engine_name)
            prev_velocity = self._velocity.get(key, self.base_lr)
            prev_velocity *= self.momentum_decay
            lr = self.momentum * prev_velocity + (1.0 - self.momentum) * lr
            
            # LRU eviction
            if len(self._velocity) >= self._max_velocity_entries:
                self._velocity.popitem(last=False)
            
            self._velocity[key] = lr
            if key in self._velocity:
                self._velocity.move_to_end(key)
        
        # 7. Apply safety limits
        lr = np.clip(lr, self.lr_min, self.lr_max)
        
        logger.debug(
            "adaptive_lr_computed",
            extra={
                "error": error,
                "regime": context.regime.value,
                "noise_ratio": context.noise_ratio,
                "consecutive_failures": context.consecutive_failures,
                "final_lr": lr,
            },
        )
        
        return float(lr)
    
    def _get_regime_factor(self, regime: RegimeType) -> float:
        """Get learning rate adjustment factor for signal regime.
        
        Regime factors:
        - STABLE: 1.0 (baseline)
        - TRENDING: 1.2 (slightly faster learning)
        - VOLATILE: 1.5 (faster adaptation needed)
        - NOISY: 0.8 (slower, more conservative)
        - UNKNOWN: 1.0 (neutral)
        
        Args:
            regime: Signal regime type
        
        Returns:
            Adjustment factor for learning rate
        """
        regime_factors = {
            RegimeType.STABLE: 1.0,
            RegimeType.TRENDING: 1.2,
            RegimeType.VOLATILE: 1.5,
            RegimeType.NOISY: 0.8,
            RegimeType.UNKNOWN: 1.0,
        }
        return regime_factors.get(regime, 1.0)
    
    def compute_batch_lr(
        self,
        errors: list[float],
        contexts: list[PlasticityContext],
    ) -> list[float]:
        """Compute adaptive learning rates for a batch of errors.
        
        Args:
            errors: List of prediction errors
            contexts: List of plasticity contexts (same length as errors)
        
        Returns:
            List of adaptive learning rates
        
        Raises:
            ValueError: If errors and contexts have different lengths
        """
        if len(errors) != len(contexts):
            raise ValueError(
                f"errors and contexts must have same length, "
                f"got {len(errors)} and {len(contexts)}"
            )
        
        return [
            self.compute_adaptive_lr(error, context)
            for error, context in zip(errors, contexts)
        ]
