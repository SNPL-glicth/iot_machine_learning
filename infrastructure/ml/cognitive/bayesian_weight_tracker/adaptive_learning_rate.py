"""Adaptive Learning Rate calculator for contextual weight tracking.

Stateful wrapper around lr_calculator pure functions.
Uses Bayesian inference — not neuroplasticity or RL.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import List, Optional, Tuple

from iot_machine_learning.domain.entities.plasticity.plasticity_context import PlasticityContext
from .lr_calculator import compute_learning_rate

logger = logging.getLogger(__name__)


class AdaptiveLearningRate:
    """Calculates adaptive learning rates with momentum tracking.
    
    Delegates actual calculation to lr_calculator.
    Manages velocity state for momentum.
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
        # Validate
        if not 0 < lr_min < lr_max:
            raise ValueError(f"Must have 0 < lr_min < lr_max, got {lr_min}, {lr_max}")
        if not 0 < base_lr <= lr_max:
            raise ValueError(f"base_lr must be in (0, lr_max], got {base_lr}")
        if error_scale <= 0:
            raise ValueError(f"error_scale must be > 0, got {error_scale}")
        if not 0 < noise_penalty <= 1:
            raise ValueError(f"noise_penalty must be in (0, 1], got {noise_penalty}")
        
        self.base_lr = base_lr
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.error_scale = error_scale
        self.noise_penalty = noise_penalty
        self.failure_boost = failure_boost
        self.failure_threshold = failure_threshold
        self.momentum = momentum
        self.momentum_decay = momentum_decay
        
        # State: velocity tracking per (series_id, engine_name)
        self._velocity: OrderedDict[Tuple[str, str], float] = OrderedDict()
        self._max_velocity_entries = 10000
    
    def compute_adaptive_lr(
        self,
        error: float,
        context: PlasticityContext,
        engine_name: Optional[str] = None,
        series_id: Optional[str] = None,
    ) -> float:
        """Compute adaptive learning rate with momentum."""
        # Base calculation (pure function)
        lr = compute_learning_rate(
            error=error,
            context=context,
            base_lr=self.base_lr,
            lr_min=self.lr_min,
            lr_max=self.lr_max,
            error_scale=self.error_scale,
            noise_penalty=self.noise_penalty,
            failure_boost=self.failure_boost,
            failure_threshold=self.failure_threshold,
        )
        
        # Apply momentum if configured
        if engine_name is not None and self.momentum > 0:
            key = (series_id, engine_name) if series_id else (engine_name, engine_name)
            prev_velocity = self._velocity.get(key, self.base_lr)
            prev_velocity *= self.momentum_decay
            lr = self.momentum * prev_velocity + (1.0 - self.momentum) * lr
            
            # LRU for velocity
            if len(self._velocity) >= self._max_velocity_entries:
                self._velocity.popitem(last=False)
            
            self._velocity[key] = lr
            if key in self._velocity:
                self._velocity.move_to_end(key)
        
        return lr
    
    def compute_batch_lr(
        self,
        errors: List[float],
        contexts: List[PlasticityContext],
    ) -> List[float]:
        """Compute adaptive learning rates for a batch."""
        if len(errors) != len(contexts):
            raise ValueError(
                f"errors and contexts must have same length, "
                f"got {len(errors)} and {len(contexts)}"
            )
        
        return [
            self.compute_adaptive_lr(error, context)
            for error, context in zip(errors, contexts)
        ]
