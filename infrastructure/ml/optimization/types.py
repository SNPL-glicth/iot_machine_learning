"""Common types for optimization."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Any


@dataclass
class OptimizationResult:
    """Result of optimization."""
    params: np.ndarray
    objective_value: float
    success: bool
    n_iterations: int
    message: str = ""
    history: Optional[Dict[str, Any]] = None
    
    def get_history_value(self, key: str) -> Optional[Any]:
        """Get value from history."""
        if self.history is None:
            return None
        return self.history.get(key)


@dataclass
class OptimizerConfig:
    """Configuration for optimizers."""
    max_iterations: int = 1000
    tolerance: float = 1e-6
    verbose: bool = False
    early_stopping: bool = True
    early_stopping_patience: int = 10
