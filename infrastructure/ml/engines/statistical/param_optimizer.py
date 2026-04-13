"""Statistical parameter optimizer — auto-tuning α and β.

Grid search over α (EMA smoothing) and β (trend smoothing) to minimize MAE.
Uses leave-one-out cross-validation on recent history.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class StatisticalParamOptimizer:
    """Grid search optimizer for EMA/Holt parameters.
    
    Finds optimal (α, β) pair that minimizes MAE on recent history.
    """
    
    ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    BETA_GRID = [0.05, 0.1, 0.15, 0.2, 0.3]
    MIN_SAMPLES = 10
    
    def optimize(
        self,
        values: List[float],
        max_samples: int = 20,
    ) -> Tuple[float, float, float]:
        """Find optimal (α, β) via grid search.
        
        Args:
            values: Historical values
            max_samples: Maximum samples to use for optimization
            
        Returns:
            Tuple of (best_alpha, best_beta, best_mae)
        """
        if len(values) < self.MIN_SAMPLES:
            logger.debug(
                "statistical_optimizer_insufficient_data",
                extra={"n_values": len(values), "min_required": self.MIN_SAMPLES},
            )
            return 0.3, 0.1, 999.0
        
        # Use last N values for optimization
        recent = values[-max_samples:] if len(values) > max_samples else values
        
        best_alpha = 0.3
        best_beta = 0.1
        best_mae = float('inf')
        
        for alpha in self.ALPHA_GRID:
            for beta in self.BETA_GRID:
                mae = self._evaluate_params(recent, alpha, beta)
                
                if mae < best_mae:
                    best_mae = mae
                    best_alpha = alpha
                    best_beta = beta
        
        logger.info(
            "statistical_params_optimized",
            extra={
                "alpha": best_alpha,
                "beta": best_beta,
                "mae": round(best_mae, 4),
                "n_samples": len(recent),
            },
        )
        
        return best_alpha, best_beta, best_mae
    
    def _evaluate_params(
        self,
        values: List[float],
        alpha: float,
        beta: float,
    ) -> float:
        """Evaluate (α, β) using leave-one-out MAE.
        
        Args:
            values: Values to evaluate on
            alpha: EMA smoothing factor
            beta: Trend smoothing factor
            
        Returns:
            Mean absolute error
        """
        if len(values) < 3:
            return 999.0
        
        errors = []
        
        # Leave-one-out: train on values[:i], predict values[i]
        for i in range(2, len(values)):
            train = values[:i]
            actual = values[i]
            
            # Holt's method
            level, trend = self._holt(train, alpha, beta)
            predicted = level + trend
            
            error = abs(predicted - actual)
            errors.append(error)
        
        return sum(errors) / len(errors) if errors else 999.0
    
    def _holt(
        self,
        values: List[float],
        alpha: float,
        beta: float,
    ) -> Tuple[float, float]:
        """Holt's double exponential smoothing.
        
        Args:
            values: Time series values
            alpha: Level smoothing factor
            beta: Trend smoothing factor
            
        Returns:
            Tuple of (level, trend) at last point
        """
        if len(values) < 2:
            return (values[0] if values else 0.0), 0.0
        
        level = values[0]
        trend = values[1] - values[0]
        
        for v in values[1:]:
            prev_level = level
            level = alpha * v + (1.0 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1.0 - beta) * trend
        
        return level, trend
