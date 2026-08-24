"""Parameter re-optimization logic."""

from __future__ import annotations

import logging
from typing import List, Optional

from iot_machine_learning.infrastructure.ml.cognitive.hyperparameters import HyperparameterAdaptor
from .smoothing import ema

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """Handles deferred re-optimization of alpha/beta parameters."""
    
    def __init__(
        self,
        adaptor: Optional[HyperparameterAdaptor],
        series_id: Optional[str],
        engine_name: str,
        min_improvement: float = 0.05,
    ):
        self._adaptor = adaptor
        self._series_id = series_id
        self._engine_name = engine_name
        self._min_improvement = min_improvement
    
    def optimize(
        self,
        prediction_history: List[float],
        current_alpha: float,
        current_beta: float,
        current_mae: float,
    ) -> tuple[float, float, float, bool]:
        """
        Re-optimize parameters.
        
        Returns: (new_alpha, new_beta, new_mae, improved)
        """
        if not self._series_id:
            return current_alpha, current_beta, current_mae, False
        
        try:
            from iot_machine_learning.infrastructure.ml.engines.statistical.param_optimizer import (
                StatisticalParamOptimizer,
            )
            
            optimizer = StatisticalParamOptimizer()
            values = list(prediction_history)
            
            new_alpha, new_beta, new_mae = optimizer.optimize(values)
            
            improvement = (current_mae - new_mae) / current_mae if current_mae > 0 else 0.0
            improved = improvement > self._min_improvement
            
            if improved:
                logger.info(
                    "statistical_params_reoptimized",
                    extra={
                        "series_id": self._series_id,
                        "new_alpha": new_alpha,
                        "new_beta": new_beta,
                        "new_mae": round(new_mae, 4),
                        "improvement_pct": round(improvement * 100, 2),
                    },
                )
            
            return new_alpha, new_beta, new_mae, improved
            
        except Exception as exc:
            logger.warning(
                "statistical_reoptimization_failed",
                extra={"series_id": self._series_id, "error": str(exc)},
            )
            return current_alpha, current_beta, current_mae, False