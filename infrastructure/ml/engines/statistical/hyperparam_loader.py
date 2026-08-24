"""Hyperparameter loading and persistence via HyperparameterAdaptor."""

from __future__ import annotations

import logging
from typing import Optional

from iot_machine_learning.infrastructure.ml.cognitive.hyperparameters import HyperparameterAdaptor

logger = logging.getLogger(__name__)


class HyperparameterLoader:
    """Loads and saves engine hyperparameters via HyperparameterAdaptor (Redis)."""
    
    def __init__(
        self,
        adaptor: Optional[HyperparameterAdaptor],
        series_id: Optional[str],
        engine_name: str,
    ):
        self._adaptor = adaptor
        self._series_id = series_id
        self._engine_name = engine_name
    
    def load(
        self,
        current_alpha: float,
        current_beta: float,
        current_mae: float,
        window_size: int,
    ) -> tuple[float, float, float]:
        """Load hyperparameters from Redis. Returns (alpha, beta, mae)."""
        if self._adaptor is None or not self._series_id:
            return current_alpha, current_beta, current_mae
        
        params = self._adaptor.load(self._series_id, self._engine_name)
        if not params:
            return current_alpha, current_beta, current_mae
        
        alpha = current_alpha
        beta = current_beta
        mae = current_mae
        
        alpha_raw = params.get("alpha")
        if alpha_raw is not None:
            try:
                a = float(alpha_raw)
                if 0.0 < a <= 1.0:
                    alpha = a
            except (TypeError, ValueError):
                pass
        
        beta_raw = params.get("beta")
        if beta_raw is not None:
            try:
                b = float(beta_raw)
                if 0.0 <= b <= 1.0:
                    beta = b
            except (TypeError, ValueError):
                pass
        
        mae_raw = params.get("mae")
        if mae_raw is not None:
            try:
                mae = float(mae_raw)
            except (TypeError, ValueError):
                pass
        
        logger.debug(
            "hyperparams_loaded series=%s engine=%s alpha=%.4f beta=%.4f window=%d",
            self._series_id,
            self._engine_name,
            alpha,
            beta,
            window_size,
        )
        
        return alpha, beta, mae
    
    def save(self, alpha: float, beta: float, mae: float) -> None:
        """Save hyperparameters to Redis."""
        if self._adaptor is None or not self._series_id:
            return
        
        self._adaptor.save(
            self._series_id,
            self._engine_name,
            {
                "alpha": alpha,
                "beta": beta,
                "mae": mae,
            },
        )