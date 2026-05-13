"""Ensemble decorrelation for highly correlated engines.

Adjusts weights to reduce impact of correlated engines.
Principio: Single Responsibility - solo ajusta pesos por correlación.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from core.parameters.numerical_constants import EPSILON
from core.ensemble.ensemble_correlation import CorrelationLevel, CorrelationResult

logger = logging.getLogger(__name__)


class EnsembleDecorrelator:
    """Ajusta pesos del ensemble basado en correlación."""

    def __init__(
        self,
        correlation_threshold: float = 0.7,
        weight_reduction_factor: float = 0.5,
    ) -> None:
        self._correlation_threshold = correlation_threshold
        self._weight_reduction_factor = weight_reduction_factor

    def adjust_weights_for_correlation(
        self,
        weights: Dict[str, float],
        correlation_matrix: np.ndarray,
        engine_names: list[str],
    ) -> Dict[str, float]:
        """
        Ajusta pesos para reducir impacto de engines altamente correlacionados.

        Args:
            weights: Dict con engine_name -> peso actual
            correlation_matrix: Matriz de correlación N x N
            engine_names: Lista de nombres de engines (orden de matriz)

        Returns:
            Dict con pesos ajustados (suma = 1.0)
        """
        if not weights or len(weights) < 2:
            return weights.copy()

        adjusted = weights.copy()
        n = len(engine_names)

        # Find highly correlated pairs
        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(correlation_matrix[i, j])
                name_i = engine_names[i]
                name_j = engine_names[j]

                if corr > self._correlation_threshold:
                    # Reduce weight of second engine in pair
                    if name_j in adjusted:
                        old_weight = adjusted[name_j]
                        reduction = old_weight * self._weight_reduction_factor
                        adjusted[name_j] = old_weight - reduction

                        logger.info(
                            "ensemble_decorrelation_applied",
                            extra={
                                "engine_1": name_i,
                                "engine_2": name_j,
                                "correlation": round(corr, 3),
                                "old_weight": round(old_weight, 4),
                                "new_weight": round(adjusted[name_j], 4),
                            },
                        )

        # Normalize to preserve sum = 1.0
        total = sum(adjusted.values())
        if total > EPSILON.COMPARISON:
            for name in adjusted:
                adjusted[name] = adjusted[name] / total
        else:
            # Fallback: equal weights
            equal_weight = 1.0 / len(adjusted)
            for name in adjusted:
                adjusted[name] = equal_weight

        return adjusted

    def apply_if_needed(
        self,
        weights: Dict[str, float],
        correlation_result: CorrelationResult,
    ) -> tuple[Dict[str, float], bool]:
        """
        Aplica decorrelation si se detecta correlación alta.

        Args:
            weights: Dict con engine_name -> peso actual
            correlation_result: Resultado del análisis de correlación

        Returns:
            Tuple de (pesos ajustados, si se aplicó decorrelation)
        """
        if correlation_result.max_correlation <= self._correlation_threshold:
            return weights.copy(), False

        # Map classification to engine names
        engine_names = correlation_result.engine_names
        correlation_matrix = correlation_result.matrix

        adjusted = self.adjust_weights_for_correlation(
            weights, correlation_matrix, engine_names
        )

        logger.warning(
            "ensemble_decorrelation_triggered",
            extra={
                "max_correlation": round(correlation_result.max_correlation, 3),
                "threshold": self._correlation_threshold,
                "recommendations": correlation_result.recommendations,
            },
        )

        return adjusted, True

    def compute_diversity_score(self, correlation_result: CorrelationResult) -> float:
        """
        Computa score de diversidad del ensemble (0-1).

        Args:
            correlation_result: Resultado del análisis de correlación

        Returns:
            Score de diversidad (1.0 = completamente diverso, 0.0 = completamente correlacionado)
        """
        if correlation_result.max_correlation == 0.0:
            return 1.0

        # Diversity = 1 - max_correlation
        diversity = max(0.0, 1.0 - correlation_result.max_correlation)
        return diversity
