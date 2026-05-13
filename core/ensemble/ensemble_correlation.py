"""Ensemble correlation analysis for prediction engines.

Analyzes correlation between engine predictions to detect redundancy.
Principio: Single Responsibility - solo analiza correlaciones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CorrelationLevel(Enum):
    """Niveles de correlación entre engines."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class CorrelationResult:
    """Resultado del análisis de correlación."""
    matrix: np.ndarray
    engine_names: List[str]
    classification: Dict[str, CorrelationLevel]
    recommendations: List[str]
    max_correlation: float
    avg_correlation: float


class EngineCorrelationAnalyzer:
    """Analiza correlación entre predicciones de engines."""

    def __init__(self, low_threshold: float = 0.3, high_threshold: float = 0.7) -> None:
        self._low_threshold = low_threshold
        self._high_threshold = high_threshold

    def compute_correlation_matrix(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Computa matriz de correlación entre engines.

        Args:
            predictions: Dict con engine_name -> array de predicciones

        Returns:
            Matriz de correlación N x N (N = número de engines)
        """
        if not predictions:
            return np.array([]).reshape(0, 0)

        engine_names = sorted(predictions.keys())
        arrays = [predictions[name] for name in engine_names]
        n = len(engine_names)

        # Single engine: return 1x1 identity
        if n == 1:
            return np.eye(1)

        # Ensure all arrays have same length
        min_len = min(len(arr) for arr in arrays)
        if min_len < 2:
            logger.warning(
                "ensemble_correlation_insufficient_data",
                extra={"min_length": min_len},
            )
            return np.eye(n)

        truncated = [arr[:min_len] for arr in arrays]
        stacked = np.column_stack(truncated)

        # Compute correlation matrix (correlate columns, not rows)
        corr_matrix = np.corrcoef(stacked, rowvar=False)

        # Handle NaN (constant arrays)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        return corr_matrix

    def classify_correlation(self, matrix: np.ndarray) -> Dict[str, CorrelationLevel]:
        """
        Clasifica nivel de correlación por engine.

        Args:
            matrix: Matriz de correlación N x N

        Returns:
            Dict con engine_name -> CorrelationLevel
        """
        n = matrix.shape[0]
        classification = {}

        for i in range(n):
            # Get correlation with other engines (excluding self)
            others = [matrix[i, j] for j in range(n) if i != j]

            if not others:
                classification[f"engine_{i}"] = CorrelationLevel.LOW
                continue

            avg_corr = np.mean(others)

            if avg_corr < self._low_threshold:
                classification[f"engine_{i}"] = CorrelationLevel.LOW
            elif avg_corr < self._high_threshold:
                classification[f"engine_{i}"] = CorrelationLevel.MODERATE
            else:
                classification[f"engine_{i}"] = CorrelationLevel.HIGH

        return classification

    def analyze(self, predictions: Dict[str, np.ndarray], engine_names: Optional[List[str]] = None) -> CorrelationResult:
        """
        Analiza correlación completa entre engines.

        Args:
            predictions: Dict con engine_name -> array de predicciones
            engine_names: Lista de nombres de engines (opcional)

        Returns:
            CorrelationResult con matriz, clasificación y recomendaciones
        """
        if engine_names is None:
            engine_names = sorted(predictions.keys())

        matrix = self.compute_correlation_matrix(predictions)
        classification = self.classify_correlation(matrix)

        # Compute statistics
        n = matrix.shape[0]
        if n > 1:
            upper_tri = matrix[np.triu_indices(n, k=1)]
            max_corr = np.max(np.abs(upper_tri)) if len(upper_tri) > 0 else 0.0
            avg_corr = np.mean(upper_tri) if len(upper_tri) > 0 else 0.0
        else:
            max_corr = 0.0
            avg_corr = 0.0

        # Generate recommendations
        recommendations = []
        if max_corr > self._high_threshold:
            recommendations.append(f"HIGH correlation detected (max={max_corr:.3f}): consider decorrelation")
        elif max_corr > self._low_threshold:
            recommendations.append(f"MODERATE correlation detected (max={max_corr:.3f}): monitor closely")
        else:
            recommendations.append(f"LOW correlation (max={max_corr:.3f}): ensemble is diverse")

        return CorrelationResult(
            matrix=matrix,
            engine_names=engine_names,
            classification=classification,
            recommendations=recommendations,
            max_correlation=max_corr,
            avg_correlation=avg_corr,
        )
