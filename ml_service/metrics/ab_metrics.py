"""Pure metric computation functions for A/B testing.

No state, no I/O, no threading — extracted from ABTester so the
computation logic can be tested and reused independently.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .ab_testing import ABTestResult

# Mínimo de muestras para calcular resultados significativos
_MIN_SAMPLES: int = 10

# Margen para declarar ganador (5%)
_WINNER_MARGIN: float = 0.05


def compute_mae(actual: List[float], predicted: List[float]) -> float:
    """Compute Mean Absolute Error.

    Args:
        actual: List of actual observed values.
        predicted: List of predicted values (same length as actual).

    Returns:
        MAE value.
    """
    n = len(actual)
    if n == 0:
        return 0.0
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / n


def compute_rmse(actual: List[float], predicted: List[float]) -> float:
    """Compute Root Mean Squared Error.

    Args:
        actual: List of actual observed values.
        predicted: List of predicted values (same length as actual).

    Returns:
        RMSE value.
    """
    n = len(actual)
    if n == 0:
        return 0.0
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n)


def determine_winner(
    baseline_mae: float,
    taylor_mae: float,
    margin: float = _WINNER_MARGIN,
) -> str:
    """Determine the winner given two MAE values and a margin.

    Args:
        baseline_mae: MAE of the baseline engine.
        taylor_mae: MAE of the Taylor engine.
        margin: Minimum relative improvement to declare a winner.

    Returns:
        ``"taylor"``, ``"baseline"``, or ``"tie"``.
    """
    if baseline_mae > 0 and taylor_mae < baseline_mae * (1.0 - margin):
        return "taylor"
    if taylor_mae > 0 and baseline_mae < taylor_mae * (1.0 - margin):
        return "baseline"
    return "tie"


def compute_improvement_pct(baseline_mae: float, taylor_mae: float) -> float:
    """Compute percentage improvement of Taylor over baseline.

    Positive = Taylor better. Negative = Taylor worse.

    Args:
        baseline_mae: MAE of the baseline engine.
        taylor_mae: MAE of the Taylor engine.

    Returns:
        Improvement percentage (0.0 if baseline_mae is near zero).
    """
    if baseline_mae > 1e-12:
        return ((baseline_mae - taylor_mae) / baseline_mae) * 100.0
    return 0.0


def compute_ab_result(
    sensor_id: int,
    actual: List[float],
    baseline_preds: List[float],
    taylor_preds: List[float],
    min_samples: int = _MIN_SAMPLES,
) -> Optional[ABTestResult]:
    """Compute a full ABTestResult from raw prediction lists.

    Args:
        sensor_id: Sensor identifier.
        actual: List of actual observed values.
        baseline_preds: Baseline predictions (same length as actual).
        taylor_preds: Taylor predictions (same length as actual).
        min_samples: Minimum required samples; returns None if not met.

    Returns:
        ``ABTestResult`` or ``None`` if insufficient data.
    """
    n = len(actual)
    if n < min_samples:
        return None

    baseline_mae = compute_mae(actual, baseline_preds)
    taylor_mae = compute_mae(actual, taylor_preds)
    baseline_rmse = compute_rmse(actual, baseline_preds)
    taylor_rmse = compute_rmse(actual, taylor_preds)

    winner = determine_winner(baseline_mae, taylor_mae)
    improvement_pct = compute_improvement_pct(baseline_mae, taylor_mae)
    confidence = min(1.0, n / 100.0)

    return ABTestResult(
        sensor_id=sensor_id,
        baseline_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        taylor_mae=taylor_mae,
        taylor_rmse=taylor_rmse,
        winner=winner,
        confidence=confidence,
        n_samples=n,
        improvement_pct=improvement_pct,
    )


def aggregate_summary(results: List[ABTestResult]) -> dict:
    """Build a global summary dict from a list of ABTestResult.

    Args:
        results: List of per-sensor results.

    Returns:
        Summary dict with aggregated statistics.
    """
    if not results:
        return {"status": "no_data"}

    taylor_wins = sum(1 for r in results if r.winner == "taylor")
    baseline_wins = sum(1 for r in results if r.winner == "baseline")
    ties = sum(1 for r in results if r.winner == "tie")
    n = len(results)

    return {
        "status": "active",
        "total_sensors": n,
        "taylor_wins": taylor_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "avg_taylor_mae": round(sum(r.taylor_mae for r in results) / n, 6),
        "avg_baseline_mae": round(sum(r.baseline_mae for r in results) / n, 6),
        "avg_improvement_pct": round(sum(r.improvement_pct for r in results) / n, 2),
        "total_samples": sum(r.n_samples for r in results),
    }
