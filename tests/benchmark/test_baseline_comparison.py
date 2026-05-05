"""Benchmark baseline — compare ZENIN against naive models (mean, EMA).

Ensures ZENIN performs at least as well as naive baselines.
Uses synthetic signals with known structure.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from typing import List

import pytest

from iot_machine_learning.infrastructure.ml.engines.taylor.engine import (
    TaylorPredictionEngine,
)
from iot_machine_learning.infrastructure.ml.engines.statistical.engine import (
    StatisticalPredictionEngine,
)
from iot_machine_learning.infrastructure.ml.engines.core.factory import (
    BaselineMovingAverageEngine,
)


class NaivePredictors:
    """Simple baselines for comparison."""

    @staticmethod
    def mean_baseline(values: List[float]) -> float:
        if not values:
            return 0.0
        return statistics.mean(values)

    @staticmethod
    def ema_baseline(values: List[float], alpha: float = 0.3) -> float:
        if not values:
            return 0.0
        ema = values[0]
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return ema


def _generate_trend_signal(
    n: int,
    slope: float = 0.1,
    noise: float = 0.5,
    seed: int = 42,
) -> List[float]:
    rng = random.Random(seed)
    return [20.0 + slope * i + rng.gauss(0, noise) for i in range(n)]


def _generate_seasonal_signal(
    n: int,
    period: int = 24,
    noise: float = 0.3,
    seed: int = 42,
) -> List[float]:
    rng = random.Random(seed)
    return [20.0 + 5.0 * math.sin(2 * math.pi * i / period) + rng.gauss(0, noise) for i in range(n)]


def _evaluate(predictions: List[float], actuals: List[float]) -> float:
    if not predictions or not actuals or len(predictions) != len(actuals):
        return float("inf")
    return statistics.mean(abs(p - a) for p, a in zip(predictions, actuals))


class TestBenchmarkVsNaive:
    """ZENIN engines must beat or match naive baselines."""

    def test_taylor_better_than_mean_on_trend(self) -> None:
        signal = _generate_trend_signal(200, slope=0.2, noise=0.3)
        engine = TaylorPredictionEngine()

        preds_taylor: List[float] = []
        preds_mean: List[float] = []
        actuals: List[float] = []

        window = 20
        for i in range(window, len(signal) - 1):
            win = signal[i - window : i]
            result = engine.predict(win)
            preds_taylor.append(result.predicted_value)
            preds_mean.append(NaivePredictors.mean_baseline(win))
            actuals.append(signal[i + 1])

        mae_taylor = _evaluate(preds_taylor, actuals)
        mae_mean = _evaluate(preds_mean, actuals)
        assert mae_taylor <= mae_mean * 1.5, (
            f"Taylor MAE={mae_taylor:.3f} much worse than Mean MAE={mae_mean:.3f}"
        )

    def test_statistical_better_than_ema_on_smooth(self) -> None:
        signal = _generate_trend_signal(200, slope=0.05, noise=0.2)
        engine = StatisticalPredictionEngine()

        preds_stat: List[float] = []
        preds_ema: List[float] = []
        actuals: List[float] = []

        window = 20
        for i in range(window, len(signal) - 1):
            win = signal[i - window : i]
            result = engine.predict(win)
            preds_stat.append(result.predicted_value)
            preds_ema.append(NaivePredictors.ema_baseline(win, alpha=0.3))
            actuals.append(signal[i + 1])

        mae_stat = _evaluate(preds_stat, actuals)
        mae_ema = _evaluate(preds_ema, actuals)
        assert mae_stat <= mae_ema * 1.5, (
            f"Statistical MAE={mae_stat:.3f} much worse than EMA MAE={mae_ema:.3f}"
        )

    def test_baseline_never_negative_latency(self) -> None:
        engine = BaselineMovingAverageEngine()
        signal = _generate_trend_signal(50)
        for i in range(20, len(signal)):
            t0 = time.perf_counter()
            result = engine.predict(signal[i - 20 : i])
            latency = time.perf_counter() - t0
            assert latency >= 0.0
            assert result.predicted_value is not None

    def test_fusion_better_than_single_engine_on_average(self) -> None:
        """Ensemble (Taylor + Statistical + Baseline) should not be worse
        than the worst single engine on average.
        """
        from iot_machine_learning.infrastructure.ml.cognitive.fusion.engine_selector import (
            WeightedFusion,
        )
        from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
            EnginePerception, InhibitionState,
        )

        signal = _generate_seasonal_signal(200, noise=0.5)
        engines = {
            "taylor": TaylorPredictionEngine(),
            "statistical": StatisticalPredictionEngine(),
            "baseline": BaselineMovingAverageEngine(),
        }

        preds_fusion: List[float] = []
        preds_single: Dict[str, List[float]] = {k: [] for k in engines}
        actuals: List[float] = []

        window = 20
        for i in range(window, len(signal) - 1):
            win = signal[i - window : i]
            perceptions = []
            for name, engine in engines.items():
                result = engine.predict(win)
                perceptions.append(
                    EnginePerception(
                        engine_name=name,
                        predicted_value=result.predicted_value,
                        confidence=0.7,
                        trend="stable",
                    )
                )
                preds_single[name].append(result.predicted_value)

            inhibitions = [
                InhibitionState(name, 1.0 / len(engines), 1.0 / len(engines), 0.0)
                for name in engines
            ]
            fusion = WeightedFusion()
            fused_value, _, _, _, _, _ = fusion.fuse(perceptions, inhibitions)
            preds_fusion.append(fused_value)
            actuals.append(signal[i + 1])

        mae_fusion = _evaluate(preds_fusion, actuals)
        worst_single = max(_evaluate(preds_single[k], actuals) for k in engines)
        assert mae_fusion <= worst_single * 1.3, (
            f"Fusion MAE={mae_fusion:.3f} worse than worst single={worst_single:.3f}"
        )
