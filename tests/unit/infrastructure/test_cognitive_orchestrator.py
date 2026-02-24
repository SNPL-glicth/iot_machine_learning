"""Tests for cognitive/orchestrator.py — MetaCognitiveOrchestrator.

Covers:
1. Single engine orchestration
2. Multi-engine weighted fusion
3. Inhibition integration (unstable engine suppressed)
4. Plasticity learning (weights adapt after record_actual)
5. Fallback when all engines fail
6. MetaDiagnostic completeness
7. Engine with diagnostic metadata extraction
"""

from __future__ import annotations

import math
import warnings

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.orchestration import (
    MetaCognitiveOrchestrator,
)
from iot_machine_learning.infrastructure.ml.cognitive.inhibition import (
    InhibitionConfig,
)
from iot_machine_learning.infrastructure.ml.engines.taylor_engine import (
    TaylorPredictionEngine,
)
from iot_machine_learning.infrastructure.ml.engines.statistical_engine import (
    StatisticalPredictionEngine,
)
from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
)
from typing import List, Optional


# -- Stub engine for controlled testing ------------------------------------

class StubEngine(PredictionEngine):
    def __init__(
        self, name: str, value: float, confidence: float = 0.8,
        trend: str = "stable", stability: float = 0.0,
        fit_error: float = 0.0, min_points: int = 1,
        should_fail: bool = False,
    ):
        self._name = name
        self._value = value
        self._confidence = confidence
        self._trend = trend
        self._stability = stability
        self._fit_error = fit_error
        self._min_points = min_points
        self._should_fail = should_fail

    @property
    def name(self) -> str:
        return self._name

    def can_handle(self, n_points: int) -> bool:
        return n_points >= self._min_points

    def predict(
        self, values: List[float], timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        if self._should_fail:
            raise RuntimeError(f"{self._name} failed")
        return PredictionResult(
            predicted_value=self._value,
            confidence=self._confidence,
            trend=self._trend,
            metadata={
                "diagnostic": {
                    "stability_indicator": self._stability,
                    "local_fit_error": self._fit_error,
                    "method": "stub",
                },
            },
        )


# -- Tests -----------------------------------------------------------------

class TestSingleEngine:

    def test_single_engine_passthrough(self) -> None:
        eng = StubEngine("stub_a", value=42.0, confidence=0.9)
        orch = MetaCognitiveOrchestrator([eng], enable_plasticity=False)
        result = orch.predict([1.0, 2.0, 3.0])
        assert result.predicted_value == pytest.approx(42.0)
        assert result.confidence == pytest.approx(0.9)

    def test_diagnostic_present(self) -> None:
        eng = StubEngine("stub_a", value=42.0)
        orch = MetaCognitiveOrchestrator([eng], enable_plasticity=False)
        orch.predict([1.0, 2.0, 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            diag = orch.last_diagnostic
        assert diag is not None
        assert diag.selected_engine == "stub_a"
        assert diag.fusion_method == "single_engine"


class TestMultiEngineFusion:

    def test_equal_weight_average(self) -> None:
        a = StubEngine("a", value=10.0, confidence=0.8)
        b = StubEngine("b", value=20.0, confidence=0.6)
        orch = MetaCognitiveOrchestrator([a, b], enable_plasticity=False)
        result = orch.predict([1.0, 2.0, 3.0])
        assert result.predicted_value == pytest.approx(15.0)
        assert result.confidence == pytest.approx(0.7)

    def test_custom_initial_weights(self) -> None:
        a = StubEngine("a", value=10.0, confidence=0.8)
        b = StubEngine("b", value=20.0, confidence=0.6)
        orch = MetaCognitiveOrchestrator(
            [a, b],
            initial_weights={"a": 0.75, "b": 0.25},
            enable_plasticity=False,
        )
        result = orch.predict([1.0, 2.0, 3.0])
        assert result.predicted_value == pytest.approx(12.5)

    def test_trend_majority_vote(self) -> None:
        a = StubEngine("a", value=10.0, trend="up")
        b = StubEngine("b", value=20.0, trend="up")
        c = StubEngine("c", value=15.0, trend="down")
        orch = MetaCognitiveOrchestrator(
            [a, b, c], enable_plasticity=False)
        result = orch.predict([1.0, 2.0, 3.0])
        assert result.trend == "up"


class TestInhibitionIntegration:

    def test_unstable_engine_suppressed(self) -> None:
        stable = StubEngine("stable", value=10.0, stability=0.1)
        unstable = StubEngine("unstable", value=100.0, stability=0.95)
        orch = MetaCognitiveOrchestrator(
            [stable, unstable], enable_plasticity=False)
        result = orch.predict([1.0, 2.0, 3.0])
        # Fused value should be closer to stable engine
        assert result.predicted_value < 60.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            diag = orch.last_diagnostic
        assert diag is not None
        inh = {s.engine_name: s for s in diag.inhibition_states}
        assert inh["unstable"].suppression_factor > 0.0
        assert inh["stable"].suppression_factor == pytest.approx(0.0)

    def test_high_fit_error_suppressed(self) -> None:
        good = StubEngine("good", value=10.0, fit_error=0.1)
        bad = StubEngine("bad", value=100.0, fit_error=50.0)
        orch = MetaCognitiveOrchestrator(
            [good, bad], enable_plasticity=False)
        result = orch.predict([1.0, 2.0, 3.0])
        assert result.predicted_value < 60.0


class TestPlasticityIntegration:

    def test_weights_adapt_after_record_actual(self) -> None:
        a = StubEngine("a", value=10.0)
        b = StubEngine("b", value=20.0)
        orch = MetaCognitiveOrchestrator([a, b], enable_plasticity=True)

        # Run several cycles where actual is always close to "a"
        for _ in range(20):
            orch.predict([1.0, 2.0, 3.0])
            orch.record_actual(10.5)  # close to engine "a"

        # After learning, "a" should have higher weight
        result = orch.predict([1.0, 2.0, 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            diag = orch.last_diagnostic
        assert diag is not None
        assert diag.final_weights["a"] > diag.final_weights["b"]


class TestFallback:

    def test_all_engines_fail(self) -> None:
        a = StubEngine("a", value=10.0, should_fail=True)
        b = StubEngine("b", value=20.0, should_fail=True)
        orch = MetaCognitiveOrchestrator(
            [a, b], enable_plasticity=False)
        result = orch.predict([5.0, 6.0, 7.0])
        # Fallback: mean of last 3
        assert result.predicted_value == pytest.approx(6.0)
        assert result.confidence == pytest.approx(0.2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            diag = orch.last_diagnostic
        assert diag is not None
        assert diag.fallback_reason == "no_valid_perceptions"

    def test_insufficient_data_engine_skipped(self) -> None:
        a = StubEngine("a", value=10.0, min_points=100)
        b = StubEngine("b", value=20.0, min_points=1)
        orch = MetaCognitiveOrchestrator(
            [a, b], enable_plasticity=False)
        result = orch.predict([1.0, 2.0, 3.0])
        # Only "b" runs
        assert result.predicted_value == pytest.approx(20.0)


class TestMetaDiagnosticCompleteness:

    def test_diagnostic_has_all_fields(self) -> None:
        a = StubEngine("a", value=10.0)
        b = StubEngine("b", value=20.0)
        orch = MetaCognitiveOrchestrator([a, b], enable_plasticity=False)
        orch.predict([1.0, 2.0, 3.0, 4.0, 5.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            diag = orch.last_diagnostic
        assert diag is not None

        d = diag.to_dict()
        assert "signal_profile" in d
        assert "perceptions" in d
        assert "inhibition_states" in d
        assert "final_weights" in d
        assert "selected_engine" in d
        assert "selection_reason" in d
        assert "fusion_method" in d

    def test_metadata_in_prediction_result(self) -> None:
        a = StubEngine("a", value=10.0)
        orch = MetaCognitiveOrchestrator([a], enable_plasticity=False)
        result = orch.predict([1.0, 2.0, 3.0])
        assert "cognitive_diagnostic" in result.metadata


class TestRealEngineIntegration:

    def test_taylor_plus_statistical(self) -> None:
        taylor = TaylorPredictionEngine(order=2, horizon=1)
        stat = StatisticalPredictionEngine(alpha=0.3, beta=0.1, horizon=1)
        orch = MetaCognitiveOrchestrator(
            [taylor, stat], enable_plasticity=False)

        values = [float(i) for i in range(20)]
        result = orch.predict(values)

        assert math.isfinite(result.predicted_value)
        assert 0.0 <= result.confidence <= 1.0
        assert result.trend in ("up", "down", "stable")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            diag = orch.last_diagnostic
        assert diag is not None
        assert len(diag.perceptions) == 2
        assert diag.signal_profile.n_points == 20

    def test_taylor_plus_statistical_noisy(self) -> None:
        import random
        random.seed(42)
        taylor = TaylorPredictionEngine(order=2, horizon=1)
        stat = StatisticalPredictionEngine(alpha=0.3, beta=0.1, horizon=1)
        orch = MetaCognitiveOrchestrator(
            [taylor, stat], enable_plasticity=False)

        values = [20.0 + random.gauss(0, 5.0) for _ in range(30)]
        result = orch.predict(values)
        assert math.isfinite(result.predicted_value)


class TestConstructorValidation:

    def test_empty_engines_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one engine"):
            MetaCognitiveOrchestrator([])

    def test_name_property(self) -> None:
        eng = StubEngine("a", value=1.0)
        orch = MetaCognitiveOrchestrator([eng])
        assert orch.name == "meta_cognitive_orchestrator"

    def test_can_handle_delegates(self) -> None:
        a = StubEngine("a", value=1.0, min_points=10)
        b = StubEngine("b", value=2.0, min_points=5)
        orch = MetaCognitiveOrchestrator([a, b])
        assert orch.can_handle(7) is True
        assert orch.can_handle(3) is False
