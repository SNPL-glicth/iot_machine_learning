"""Tests for PredictionEnginePortBridge — generic adapter PredictionEngine → PredictionPort.

Covers:
- as_port() convenience method on PredictionEngine
- PredictionEnginePortBridge wraps any engine correctly
- predict(window) bridges to predict(values, timestamps)
- predict_series(series) bridges to predict(values, timestamps)
- create_as_port() on EngineFactory
- Deprecation warnings on old adapters
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import pytest

from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.sensor_reading import SensorReading, SensorWindow
from iot_machine_learning.domain.entities.time_series import TimeSeries, TimePoint
from iot_machine_learning.domain.ports.prediction_port import PredictionPort
from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionEnginePortBridge,
    PredictionResult,
)


class _StubEngine(PredictionEngine):
    """Minimal engine for testing the bridge."""

    @property
    def name(self) -> str:
        return "stub_engine"

    def can_handle(self, n_points: int) -> bool:
        return n_points >= 2

    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        return PredictionResult(
            predicted_value=sum(values) / len(values),
            confidence=0.85,
            trend="up",
            metadata={"n": len(values), "has_ts": timestamps is not None},
        )

    def supports_uncertainty(self) -> bool:
        return True


def _make_window(values: List[float]) -> SensorWindow:
    readings = [
        SensorReading(sensor_id=42, value=v, timestamp=float(i))
        for i, v in enumerate(values)
    ]
    return SensorWindow(sensor_id=42, readings=readings)


# ── as_port() ─────────────────────────────────────────────────


class TestAsPort:
    def test_returns_bridge(self) -> None:
        engine = _StubEngine()
        port = engine.as_port()
        assert isinstance(port, PredictionEnginePortBridge)

    def test_bridge_is_prediction_port(self) -> None:
        engine = _StubEngine()
        port = engine.as_port()
        assert isinstance(port, PredictionPort)

    def test_name_delegated(self) -> None:
        port = _StubEngine().as_port()
        assert port.name == "stub_engine"

    def test_can_handle_delegated(self) -> None:
        port = _StubEngine().as_port()
        assert port.can_handle(5) is True
        assert port.can_handle(1) is False

    def test_engine_property(self) -> None:
        engine = _StubEngine()
        port = engine.as_port()
        assert port.engine is engine

    def test_supports_confidence_interval_delegated(self) -> None:
        port = _StubEngine().as_port()
        assert port.supports_confidence_interval() is True


# ── predict(window) ───────────────────────────────────────────


class TestPredictWindow:
    def test_returns_prediction(self) -> None:
        port = _StubEngine().as_port()
        window = _make_window([10.0, 20.0, 30.0])
        result = port.predict(window)
        assert isinstance(result, Prediction)

    def test_predicted_value(self) -> None:
        port = _StubEngine().as_port()
        window = _make_window([10.0, 20.0, 30.0])
        result = port.predict(window)
        assert result.predicted_value == pytest.approx(20.0)

    def test_confidence_mapped(self) -> None:
        port = _StubEngine().as_port()
        window = _make_window([10.0, 20.0])
        result = port.predict(window)
        assert result.confidence_score == pytest.approx(0.85)

    def test_trend_mapped(self) -> None:
        port = _StubEngine().as_port()
        window = _make_window([10.0, 20.0])
        result = port.predict(window)
        assert result.trend == "up"

    def test_series_id_from_window(self) -> None:
        port = _StubEngine().as_port()
        window = _make_window([10.0, 20.0])
        result = port.predict(window)
        assert result.series_id == "42"

    def test_engine_name_from_engine(self) -> None:
        port = _StubEngine().as_port()
        window = _make_window([10.0, 20.0])
        result = port.predict(window)
        assert result.engine_name == "stub_engine"

    def test_metadata_passed_through(self) -> None:
        port = _StubEngine().as_port()
        window = _make_window([10.0, 20.0])
        result = port.predict(window)
        assert result.metadata["n"] == 2

    def test_timestamps_forwarded(self) -> None:
        port = _StubEngine().as_port()
        window = _make_window([10.0, 20.0, 30.0])
        result = port.predict(window)
        assert result.metadata["has_ts"] is True


# ── predict_series(series) ────────────────────────────────────


class TestPredictSeries:
    def test_returns_prediction(self) -> None:
        port = _StubEngine().as_port()
        series = TimeSeries(
            series_id="temp_1",
            points=[TimePoint(t=0.0, v=10.0), TimePoint(t=1.0, v=20.0)],
        )
        result = port.predict_series(series)
        assert isinstance(result, Prediction)

    def test_series_id_from_series(self) -> None:
        port = _StubEngine().as_port()
        series = TimeSeries(
            series_id="room_temp",
            points=[TimePoint(t=0.0, v=10.0), TimePoint(t=1.0, v=20.0)],
        )
        result = port.predict_series(series)
        assert result.series_id == "room_temp"

    def test_values_extracted(self) -> None:
        port = _StubEngine().as_port()
        series = TimeSeries(
            series_id="s1",
            points=[TimePoint(t=0.0, v=5.0), TimePoint(t=1.0, v=15.0)],
        )
        result = port.predict_series(series)
        assert result.predicted_value == pytest.approx(10.0)


# ── EngineFactory.create_as_port() ────────────────────────────


class TestCreateAsPort:
    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        from iot_machine_learning.infrastructure.ml.engines.engine_factory import EngineFactory
        original = dict(EngineFactory._registry)
        yield
        EngineFactory._registry = original

    def test_create_as_port_returns_bridge(self) -> None:
        from iot_machine_learning.infrastructure.ml.engines.engine_factory import EngineFactory
        port = EngineFactory.create_as_port("baseline_moving_average")
        assert isinstance(port, PredictionEnginePortBridge)
        assert isinstance(port, PredictionPort)
        assert port.name == "baseline_moving_average"


# ── Deprecation warnings on old adapters ──────────────────────


class TestAdapterDeprecation:
    def test_taylor_adapter_warns(self) -> None:
        from iot_machine_learning.infrastructure.ml.engines.taylor_adapter import (
            TaylorPredictionAdapter,
        )
        with pytest.warns(DeprecationWarning, match="TaylorPredictionAdapter"):
            TaylorPredictionAdapter(order=2)

    def test_cognitive_adapter_warns(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration import (
            MetaCognitiveOrchestrator,
        )
        from iot_machine_learning.infrastructure.ml.cognitive.cognitive_adapter import (
            CognitivePredictionAdapter,
        )

        class _MinimalEngine(PredictionEngine):
            @property
            def name(self) -> str:
                return "mini"

            def can_handle(self, n_points: int) -> bool:
                return True

            def predict(self, values, timestamps=None):
                return PredictionResult(
                    predicted_value=0.0, confidence=0.5, trend="stable",
                )

        orch = MetaCognitiveOrchestrator(engines=[_MinimalEngine()])
        with pytest.warns(DeprecationWarning, match="CognitivePredictionAdapter"):
            CognitivePredictionAdapter(orch)
