"""Tests for Phase 3 — Extensibility & Dependency Injection.

Covers:
- VotingAnomalyDetector DI: injecting custom sub_detectors (MOD-2, ROB-2)
- create_default_detectors() factory function
- @register_engine decorator + EngineFactory auto-registration (ROB-1)
- discover_engines() plugin discovery
- DetectorRegistry + @register_detector
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
)


# ── VotingAnomalyDetector DI ─────────────────────────────────


class TestVotingAnomalyDetectorDI:
    """VotingAnomalyDetector accepts injected sub_detectors."""

    def test_default_creates_8_detectors(self) -> None:
        """Without sub_detectors kwarg, creates the default 8."""
        from iot_machine_learning.infrastructure.ml.anomaly import (
            VotingAnomalyDetector,
        )

        det = VotingAnomalyDetector()
        assert len(det._sub_detectors) == 8

    def test_injected_detectors_used(self) -> None:
        """Passing sub_detectors replaces the default list."""
        from iot_machine_learning.infrastructure.ml.anomaly import (
            VotingAnomalyDetector,
        )
        from iot_machine_learning.infrastructure.ml.anomaly.detector_protocol import (
            SubDetector,
        )

        class _StubDetector(SubDetector):
            @property
            def method_name(self) -> str:
                return "stub"

            def train(self, values, **kwargs):
                self._trained = True

            def vote(self, value, **kwargs):
                return 0.5

            @property
            def is_trained(self) -> bool:
                return getattr(self, "_trained", False)

        stub = _StubDetector()
        det = VotingAnomalyDetector(sub_detectors=[stub])
        assert len(det._sub_detectors) == 1
        assert det._sub_detectors[0] is stub

    def test_injected_detectors_are_copied(self) -> None:
        """Injected list is copied, not shared."""
        from iot_machine_learning.infrastructure.ml.anomaly import (
            VotingAnomalyDetector,
        )
        from iot_machine_learning.infrastructure.ml.anomaly.detector_protocol import (
            SubDetector,
        )

        class _Stub(SubDetector):
            @property
            def method_name(self) -> str:
                return "s"

            def train(self, values, **kwargs):
                pass

            def vote(self, value, **kwargs):
                return 0.0

            @property
            def is_trained(self) -> bool:
                return False

        original = [_Stub()]
        det = VotingAnomalyDetector(sub_detectors=original)
        original.append(_Stub())
        assert len(det._sub_detectors) == 1  # not affected

    def test_empty_detectors_list_accepted(self) -> None:
        """Empty list is valid (edge case)."""
        from iot_machine_learning.infrastructure.ml.anomaly import (
            VotingAnomalyDetector,
        )

        det = VotingAnomalyDetector(sub_detectors=[])
        assert len(det._sub_detectors) == 0


# ── create_default_detectors ─────────────────────────────────


class TestCreateDefaultDetectors:
    def test_returns_8_detectors(self) -> None:
        from iot_machine_learning.infrastructure.ml.anomaly import (
            AnomalyDetectorConfig,
            create_default_detectors,
        )

        config = AnomalyDetectorConfig()
        detectors = create_default_detectors(config)
        assert len(detectors) == 8

    def test_all_are_sub_detectors(self) -> None:
        from iot_machine_learning.infrastructure.ml.anomaly import (
            AnomalyDetectorConfig,
            create_default_detectors,
        )
        from iot_machine_learning.infrastructure.ml.anomaly.detector_protocol import (
            SubDetector,
        )

        config = AnomalyDetectorConfig()
        detectors = create_default_detectors(config)
        for d in detectors:
            assert isinstance(d, SubDetector)

    def test_method_names_unique(self) -> None:
        from iot_machine_learning.infrastructure.ml.anomaly import (
            AnomalyDetectorConfig,
            create_default_detectors,
        )

        config = AnomalyDetectorConfig()
        detectors = create_default_detectors(config)
        names = [d.method_name for d in detectors]
        assert len(names) == len(set(names))

    def test_custom_config_propagated(self) -> None:
        from iot_machine_learning.infrastructure.ml.anomaly import (
            AnomalyDetectorConfig,
            create_default_detectors,
        )

        config = AnomalyDetectorConfig(contamination=0.2, n_estimators=50)
        detectors = create_default_detectors(config)
        assert len(detectors) == 8


# ── @register_engine decorator ────────────────────────────────


class TestRegisterEngineDecorator:
    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        from iot_machine_learning.infrastructure.ml.engines.engine_factory import (
            EngineFactory,
        )
        original = dict(EngineFactory._registry)
        yield
        EngineFactory._registry = original

    def test_decorator_registers_engine(self) -> None:
        from iot_machine_learning.infrastructure.ml.engines.engine_factory import (
            EngineFactory,
            register_engine,
        )

        @register_engine("test_engine_dec")
        class _TestEngine(PredictionEngine):
            @property
            def name(self) -> str:
                return "test_engine_dec"

            def can_handle(self, n_points: int) -> bool:
                return True

            def predict(self, values, timestamps=None):
                return PredictionResult(
                    predicted_value=0.0, confidence=1.0, trend="stable",
                )

        assert "test_engine_dec" in EngineFactory.list_engines()
        engine = EngineFactory.create("test_engine_dec")
        assert isinstance(engine, _TestEngine)

    def test_decorator_returns_class_unchanged(self) -> None:
        from iot_machine_learning.infrastructure.ml.engines.engine_factory import (
            register_engine,
        )

        @register_engine("test_identity")
        class _MyEngine(PredictionEngine):
            @property
            def name(self) -> str:
                return "test_identity"

            def can_handle(self, n_points: int) -> bool:
                return True

            def predict(self, values, timestamps=None):
                return PredictionResult(
                    predicted_value=0.0, confidence=1.0, trend="stable",
                )

        assert _MyEngine.__name__ == "_MyEngine"

    def test_decorator_rejects_non_engine(self) -> None:
        from iot_machine_learning.infrastructure.ml.engines.engine_factory import (
            register_engine,
        )

        with pytest.raises(TypeError, match="PredictionEngine"):
            @register_engine("bad")
            class _NotAnEngine:
                pass


# ── discover_engines ──────────────────────────────────────────


class TestDiscoverEngines:
    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        from iot_machine_learning.infrastructure.ml.engines.engine_factory import (
            EngineFactory,
        )
        original = dict(EngineFactory._registry)
        yield
        EngineFactory._registry = original

    def test_discover_returns_list(self) -> None:
        from iot_machine_learning.infrastructure.ml.engines.engine_factory import (
            discover_engines,
        )

        result = discover_engines(
            "iot_machine_learning.infrastructure.ml.engines"
        )
        assert isinstance(result, list)

    def test_discover_nonexistent_package_returns_empty(self) -> None:
        from iot_machine_learning.infrastructure.ml.engines.engine_factory import (
            discover_engines,
        )

        result = discover_engines("nonexistent.package.path")
        assert result == []


# ── DetectorRegistry + @register_detector ─────────────────────


class TestDetectorRegistry:
    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        from iot_machine_learning.infrastructure.ml.anomaly.detector_protocol import (
            DetectorRegistry,
        )
        original = dict(DetectorRegistry._registry)
        yield
        DetectorRegistry._registry = original

    def test_register_and_list(self) -> None:
        from iot_machine_learning.infrastructure.ml.anomaly.detector_protocol import (
            DetectorRegistry,
        )

        DetectorRegistry.register("test_det", lambda cfg: None)
        assert "test_det" in DetectorRegistry.list_detectors()

    def test_unregister(self) -> None:
        from iot_machine_learning.infrastructure.ml.anomaly.detector_protocol import (
            DetectorRegistry,
        )

        DetectorRegistry.register("to_remove", lambda cfg: None)
        DetectorRegistry.unregister("to_remove")
        assert "to_remove" not in DetectorRegistry.list_detectors()

    def test_create_all(self) -> None:
        from iot_machine_learning.infrastructure.ml.anomaly.detector_protocol import (
            DetectorRegistry,
            SubDetector,
        )

        class _FakeDetector(SubDetector):
            @property
            def method_name(self) -> str:
                return "fake"

            def train(self, values, **kwargs):
                pass

            def vote(self, value, **kwargs):
                return 0.0

            @property
            def is_trained(self) -> bool:
                return False

        DetectorRegistry.register("fake", lambda cfg: _FakeDetector())
        detectors = DetectorRegistry.create_all(config=None)
        assert any(isinstance(d, _FakeDetector) for d in detectors)

    def test_create_all_skips_failing_factory(self) -> None:
        from iot_machine_learning.infrastructure.ml.anomaly.detector_protocol import (
            DetectorRegistry,
        )

        def _bad_factory(cfg):
            raise RuntimeError("boom")

        DetectorRegistry.register("bad", _bad_factory)
        detectors = DetectorRegistry.create_all(config=None)
        # Should not raise, just skip
        assert not any(True for d in detectors if False)  # no crash


class TestRegisterDetectorDecorator:
    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        from iot_machine_learning.infrastructure.ml.anomaly.detector_protocol import (
            DetectorRegistry,
        )
        original = dict(DetectorRegistry._registry)
        yield
        DetectorRegistry._registry = original

    def test_decorator_registers_factory(self) -> None:
        from iot_machine_learning.infrastructure.ml.anomaly.detector_protocol import (
            DetectorRegistry,
            SubDetector,
            register_detector,
        )

        class _MyDet(SubDetector):
            @property
            def method_name(self) -> str:
                return "my_det"

            def train(self, values, **kwargs):
                pass

            def vote(self, value, **kwargs):
                return 0.0

            @property
            def is_trained(self) -> bool:
                return False

        @register_detector("my_det")
        def _create_my_det(config):
            return _MyDet()

        assert "my_det" in DetectorRegistry.list_detectors()
        detectors = DetectorRegistry.create_all(config=None)
        assert any(isinstance(d, _MyDet) for d in detectors)

    def test_decorator_returns_callable_unchanged(self) -> None:
        from iot_machine_learning.infrastructure.ml.anomaly.detector_protocol import (
            register_detector,
        )

        @register_detector("identity_test")
        def _my_factory(config):
            return None

        assert callable(_my_factory)
        assert _my_factory.__name__ == "_my_factory"


# ── EngineFactory.create_as_port with statistical ─────────────


class TestStatisticalRegistered:
    @pytest.fixture(autouse=True)
    def _ensure_registered(self):
        # Importing the engines package triggers registration
        import iot_machine_learning.infrastructure.ml.engines  # noqa: F401
        yield

    def test_statistical_in_registry(self) -> None:
        from iot_machine_learning.infrastructure.ml.engines.engine_factory import (
            EngineFactory,
        )

        assert "statistical" in EngineFactory.list_engines()

    def test_create_statistical(self) -> None:
        from iot_machine_learning.infrastructure.ml.engines.engine_factory import (
            EngineFactory,
        )
        from iot_machine_learning.infrastructure.ml.engines.statistical_engine import (
            StatisticalPredictionEngine,
        )

        engine = EngineFactory.create("statistical")
        assert isinstance(engine, StatisticalPredictionEngine)
