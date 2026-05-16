"""Tests for OrchestratorPredictionAdapter and get_cognitive_adapter()."""

from unittest.mock import MagicMock, patch

import pytest

from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
from iot_machine_learning.ml_service.runners.adapters.orchestrator_prediction import (
    OrchestratorPredictionAdapter,
)
from iot_machine_learning.ml_service.runners.wiring.container import (
    BatchEnterpriseContainer,
)


class MockEngine:
    """Minimal PredictionEngine mock."""

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def predict(self, values, timestamps=None):
        from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult
        return PredictionResult(
            predicted_value=sum(values) / len(values),
            confidence=0.8,
            trend="stable",
            metadata={"selected_engine": self._name},
        )

    def can_handle(self, n_points):
        return n_points >= 2

    def as_port(self):
        from iot_machine_learning.infrastructure.ml.interfaces import (
            PredictionEnginePortBridge,
        )
        return PredictionEnginePortBridge(self)


class TestCognitiveAdapter:
    def test_container_has_cognitive_adapter_method(self):
        flags = FeatureFlags()
        container = BatchEnterpriseContainer(
            engine=MagicMock(), flags=flags,
        )
        assert hasattr(container, "get_cognitive_adapter")

    @patch(
        "iot_machine_learning.ml_service.runners.wiring.container.BatchEnterpriseContainer._build_prediction_engines"
    )
    def test_cognitive_adapter_returns_batch_result(self, mock_build_engines):
        mock_build_engines.return_value = [MockEngine("baseline")], None

        flags = FeatureFlags()
        container = BatchEnterpriseContainer(
            engine=MagicMock(), flags=flags,
        )

        adapter = container.get_cognitive_adapter()
        assert isinstance(adapter, OrchestratorPredictionAdapter)

    def test_orchestrator_prediction_adapter_predict_with_window(self):
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import (
            MetaCognitiveOrchestrator,
        )

        engine = MockEngine("baseline")
        orchestrator = MetaCognitiveOrchestrator(
            engines=[engine],
            budget_ms=500.0,
            enable_plasticity=True,
        )

        storage = MagicMock()
        audit = MagicMock()
        flags = FeatureFlags()

        adapter = OrchestratorPredictionAdapter(
            orchestrator=orchestrator,
            storage=storage,
            audit=audit,
            flags=flags,
        )

        # predict_with_window con datos sintéticos
        from iot_machine_learning.domain.entities.iot.sensor_reading import (
            Reading, SensorWindow,
        )
        readings = [
            Reading(series_id="1", value=float(v), timestamp=float(i))
            for i, v in enumerate([10.0, 11.0, 12.0, 13.0, 14.0])
        ]
        window = SensorWindow(series_id="1", readings=readings)

        result = adapter.predict_with_window(window)

        assert result.predicted_value is not None
        assert 0.0 <= result.confidence <= 1.0
        assert result.trend in ("up", "down", "stable")

    def test_orchestrator_has_plasticity_enabled(self):
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import (
            MetaCognitiveOrchestrator,
        )

        engine = MockEngine("baseline")
        orchestrator = MetaCognitiveOrchestrator(
            engines=[engine],
            budget_ms=500.0,
            enable_plasticity=True,
        )

        # Verify plasticity tracker exists and is configured
        assert orchestrator._plasticity is not None
        assert hasattr(orchestrator, "record_actual")

    def test_container_cognitive_vs_standard_adapter(self):
        flags = FeatureFlags()
        container = BatchEnterpriseContainer(
            engine=MagicMock(), flags=flags,
        )

        # By default (flag False) -> standard adapter
        assert flags.ML_USE_COGNITIVE_ORCHESTRATOR is False

        # Cognitive flag True
        flags.ML_USE_COGNITIVE_ORCHESTRATOR = True
        container2 = BatchEnterpriseContainer(
            engine=MagicMock(), flags=flags,
        )
        assert hasattr(container2, "get_cognitive_adapter")

    def test_predict_times_out_and_falls_back(self):
        import time
        from iot_machine_learning.ml_service.runners.adapters.orchestrator_prediction import (
            OrchestratorPredictionAdapter,
        )

        # Mock orchestrator that sleeps longer than timeout
        slow_orchestrator = MagicMock()

        def slow_predict(*args, **kwargs):
            time.sleep(10)
            return MagicMock(
                predicted_value=99.0,
                confidence=0.5,
                trend="stable",
                metadata={},
            )

        slow_orchestrator.predict.side_effect = slow_predict

        storage = MagicMock()
        audit = MagicMock()

        # Mock flags with timeout attribute (FeatureFlags is frozen dataclass)
        flags = MagicMock()
        flags.ML_ORCHESTRATOR_TIMEOUT_S = 0.5  # 500ms timeout for test
        flags.ML_ENABLE_AUDIT_LOGGING = False

        adapter = OrchestratorPredictionAdapter(
            orchestrator=slow_orchestrator,
            storage=storage,
            audit=audit,
            flags=flags,
        )

        from iot_machine_learning.domain.entities.iot.sensor_reading import (
            Reading, SensorWindow,
        )
        readings = [
            Reading(series_id="1", value=10.0, timestamp=0.0)
            for _ in range(5)
        ]
        window = SensorWindow(series_id="1", readings=readings)

        t0 = time.monotonic()
        result = adapter.predict_with_window(window)
        elapsed = time.monotonic() - t0

        # Must return quickly (< 6 seconds, but ideally < 2s with 0.5s timeout + overhead)
        assert elapsed < 3.0, f"Took {elapsed:.1f}s, expected < 3s"
        # Must return a BatchPredictionResult (fallback), not raise
        assert result is not None
        assert result.predicted_value is not None
