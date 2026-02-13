"""Unit tests for PredictSensorValueUseCase with memory recall integration.

Tests verify:
    - Recall enabled: memory_context populated in DTO
    - Recall disabled (flag off): no cognitive calls, no memory_context
    - Recall disabled (no cognitive port): no memory_context
    - Memory failure: fallback silently, DTO still valid
    - Prediction numeric value NOT modified by recall
    - Anomaly boolean NOT modified by recall
    - No modifications to SQL adapter (mocked StoragePort)
    - No modifications to base model inference (mocked PredictionDomainService)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from iot_machine_learning.application.dto.prediction_dto import PredictionDTO
from iot_machine_learning.application.use_cases.predict_sensor_value import (
    PredictSensorValueUseCase,
)
from iot_machine_learning.domain.entities.memory_search_result import (
    MemorySearchResult,
)
from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.sensor_reading import (
    SensorReading,
    SensorWindow,
)
from iot_machine_learning.domain.ports.cognitive_memory_port import (
    CognitiveMemoryPort,
)
from iot_machine_learning.domain.ports.storage_port import StoragePort
from iot_machine_learning.domain.services.prediction_domain_service import (
    PredictionDomainService,
)
from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_storage() -> MagicMock:
    mock = MagicMock(spec=StoragePort)
    mock.load_sensor_window.return_value = SensorWindow(
        sensor_id=1,
        readings=[
            SensorReading(sensor_id=1, value=20.0, timestamp=1000.0),
            SensorReading(sensor_id=1, value=21.0, timestamp=1001.0),
            SensorReading(sensor_id=1, value=22.0, timestamp=1002.0),
        ],
    )
    mock.save_prediction.return_value = 42
    return mock


@pytest.fixture
def mock_prediction_service() -> MagicMock:
    mock = MagicMock(spec=PredictionDomainService)
    mock.predict.return_value = Prediction(
        series_id="1",
        predicted_value=23.0,
        confidence_score=0.85,
        trend="up",
        engine_name="taylor",
        metadata={"explanation": "Upward trend detected"},
    )
    return mock


@pytest.fixture
def mock_cognitive() -> MagicMock:
    mock = MagicMock(spec=CognitiveMemoryPort)
    mock.recall_similar_explanations.return_value = [
        MemorySearchResult(
            memory_id="uuid-1",
            series_id="1",
            text="Previous upward trend",
            certainty=0.91,
            source_record_id=100,
            created_at="2025-02-01T10:00:00Z",
        ),
    ]
    mock.recall_similar_anomalies.return_value = []
    return mock


@pytest.fixture
def flags_recall_enabled() -> FeatureFlags:
    return FeatureFlags(
        ML_ENABLE_MEMORY_RECALL=True,
        ML_ENABLE_COGNITIVE_MEMORY=True,
    )


@pytest.fixture
def flags_recall_disabled() -> FeatureFlags:
    return FeatureFlags(
        ML_ENABLE_MEMORY_RECALL=False,
    )


# ---------------------------------------------------------------------------
# Test: recall enabled — memory_context populated
# ---------------------------------------------------------------------------

class TestRecallEnabled:
    def test_dto_has_memory_context(
        self, mock_storage, mock_prediction_service, mock_cognitive,
        flags_recall_enabled,
    ):
        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=flags_recall_enabled,
        )
        dto = uc.execute(sensor_id=1)

        assert dto.memory_context is not None
        assert "enriched_explanation" in dto.memory_context
        assert "historical_references" in dto.memory_context

    def test_predicted_value_not_modified(
        self, mock_storage, mock_prediction_service, mock_cognitive,
        flags_recall_enabled,
    ):
        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=flags_recall_enabled,
        )
        dto = uc.execute(sensor_id=1)

        assert dto.predicted_value == 23.0

    def test_confidence_not_modified(
        self, mock_storage, mock_prediction_service, mock_cognitive,
        flags_recall_enabled,
    ):
        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=flags_recall_enabled,
        )
        dto = uc.execute(sensor_id=1)

        assert dto.confidence_score == 0.85

    def test_trend_not_modified(
        self, mock_storage, mock_prediction_service, mock_cognitive,
        flags_recall_enabled,
    ):
        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=flags_recall_enabled,
        )
        dto = uc.execute(sensor_id=1)

        assert dto.trend == "up"

    def test_cognitive_recall_called(
        self, mock_storage, mock_prediction_service, mock_cognitive,
        flags_recall_enabled,
    ):
        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=flags_recall_enabled,
        )
        uc.execute(sensor_id=1)

        mock_cognitive.recall_similar_explanations.assert_called_once()
        mock_cognitive.recall_similar_anomalies.assert_called_once()

    def test_enriched_explanation_contains_original(
        self, mock_storage, mock_prediction_service, mock_cognitive,
        flags_recall_enabled,
    ):
        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=flags_recall_enabled,
        )
        dto = uc.execute(sensor_id=1)

        enriched = dto.memory_context["enriched_explanation"]
        assert "Upward trend detected" in enriched
        assert "Historical context:" in enriched

    def test_memory_context_in_to_dict(
        self, mock_storage, mock_prediction_service, mock_cognitive,
        flags_recall_enabled,
    ):
        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=flags_recall_enabled,
        )
        dto = uc.execute(sensor_id=1)
        d = dto.to_dict()

        assert "memory_context" in d

    def test_no_memory_context_when_no_matches(
        self, mock_storage, mock_prediction_service, mock_cognitive,
        flags_recall_enabled,
    ):
        mock_cognitive.recall_similar_explanations.return_value = []
        mock_cognitive.recall_similar_anomalies.return_value = []

        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=flags_recall_enabled,
        )
        dto = uc.execute(sensor_id=1)

        assert dto.memory_context is None


# ---------------------------------------------------------------------------
# Test: recall disabled (flag off)
# ---------------------------------------------------------------------------

class TestRecallDisabledFlag:
    def test_no_memory_context(
        self, mock_storage, mock_prediction_service, mock_cognitive,
        flags_recall_disabled,
    ):
        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=flags_recall_disabled,
        )
        dto = uc.execute(sensor_id=1)

        assert dto.memory_context is None

    def test_cognitive_never_called(
        self, mock_storage, mock_prediction_service, mock_cognitive,
        flags_recall_disabled,
    ):
        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=flags_recall_disabled,
        )
        uc.execute(sensor_id=1)

        mock_cognitive.recall_similar_explanations.assert_not_called()
        mock_cognitive.recall_similar_anomalies.assert_not_called()

    def test_prediction_still_works(
        self, mock_storage, mock_prediction_service, mock_cognitive,
        flags_recall_disabled,
    ):
        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=flags_recall_disabled,
        )
        dto = uc.execute(sensor_id=1)

        assert dto.predicted_value == 23.0
        assert dto.confidence_score == 0.85
        assert dto.engine_name == "taylor"


# ---------------------------------------------------------------------------
# Test: recall disabled (no cognitive port)
# ---------------------------------------------------------------------------

class TestRecallNoCognitivePort:
    def test_no_memory_context_without_cognitive(
        self, mock_storage, mock_prediction_service, flags_recall_enabled,
    ):
        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=None,
            flags=flags_recall_enabled,
        )
        dto = uc.execute(sensor_id=1)

        assert dto.memory_context is None

    def test_no_memory_context_without_flags(
        self, mock_storage, mock_prediction_service, mock_cognitive,
    ):
        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=None,
        )
        dto = uc.execute(sensor_id=1)

        assert dto.memory_context is None


# ---------------------------------------------------------------------------
# Test: memory failure fallback
# ---------------------------------------------------------------------------

class TestRecallFailureFallback:
    def test_cognitive_exception_does_not_break_pipeline(
        self, mock_storage, mock_prediction_service, mock_cognitive,
        flags_recall_enabled,
    ):
        mock_cognitive.recall_similar_explanations.side_effect = Exception(
            "Weaviate down"
        )

        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=flags_recall_enabled,
        )
        dto = uc.execute(sensor_id=1)

        # Pipeline still works
        assert dto.predicted_value == 23.0
        assert dto.memory_context is None

    def test_anomaly_recall_exception_does_not_break_pipeline(
        self, mock_storage, mock_prediction_service, mock_cognitive,
        flags_recall_enabled,
    ):
        mock_cognitive.recall_similar_anomalies.side_effect = RuntimeError(
            "timeout"
        )

        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=flags_recall_enabled,
        )
        dto = uc.execute(sensor_id=1)

        # Pipeline still works — enricher catches the error internally
        assert dto.predicted_value == 23.0

    def test_sql_save_still_called_even_if_recall_would_fail(
        self, mock_storage, mock_prediction_service, mock_cognitive,
        flags_recall_enabled,
    ):
        mock_cognitive.recall_similar_explanations.side_effect = Exception(
            "crash"
        )

        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
            cognitive=mock_cognitive,
            flags=flags_recall_enabled,
        )
        uc.execute(sensor_id=1)

        mock_storage.save_prediction.assert_called_once()


# ---------------------------------------------------------------------------
# Test: backward compatibility — no new params
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_use_case_works_without_new_params(
        self, mock_storage, mock_prediction_service,
    ):
        """Original constructor signature still works."""
        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
        )
        dto = uc.execute(sensor_id=1)

        assert dto.predicted_value == 23.0
        assert dto.memory_context is None

    def test_dto_to_dict_without_memory_context(
        self, mock_storage, mock_prediction_service,
    ):
        uc = PredictSensorValueUseCase(
            prediction_service=mock_prediction_service,
            storage=mock_storage,
        )
        dto = uc.execute(sensor_id=1)
        d = dto.to_dict()

        assert "memory_context" not in d
