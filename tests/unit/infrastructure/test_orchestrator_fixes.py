"""Tests verifying fixes for pre-existing orchestrator bugs.

These bugs were latent because the cognitive pipeline was only used for
document analysis, never for sensor data. They surfaced when the
orchestrator was integrated into the batch runner via
OrchestratorPredictionAdapter.
"""

from unittest.mock import MagicMock

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import (
    MetaCognitiveOrchestrator,
)
from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags


class TestFlagsSnapshotExtractedFromKwargs:
    """Bug: _predict_internal checked 'flags_snapshot' in kwargs but never
    assigned it to a local variable, causing NameError at line 277."""

    def test_flags_snapshot_extracted_from_kwargs(self):
        """Predict with flags_snapshot should not raise NameError."""
        mock_engine = MagicMock()
        mock_engine.name = "mock_engine"
        mock_engine.can_handle.return_value = True
        mock_engine.predict.return_value = MagicMock(
            predicted_value=42.0,
            confidence=0.8,
            trend="stable",
            metadata={},
        )

        orchestrator = MetaCognitiveOrchestrator(
            engines=[mock_engine],
            budget_ms=500.0,
            enable_plasticity=False,
        )

        flags = FeatureFlags()
        # Before the fix, this raised:
        #   NameError: name 'flags_snapshot' is not defined
        result = orchestrator.predict(
            series_id="test_series",
            values=[10.0, 11.0, 12.0, 13.0, 14.0],
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],
            flags_snapshot=flags,
        )

        assert result is not None
        assert result.predicted_value == 42.0


class TestCorrelationPortStoredInInit:
    """Bug: correlation_port parameter was accepted in __init__ but never
    stored as self._correlation_port, causing AttributeError in FusePhase."""

    def test_correlation_port_stored_in_init(self):
        """_correlation_port should be accessible after construction."""
        mock_engine = MagicMock()
        mock_engine.name = "mock_engine"
        mock_engine.can_handle.return_value = True
        mock_engine.predict.return_value = MagicMock(
            predicted_value=42.0,
            confidence=0.8,
            trend="stable",
            metadata={},
        )

        mock_port = MagicMock()

        orchestrator = MetaCognitiveOrchestrator(
            engines=[mock_engine],
            budget_ms=500.0,
            enable_plasticity=False,
            correlation_port=mock_port,
        )

        assert hasattr(orchestrator, "_correlation_port")
        assert orchestrator._correlation_port is mock_port

    def test_correlation_port_none_by_default(self):
        """When not provided, _correlation_port should be None."""
        mock_engine = MagicMock()
        mock_engine.name = "mock_engine"

        orchestrator = MetaCognitiveOrchestrator(
            engines=[mock_engine],
            budget_ms=500.0,
            enable_plasticity=False,
        )

        assert hasattr(orchestrator, "_correlation_port")
        assert orchestrator._correlation_port is None

    def test_predict_with_correlation_port_does_not_crash(self):
        """FusePhase accesses _correlation_port; must not crash."""
        mock_engine = MagicMock()
        mock_engine.name = "mock_engine"
        mock_engine.can_handle.return_value = True
        mock_engine.predict.return_value = MagicMock(
            predicted_value=42.0,
            confidence=0.8,
            trend="stable",
            metadata={},
        )

        mock_port = MagicMock()

        orchestrator = MetaCognitiveOrchestrator(
            engines=[mock_engine],
            budget_ms=500.0,
            enable_plasticity=False,
            correlation_port=mock_port,
        )

        flags = FeatureFlags()
        # Before the fix, this raised in FusePhase.execute line 237:
        #   AttributeError: 'MetaCognitiveOrchestrator' object has no
        #   attribute '_correlation_port'
        result = orchestrator.predict(
            series_id="test_series",
            values=[10.0, 11.0, 12.0, 13.0, 14.0],
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],
            flags_snapshot=flags,
        )

        assert result is not None
