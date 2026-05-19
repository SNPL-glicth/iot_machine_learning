"""Integration tests for FASE-1A: MetaCognitiveOrchestrator injection.

Validates:
- PredictionService accepts cognitive_orchestrator parameter
- _CognitivePredictionPort correctly adapts orchestrator calls
- PredictionDomainService receives orchestrator as primary engine
- Fallback to baseline+kalman when orchestrator is absent or fails
"""

from __future__ import annotations

import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock dotenv before any downstream import triggers it
if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = MagicMock()
    sys.modules["dotenv.main"] = MagicMock()
    sys.modules["dotenv.variables"] = MagicMock()

pytestmark = pytest.mark.integration


class TestCognitiveOrchestratorInjection:
    """FASE-1A: Orchestrator injected into PredictionService."""

    def test_prediction_service_accepts_cognitive_orchestrator(self) -> None:
        """PredictionService.__init__ accepts cognitive_orchestrator param."""
        from iot_machine_learning.ml_service.api.services.prediction_service import (
            PredictionService,
        )

        mock_conn = MagicMock()
        mock_storage = MagicMock()
        mock_repo = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.can_handle.return_value = True

        with patch(
            "iot_machine_learning.ml_service.api.services.prediction_service.EngineFactory"
        ) as mock_factory, patch(
            "iot_machine_learning.ml_service.api.services.prediction_service.get_feature_flags"
        ) as mock_flags:
            mock_flags.return_value = MagicMock()
            mock_baseline = MagicMock()
            mock_baseline.as_port.return_value = MagicMock(name="baseline_port")
            mock_kalman = MagicMock()
            mock_kalman.as_port.return_value = MagicMock(name="kalman_port")
            mock_factory.create.side_effect = lambda name: {
                "baseline_moving_average": mock_baseline,
                "kalman": mock_kalman,
            }[name]
            svc = PredictionService(
                conn=mock_conn,
                storage=mock_storage,
                threshold_repo=mock_repo,
                cognitive_orchestrator=mock_orchestrator,
            )

        assert svc._cognitive_orchestrator is mock_orchestrator

    def test_cognitive_prediction_port_adapts_flags_snapshot(self) -> None:
        """_CognitivePredictionPort passes flags_snapshot to orchestrator."""
        from iot_machine_learning.ml_service.api.services.prediction_service import (
            _CognitivePredictionPort,
        )

        mock_orchestrator = MagicMock()
        mock_flags = MagicMock()
        mock_result = MagicMock()
        mock_result.predicted_value = 42.0
        mock_result.confidence = 0.85
        mock_result.trend = "up"
        mock_result.metadata = {"regime": "STABLE"}
        mock_orchestrator.predict.return_value = mock_result

        port = _CognitivePredictionPort(mock_orchestrator, mock_flags)

        # Build a minimal fake window
        fake_window = MagicMock()
        fake_window.sensor_id = 123
        fake_window.values = [10.0, 11.0, 12.0]
        fake_window.timestamps = [1.0, 2.0, 3.0]

        prediction = port.predict(fake_window)

        mock_orchestrator.predict.assert_called_once()
        call_kwargs = mock_orchestrator.predict.call_args[1]
        assert "flags_snapshot" in call_kwargs
        assert call_kwargs["flags_snapshot"] is mock_flags
        assert prediction.predicted_value == 42.0
        assert prediction.confidence_score == 0.85
        assert prediction.trend == "up"
        assert prediction.engine_name == "meta_cognitive_orchestrator"

    def test_prediction_domain_service_uses_cognitive_first(self) -> None:
        """When cognitive is injected, it is the first engine in the list."""
        from iot_machine_learning.ml_service.api.services.prediction_service import (
            PredictionService,
        )

        mock_conn = MagicMock()
        mock_storage = MagicMock()
        mock_repo = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.can_handle.return_value = True

        with patch(
            "iot_machine_learning.ml_service.api.services.prediction_service.EngineFactory"
        ) as mock_factory, patch(
            "iot_machine_learning.ml_service.api.services.prediction_service.get_feature_flags"
        ) as mock_flags:
            mock_flags.return_value = MagicMock()
            mock_baseline = MagicMock()
            mock_baseline.as_port.return_value = MagicMock(name="baseline_port")
            mock_kalman = MagicMock()
            mock_kalman.as_port.return_value = MagicMock(name="kalman_port")
            mock_factory.create.side_effect = lambda name: {
                "baseline_moving_average": mock_baseline,
                "kalman": mock_kalman,
            }[name]
            svc = PredictionService(
                conn=mock_conn,
                storage=mock_storage,
                threshold_repo=mock_repo,
                cognitive_orchestrator=mock_orchestrator,
            )

            # Access the PredictionDomainService engines
            engines = svc._predict_use_case._prediction_service._engines
            assert len(engines) == 3
            assert engines[0].name == "meta_cognitive_orchestrator"

    def test_fallback_to_baseline_when_cognitive_fails(self) -> None:
        """PredictionDomainService falls back to last engine when cognitive fails."""
        from iot_machine_learning.domain.services.prediction_domain_service import (
            PredictionDomainService,
        )
        from iot_machine_learning.ml_service.api.services.prediction_service import (
            _CognitivePredictionPort,
        )

        mock_orchestrator = MagicMock()
        mock_flags = MagicMock()
        mock_orchestrator.can_handle.return_value = True
        mock_orchestrator.predict.side_effect = RuntimeError("cognitive_failure")

        cognitive_port = _CognitivePredictionPort(mock_orchestrator, mock_flags)

        mock_baseline = MagicMock()
        mock_baseline.name = "baseline_moving_average"
        mock_baseline.can_handle.return_value = True

        from iot_machine_learning.domain.entities.prediction import Prediction
        fallback_prediction = Prediction(
            series_id="1",
            predicted_value=25.0,
            confidence_score=0.5,
            trend="stable",
            engine_name="baseline_moving_average",
        )
        mock_baseline.predict.return_value = fallback_prediction

        domain_service = PredictionDomainService(
            engines=[cognitive_port, mock_baseline],
        )

        fake_window = MagicMock()
        fake_window.sensor_id = 1
        fake_window.size = 10
        fake_window.is_empty = False

        prediction = domain_service.predict(fake_window)

        # Should have fallen back to baseline (fallback suffix may vary by
        # domain service version; what matters is that baseline was used)
        assert prediction.predicted_value == 25.0
        assert prediction.engine_name in (
            "baseline_moving_average",
            "baseline_moving_average_fallback",
        )

    def test_routes_contains_cognitive_injection_logic(self) -> None:
        """routes.py contains the wiring code for cognitive orchestrator injection."""
        from pathlib import Path

        routes_path = Path(__file__).resolve().parent.parent.parent / "ml_service" / "api" / "routes.py"
        source = routes_path.read_text()

        assert "BatchEnterpriseContainer" in source
        assert "cognitive_adapter = container.get_cognitive_adapter()" in source
        assert "cognitive_orchestrator = cognitive_adapter.orchestrator" in source
        assert "cognitive_orchestrator_init_failed" in source

    def test_zero_regression_when_flag_disabled(self) -> None:
        """With flag false, PredictionService has no cognitive orchestrator."""
        from iot_machine_learning.ml_service.api.services.prediction_service import (
            PredictionService,
        )

        mock_conn = MagicMock()
        mock_storage = MagicMock()
        mock_repo = MagicMock()

        with patch(
            "iot_machine_learning.ml_service.api.services.prediction_service.EngineFactory"
        ) as mock_factory, patch(
            "iot_machine_learning.ml_service.api.services.prediction_service.get_feature_flags"
        ) as mock_flags:
            mock_flags.return_value = MagicMock()
            mock_baseline = MagicMock()
            mock_baseline.as_port.return_value = MagicMock(name="baseline_port")
            mock_kalman = MagicMock()
            mock_kalman.as_port.return_value = MagicMock(name="kalman_port")
            mock_factory.create.side_effect = lambda name: {
                "baseline_moving_average": mock_baseline,
                "kalman": mock_kalman,
            }[name]
            svc = PredictionService(
                conn=mock_conn,
                storage=mock_storage,
                threshold_repo=mock_repo,
            )

        assert svc._cognitive_orchestrator is None
