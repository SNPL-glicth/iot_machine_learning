"""Tests for Governance Routes - FASE-9."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException

from ml_service.api.routes_governance import router


@pytest.fixture
def mock_governance():
    """Mock governance components."""
    governance = Mock()
    governance.registry = Mock()
    governance.registry._parameters = {
        "ML_BAYES_ALPHA": Mock(category=Mock(value="LEARNING_RATE")),
    }
    governance.registry.get_value = Mock(return_value=0.1)
    governance.registry._changes = []
    governance.dynamic_tuner = Mock()
    governance.dynamic_tuner._convergence_detector = Mock()
    governance.dynamic_tuner._convergence_detector.reset = Mock()
    governance.dynamic_tuner._convergence_detector.get_report = Mock(return_value=None)
    return governance


@pytest.fixture
def mock_request(mock_governance):
    """Mock FastAPI request with governance state."""
    request = Mock()
    request.app.state.governance = mock_governance
    return request


class TestStatusEndpoint:
    """Test /governance/status endpoint."""

    def test_status_endpoint_200(self, mock_request):
        """Status endpoint should return 200."""
        mock_governance = mock_request.app.state.governance
        mock_governance.get_status = Mock(
            return_value={
                "registry": {"total_parameters": 14},
                "convergence": {},
                "bounds_violations": [],
                "ensemble": {},
                "temperature_scaling": {},
            }
        )
        from ml_service.api.routes_governance import get_governance_status
        result = get_governance_status(mock_request)
        assert result["registry"]["total_parameters"] == 14

    def test_status_response_structure(self, mock_request):
        """Status response should have expected structure."""
        mock_governance = mock_request.app.state.governance
        mock_governance.get_status = Mock(
            return_value={
                "registry": {"total_parameters": 14},
                "convergence": {},
                "bounds_violations": [],
                "ensemble": {},
                "temperature_scaling": {},
            }
        )
        from ml_service.api.routes_governance import get_governance_status
        result = get_governance_status(mock_request)
        assert "registry" in result
        assert "convergence" in result
        assert "bounds_violations" in result


class TestParametersEndpoint:
    """Test /governance/parameters endpoint."""

    def test_parameters_endpoint_200(self, mock_request):
        """Parameters endpoint should return 200."""
        from ml_service.api.routes_governance import get_parameters
        result = get_parameters(mock_request)
        assert "parameters" in result

    def test_parameters_response_is_list(self, mock_request):
        """Parameters response should contain list."""
        from ml_service.api.routes_governance import get_parameters
        result = get_parameters(mock_request)
        assert "parameters" in result
        assert isinstance(result["parameters"], list)


class TestHistoryEndpoint:
    """Test /governance/history endpoint."""

    def test_history_endpoint_200(self, mock_request):
        """History endpoint should return 200."""
        from ml_service.api.routes_governance import get_parameter_history
        result = get_parameter_history(mock_request)
        assert "history" in result

    def test_history_with_parameter_filter(self, mock_request):
        """History should filter by parameter name when provided."""
        mock_governance = mock_request.app.state.governance
        mock_change = Mock()
        mock_change.parameter_name = "ML_BAYES_ALPHA"
        mock_change.old_value = 0.1
        mock_change.new_value = 0.08
        mock_change.timestamp = Mock()
        mock_change.timestamp.isoformat = Mock(return_value="2026-02-13")
        mock_change.reason = "convergence"

        mock_governance.registry._changes = [mock_change]
        from ml_service.api.routes_governance import get_parameter_history
        result = get_parameter_history(mock_request, parameter_name="ML_BAYES_ALPHA")
        assert len(result["history"]) == 1

    def test_history_limit_respected(self, mock_request):
        """History should respect limit parameter."""
        mock_governance = mock_request.app.state.governance
        # Create many changes
        changes = []
        for i in range(100):
            mock_change = Mock()
            mock_change.parameter_name = f"PARAM_{i}"
            mock_change.old_value = i
            mock_change.new_value = i + 1
            mock_change.timestamp = Mock()
            mock_change.timestamp.isoformat = Mock(return_value="2026-02-13")
            mock_change.reason = "test"
            changes.append(mock_change)

        mock_governance.registry._changes = changes
        from ml_service.api.routes_governance import get_parameter_history
        result = get_parameter_history(mock_request, limit=10)
        assert len(result["history"]) <= 10


class TestResetConvergenceEndpoint:
    """Test /governance/reset-convergence endpoint."""

    def test_reset_convergence_endpoint_200(self, mock_request):
        """Reset convergence endpoint should return 200."""
        from ml_service.api.routes_governance import reset_convergence
        result = reset_convergence(mock_request)
        assert result["status"] == "reset_complete"


class TestGovernanceNotInitialized:
    """Test endpoints when governance not initialized."""

    def test_status_503_without_governance(self):
        """Status should return 503 when governance not initialized."""
        request = Mock()
        request.app.state.governance = None
        from ml_service.api.routes_governance import get_governance_status
        with pytest.raises(HTTPException) as exc_info:
            get_governance_status(request)
        assert exc_info.value.status_code == 503

    def test_parameters_503_without_governance(self):
        """Parameters should return 503 when governance not initialized."""
        request = Mock()
        request.app.state.governance = None
        from ml_service.api.routes_governance import get_parameters
        with pytest.raises(HTTPException) as exc_info:
            get_parameters(request)
        assert exc_info.value.status_code == 503
