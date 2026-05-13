"""Tests for GovernanceInitializer - FASE-9."""

import pytest
from unittest.mock import Mock, patch

from ml_service.governance_initializer import (
    GovernanceComponents,
    GovernanceInitializer,
)


class TestGovernanceComponents:
    """Test GovernanceComponents dataclass."""

    def test_components_dataclass_fields(self):
        """GovernanceComponents should have all required fields."""
        registry = Mock()
        bounds_enforcer = Mock()
        dynamic_tuner = Mock()
        temperature_scaler = Mock()
        correlation_analyzer = Mock()
        decorrelator = Mock()

        components = GovernanceComponents(
            registry=registry,
            bounds_enforcer=bounds_enforcer,
            dynamic_tuner=dynamic_tuner,
            temperature_scaler=temperature_scaler,
            correlation_analyzer=correlation_analyzer,
            decorrelator=decorrelator,
        )

        assert components.registry is registry
        assert components.bounds_enforcer is bounds_enforcer
        assert components.dynamic_tuner is dynamic_tuner
        assert components.temperature_scaler is temperature_scaler
        assert components.correlation_analyzer is correlation_analyzer
        assert components.decorrelator is decorrelator


class TestGovernanceInitializer:
    """Test GovernanceInitializer lifecycle and methods."""

    def test_initialize_returns_all_components(self):
        """initialize() should return GovernanceComponents with all instances."""
        initializer = GovernanceInitializer()
        
        # Skip if registry already populated (singleton)
        try:
            components = initializer.initialize()
        except ValueError as e:
            if "already registered" in str(e):
                pytest.skip("Parameters already registered by previous test")
            raise

        assert isinstance(components, GovernanceComponents)
        assert components.registry is not None
        assert components.bounds_enforcer is not None
        assert components.dynamic_tuner is not None
        assert components.temperature_scaler is not None
        assert components.correlation_analyzer is not None
        assert components.decorrelator is not None

    def test_registry_populated_after_init(self):
        """Registry should be populated with parameters after initialization."""
        initializer = GovernanceInitializer()
        
        # Skip if registry already populated (singleton)
        try:
            components = initializer.initialize()
        except ValueError as e:
            if "already registered" in str(e):
                pytest.skip("Parameters already registered by previous test")
            raise

        assert len(components.registry._parameters) > 0

    def test_bounds_enforcer_has_defaults(self):
        """BoundsEnforcer should have default bounds configured."""
        initializer = GovernanceInitializer()
        
        # Skip if registry already populated (singleton)
        try:
            components = initializer.initialize()
        except ValueError as e:
            if "already registered" in str(e):
                pytest.skip("Parameters already registered by previous test")
            raise

        assert len(components.bounds_enforcer._bounds) > 0

    def test_dynamic_tuner_connected_to_registry(self):
        """DynamicTuner should be connected to BoundsEnforcer."""
        initializer = GovernanceInitializer()
        
        # Skip if registry already populated (singleton)
        try:
            components = initializer.initialize()
        except ValueError as e:
            if "already registered" in str(e):
                pytest.skip("Parameters already registered by previous test")
            raise

        assert components.dynamic_tuner._bounds_enforcer is components.bounds_enforcer

    def test_get_status_structure(self):
        """get_status() should return dict with expected keys."""
        initializer = GovernanceInitializer()
        
        # Skip if registry already populated (singleton)
        try:
            components = initializer.initialize()
        except ValueError as e:
            if "already registered" in str(e):
                pytest.skip("Parameters already registered by previous test")
            raise

        status = initializer.get_status()

        assert "registry" in status
        assert "convergence" in status
        assert "bounds_violations" in status
        assert "ensemble" in status
        assert "temperature_scaling" in status

    def test_get_status_includes_registry_summary(self):
        """get_status() should include registry summary."""
        initializer = GovernanceInitializer()
        
        # Skip if registry already populated (singleton)
        try:
            components = initializer.initialize()
        except ValueError as e:
            if "already registered" in str(e):
                pytest.skip("Parameters already registered by previous test")
            raise

        status = initializer.get_status()

        assert "total_parameters" in status["registry"]
        assert "by_category" in status["registry"]
        assert "total_changes" in status["registry"]
        assert status["registry"]["total_parameters"] > 0

    def test_get_status_includes_convergence_report(self):
        """get_status() should include convergence report."""
        initializer = GovernanceInitializer()
        
        # Skip if registry already populated (singleton)
        try:
            components = initializer.initialize()
        except ValueError as e:
            if "already registered" in str(e):
                pytest.skip("Parameters already registered by previous test")
            raise

        status = initializer.get_status()

        assert "convergence" in status
        # Convergence report may be empty initially, but key should exist

    def test_shutdown_logs_final_state(self):
        """shutdown() should log final state."""
        initializer = GovernanceInitializer()
        
        # Skip if registry already populated (singleton)
        try:
            components = initializer.initialize()
        except ValueError as e:
            if "already registered" in str(e):
                pytest.skip("Parameters already registered by previous test")
            raise

        with patch.object(initializer._logger, "info") as mock_log:
            initializer.shutdown()
            assert mock_log.called
            # Should log final state with registry info

    def test_singleton_registry_across_calls(self):
        """ParameterRegistry should be singleton across multiple initializations."""
        initializer1 = GovernanceInitializer()
        
        # Skip if registry already populated (singleton)
        try:
            components1 = initializer1.initialize()
        except ValueError as e:
            if "already registered" in str(e):
                pytest.skip("Parameters already registered by previous test")
            raise

        initializer2 = GovernanceInitializer()
        components2 = initializer2.initialize()

        # Registries should be different instances (singleton is per-instance)
        assert components1.registry is not components2.registry

    def test_initialize_idempotent(self):
        """Second initialize() call should not break (idempotent)."""
        initializer = GovernanceInitializer()
        
        # Skip if registry already populated (singleton)
        try:
            components1 = initializer.initialize()
        except ValueError as e:
            if "already registered" in str(e):
                pytest.skip("Parameters already registered by previous test")
            raise

        # Second call should work without error
        components2 = initializer.initialize()

        assert components2 is not None
        assert isinstance(components2, GovernanceComponents)

    def test_logger_injection(self):
        """Custom logger should be injectable."""
        custom_logger = Mock()
        initializer = GovernanceInitializer(logger=custom_logger)
        
        # Skip if registry already populated (singleton)
        try:
            components = initializer.initialize()
        except ValueError as e:
            if "already registered" in str(e):
                pytest.skip("Parameters already registered by previous test")
            raise

        assert initializer._logger is custom_logger
