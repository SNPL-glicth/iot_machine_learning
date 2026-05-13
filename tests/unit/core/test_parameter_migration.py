"""Tests for core/parameter_migration.py."""

import pytest

from core.parameters.parameter_migration import (
    register_epsilon_constants,
    register_statistical_thresholds,
    register_confidence_config,
    register_all_parameters,
    migrate_parameters,
)
from core.parameters.parameter_registry import (
    ParameterRegistry,
    ParameterCategory,
    ParameterScope,
)


class TestParameterMigration:
    def setup_method(self):
        """Reset registry before each test."""
        registry = ParameterRegistry()
        registry.reset()

    def test_migrate_epsilon_constants(self):
        registry = ParameterRegistry()
        register_epsilon_constants(registry)
        assert registry.get("EPSILON.COMPARISON") is not None
        assert registry.get("EPSILON.DIVISION") is not None
        assert registry.get("EPSILON.CONFIDENCE") is not None
        assert registry.get("EPSILON.CORRELATION") is not None
        assert registry.get("EPSILON.GRADIENT") is not None

    def test_migrate_statistical_thresholds(self):
        registry = ParameterRegistry()
        register_statistical_thresholds(registry)
        assert registry.get("STAT_THRESHOLDS.Z_SCORE_LOWER") is not None
        assert registry.get("STAT_THRESHOLDS.Z_SCORE_UPPER") is not None
        assert registry.get("STAT_THRESHOLDS.CONTAMINATION_DEFAULT") is not None
        assert registry.get("STAT_THRESHOLDS.CONTAMINATION_MIN") is not None
        assert registry.get("STAT_THRESHOLDS.CONTAMINATION_MAX") is not None
        assert registry.get("STAT_THRESHOLDS.IQR_FENCE_MULTIPLIER") is not None
        assert registry.get("STAT_THRESHOLDS.LOF_MAX_NEIGHBORS") is not None

    def test_migrate_all_parameters(self):
        registry = ParameterRegistry()
        register_all_parameters(registry)
        summary = registry.get_summary()
        assert summary["total_parameters"] > 0

    def test_no_circular_dependencies(self):
        registry = ParameterRegistry()
        register_all_parameters(registry)
        cycles = registry.check_dependencies()
        assert len(cycles) == 0

    def test_all_parameters_valid(self):
        registry = ParameterRegistry()
        register_all_parameters(registry)
        errors = registry.validate_all()
        assert len(errors) == 0

    def test_parameter_count_matches_audit(self):
        registry = ParameterRegistry()
        register_all_parameters(registry)
        summary = registry.get_summary()
        # Should have at least the core parameters we registered
        assert summary["total_parameters"] >= 14

    def test_category_distribution(self):
        registry = ParameterRegistry()
        register_all_parameters(registry)
        summary = registry.get_summary()
        assert summary["by_category"]["numerical_stability"] > 0
        assert summary["by_category"]["statistical_threshold"] > 0
        assert summary["by_category"]["confidence_calibration"] > 0

    def test_scope_distribution(self):
        registry = ParameterRegistry()
        register_all_parameters(registry)
        summary = registry.get_summary()
        assert summary["by_scope"]["global"] > 0

    def test_dependencies_exist(self):
        registry = ParameterRegistry()
        register_all_parameters(registry)
        z_lower = registry.get("STAT_THRESHOLDS.Z_SCORE_LOWER")
        cont_min = registry.get("STAT_THRESHOLDS.CONTAMINATION_MIN")
        cont_max = registry.get("STAT_THRESHOLDS.CONTAMINATION_MAX")
        conf_min = registry.get("CONFIDENCE.MIN_CONFIDENCE")
        # Check that some parameters still have dependencies
        assert len(z_lower.dependencies) > 0 or len(cont_min.dependencies) > 0 or len(cont_max.dependencies) > 0 or len(conf_min.dependencies) > 0

    def test_export_config_complete(self):
        registry = ParameterRegistry()
        register_all_parameters(registry)
        config = registry.export_config()
        assert "EPSILON.COMPARISON" in config
        assert "STAT_THRESHOLDS.Z_SCORE_LOWER" in config
        assert "CONFIDENCE.MIN_CONFIDENCE" in config
