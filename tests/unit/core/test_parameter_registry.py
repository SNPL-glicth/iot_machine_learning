"""Tests for core/parameter_registry.py."""

import pytest

from core.parameters.parameter_registry import (
    ParameterRegistry,
    ParameterMetadata,
    ParameterCategory,
    ParameterScope,
    ParameterChange,
)


class TestParameterRegistry:
    def setup_method(self):
        """Reset registry before each test."""
        registry = ParameterRegistry()
        registry.reset()

    def test_register_parameter(self):
        registry = ParameterRegistry()
        metadata = ParameterMetadata(
            name="TEST_PARAM",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=1.0,
            default_value=1.0,
        )
        registry.register(metadata)
        assert registry.get("TEST_PARAM") is not None
        assert registry.get_value("TEST_PARAM") == 1.0

    def test_get_parameter(self):
        registry = ParameterRegistry()
        metadata = ParameterMetadata(
            name="TEST_PARAM2",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=2.0,
            default_value=2.0,
        )
        registry.register(metadata)
        result = registry.get("TEST_PARAM2")
        assert result is not None
        assert result.value == 2.0

    def test_get_value(self):
        registry = ParameterRegistry()
        metadata = ParameterMetadata(
            name="TEST_PARAM3",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=3.0,
            default_value=3.0,
        )
        registry.register(metadata)
        assert registry.get_value("TEST_PARAM3") == 3.0

    def test_set_value_with_audit_trail(self):
        registry = ParameterRegistry()
        metadata = ParameterMetadata(
            name="TEST_PARAM4",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=1.0,
            default_value=1.0,
        )
        registry.register(metadata)
        registry.set_value("TEST_PARAM4", 2.0, "test change", "test_user")
        assert registry.get_value("TEST_PARAM4") == 2.0
        history = registry.get_history("TEST_PARAM4")
        assert len(history) == 1
        assert history[0].old_value == 1.0
        assert history[0].new_value == 2.0
        assert history[0].reason == "test change"
        assert history[0].changed_by == "test_user"

    def test_validate_all(self):
        registry = ParameterRegistry()
        metadata1 = ParameterMetadata(
            name="VALID_PARAM",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=1.0,
            default_value=1.0,
            min_value=0.0,
            max_value=2.0,
        )
        # Register with invalid value (should fail registration validation)
        metadata2 = ParameterMetadata(
            name="INVALID_PARAM",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=5.0,
            default_value=5.0,
            min_value=0.0,
            max_value=2.0,
        )
        registry.register(metadata1)
        # Register valid, then modify to invalid directly (bypass set_value validation)
        metadata3 = ParameterMetadata(
            name="VALID_BECOMES_INVALID",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=1.0,
            default_value=1.0,
            min_value=0.0,
            max_value=2.0,
        )
        registry.register(metadata3)
        registry.get("VALID_BECOMES_INVALID").value = 5.0
        errors = registry.validate_all()
        assert len(errors) == 1
        assert "VALID_BECOMES_INVALID" in errors[0]

    def test_check_dependencies_no_cycle(self):
        registry = ParameterRegistry()
        metadata1 = ParameterMetadata(
            name="PARAM_A",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=1.0,
            default_value=1.0,
            dependencies=["PARAM_B"],
        )
        metadata2 = ParameterMetadata(
            name="PARAM_B",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=2.0,
            default_value=2.0,
            dependencies=[],
        )
        registry.register(metadata1)
        registry.register(metadata2)
        cycles = registry.check_dependencies()
        assert len(cycles) == 0

    def test_check_dependencies_detects_cycle(self):
        registry = ParameterRegistry()
        metadata1 = ParameterMetadata(
            name="PARAM_C",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=1.0,
            default_value=1.0,
            dependencies=["PARAM_D"],
        )
        metadata2 = ParameterMetadata(
            name="PARAM_D",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=2.0,
            default_value=2.0,
            dependencies=["PARAM_C"],
        )
        registry.register(metadata1)
        registry.register(metadata2)
        cycles = registry.check_dependencies()
        assert len(cycles) == 1

    def test_get_by_category(self):
        registry = ParameterRegistry()
        metadata1 = ParameterMetadata(
            name="PARAM_E",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=1.0,
            default_value=1.0,
        )
        metadata2 = ParameterMetadata(
            name="PARAM_F",
            category=ParameterCategory.STATISTICAL_THRESHOLD,
            scope=ParameterScope.GLOBAL,
            value=2.0,
            default_value=2.0,
        )
        registry.register(metadata1)
        registry.register(metadata2)
        results = registry.get_by_category(ParameterCategory.NUMERICAL_STABILITY)
        assert len(results) == 1
        assert results[0].name == "PARAM_E"

    def test_get_by_scope(self):
        registry = ParameterRegistry()
        metadata1 = ParameterMetadata(
            name="PARAM_G",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=1.0,
            default_value=1.0,
        )
        metadata2 = ParameterMetadata(
            name="PARAM_H",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.PER_ENGINE,
            value=2.0,
            default_value=2.0,
        )
        registry.register(metadata1)
        registry.register(metadata2)
        results = registry.get_by_scope(ParameterScope.GLOBAL)
        assert len(results) == 1
        assert results[0].name == "PARAM_G"

    def test_get_history(self):
        registry = ParameterRegistry()
        metadata = ParameterMetadata(
            name="TEST_PARAM5",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=1.0,
            default_value=1.0,
        )
        registry.register(metadata)
        registry.set_value("TEST_PARAM5", 2.0, "change1")
        registry.set_value("TEST_PARAM5", 3.0, "change2")
        history = registry.get_history("TEST_PARAM5")
        assert len(history) == 2

    def test_export_config(self):
        registry = ParameterRegistry()
        metadata = ParameterMetadata(
            name="TEST_PARAM6",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=1.0,
            default_value=1.0,
        )
        registry.register(metadata)
        config = registry.export_config()
        assert "TEST_PARAM6" in config
        assert config["TEST_PARAM6"]["value"] == 1.0

    def test_get_summary(self):
        registry = ParameterRegistry()
        metadata = ParameterMetadata(
            name="TEST_PARAM7",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=1.0,
            default_value=1.0,
        )
        registry.register(metadata)
        summary = registry.get_summary()
        assert summary["total_parameters"] == 1
        assert summary["total_changes"] == 0

    def test_singleton_pattern(self):
        registry1 = ParameterRegistry()
        registry2 = ParameterRegistry()
        assert registry1 is registry2

    def test_validation_failure_rollback(self):
        registry = ParameterRegistry()
        metadata = ParameterMetadata(
            name="TEST_PARAM8",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=1.0,
            default_value=1.0,
            min_value=0.0,
            max_value=2.0,
        )
        registry.register(metadata)
        with pytest.raises(ValueError):
            registry.set_value("TEST_PARAM8", 5.0, "invalid change")
        assert registry.get_value("TEST_PARAM8") == 1.0

    def test_version_bump_on_change(self):
        registry = ParameterRegistry()
        metadata = ParameterMetadata(
            name="TEST_PARAM9",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=1.0,
            default_value=1.0,
        )
        registry.register(metadata)
        assert registry.get("TEST_PARAM9").version == "1.0.0"
        registry.set_value("TEST_PARAM9", 2.0, "change")
        assert registry.get("TEST_PARAM9").version == "1.0.1"

    def test_duplicate_registration_fails(self):
        registry = ParameterRegistry()
        metadata = ParameterMetadata(
            name="TEST_PARAM10",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=1.0,
            default_value=1.0,
        )
        registry.register(metadata)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(metadata)
