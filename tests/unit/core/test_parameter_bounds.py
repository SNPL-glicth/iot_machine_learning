"""Tests for core/parameter_bounds.py."""

import pytest

from core.parameters.parameter_bounds import (
    ParameterBoundsEnforcer,
    BoundsConfig,
    BoundsResult,
)


class TestParameterBoundsEnforcer:
    def test_init_with_defaults(self):
        enforcer = ParameterBoundsEnforcer()
        assert "ML_BAYES_ALPHA" in enforcer._bounds
        assert "contamination" in enforcer._bounds

    def test_register_custom_bounds(self):
        enforcer = ParameterBoundsEnforcer(use_defaults=False)
        config = BoundsConfig(min_value=0.0, max_value=1.0)
        enforcer.register_bounds("custom_param", config)
        assert "custom_param" in enforcer._bounds

    def test_enforce_within_bounds(self):
        enforcer = ParameterBoundsEnforcer(use_defaults=False)
        enforcer.register_bounds("test_param", BoundsConfig(0.0, 1.0))
        result = enforcer.enforce("test_param", 0.5)
        assert result.was_clipped is False
        assert result.clipped_value == 0.5
        assert result.clip_reason == "ok"

    def test_enforce_below_min(self):
        enforcer = ParameterBoundsEnforcer(use_defaults=False)
        enforcer.register_bounds("test_param", BoundsConfig(0.1, 1.0))
        result = enforcer.enforce("test_param", 0.05)
        assert result.was_clipped is True
        assert result.clipped_value == 0.1
        assert result.clip_reason == "below_min"

    def test_enforce_above_max(self):
        enforcer = ParameterBoundsEnforcer(use_defaults=False)
        enforcer.register_bounds("test_param", BoundsConfig(0.0, 1.0))
        result = enforcer.enforce("test_param", 1.5)
        assert result.was_clipped is True
        assert result.clipped_value == 1.0
        assert result.clip_reason == "above_max"

    def test_enforce_soft_warning(self):
        enforcer = ParameterBoundsEnforcer(use_defaults=False)
        enforcer.register_bounds(
            "test_param",
            BoundsConfig(0.0, 1.0, soft_min=0.2, soft_max=0.8)
        )
        result = enforcer.enforce("test_param", 0.15)
        assert result.was_clipped is False
        assert result.clipped_value == 0.15
        assert result.clip_reason == "soft_warning_below"

    def test_enforce_all_multiple_params(self):
        enforcer = ParameterBoundsEnforcer(use_defaults=False)
        enforcer.register_bounds("param1", BoundsConfig(0.0, 1.0))
        enforcer.register_bounds("param2", BoundsConfig(0.0, 1.0))
        params = {"param1": 0.5, "param2": 1.5}
        results = enforcer.enforce_all(params)
        assert len(results) == 2
        assert results["param1"].was_clipped is False
        assert results["param2"].was_clipped is True

    def test_unknown_parameter_passthrough(self):
        enforcer = ParameterBoundsEnforcer(use_defaults=False)
        result = enforcer.enforce("unknown_param", 999.0)
        assert result.was_clipped is False
        assert result.clipped_value == 999.0
        assert result.clip_reason == "no_bounds_registered"

    def test_bounds_result_fields(self):
        enforcer = ParameterBoundsEnforcer(use_defaults=False)
        enforcer.register_bounds("test_param", BoundsConfig(0.0, 1.0))
        result = enforcer.enforce("test_param", 0.5)
        assert hasattr(result, "original_value")
        assert hasattr(result, "clipped_value")
        assert hasattr(result, "was_clipped")
        assert hasattr(result, "clip_reason")

    def test_clip_reason_populated(self):
        enforcer = ParameterBoundsEnforcer(use_defaults=False)
        enforcer.register_bounds("test_param", BoundsConfig(0.0, 1.0))
        result_ok = enforcer.enforce("test_param", 0.5)
        result_below = enforcer.enforce("test_param", -0.1)
        result_above = enforcer.enforce("test_param", 1.1)
        assert result_ok.clip_reason == "ok"
        assert result_below.clip_reason == "below_min"
        assert result_above.clip_reason == "above_max"
