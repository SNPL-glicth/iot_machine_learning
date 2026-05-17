"""Tests for Fase 1: equipment-aware FeatureContext, regime, and Hampel."""

from __future__ import annotations

from iot_machine_learning.domain.entities.sensor_profile import SensorProfile
from iot_machine_learning.domain.value_objects.equipment_class import EquipmentClass
from iot_machine_learning.infrastructure.ml.moe.feature_context import FeatureContext
from iot_machine_learning.infrastructure.ml.moe.regime.equipment_aware_regime_classifier import classify_regime
from iot_machine_learning.infrastructure.ml.cognitive.fusion.hampel_filter import hampel_filter, hampel_filter_with_profile

# --- FeatureContext ---

def test_from_structural_analysis_backward_compatible() -> None:
    ctx = FeatureContext.from_structural_analysis(
        regime="stable", mean=10.0, std=0.5, slope=0.0,
        curvature=0.0, noise_ratio=0.05, stability=0.1,
    )
    assert ctx.equipment_class == "GENERIC"
    assert ctx.relative_deviation == 0.0
    assert ctx.sensor_profile is None

def test_from_structural_analysis_with_profile_pasteurizer() -> None:
    profile = SensorProfile(
        series_id="1", equipment_class=EquipmentClass.PASTEURIZER,
        operational_range=(60.0, 85.0), setpoint_tolerance=0.5,
        noise_floor=0.3, maintenance_history_score=0.5,
    )
    ctx = FeatureContext.from_structural_analysis_with_profile(
        regime="stable", mean=72.0, std=0.8, slope=0.0,
        curvature=0.0, noise_ratio=0.005, stability=0.1,
        sensor_profile=profile,
    )
    assert ctx.relative_deviation == (0.8 - 0.3) / 0.5
    assert ctx.equipment_class == "PASTEURIZER"
    assert ctx.sensor_profile is profile

# --- EquipmentAwareRegimeClassifier ---

def test_classify_regime_no_profile_stable() -> None:
    r = classify_regime(noise_ratio=0.05, slope=0.001, std=0.1, mean=10.0)
    assert r.value == "STABLE"

def test_classify_regime_no_profile_volatile() -> None:
    r = classify_regime(noise_ratio=0.3, slope=0.001, std=0.3, mean=2.0)
    assert r.value == "VOLATILE"

def test_classify_regime_with_profile_stable_despite_high_absolute_noise() -> None:
    profile = SensorProfile(
        series_id="1", equipment_class=EquipmentClass.PASTEURIZER,
        operational_range=(60.0, 85.0), setpoint_tolerance=0.5,
        noise_floor=0.3, maintenance_history_score=0.5,
    )
    r = classify_regime(
        noise_ratio=0.005, slope=0.0, std=0.4, mean=72.0, sensor_profile=profile
    )
    assert r.value == "STABLE"

def test_classify_regime_with_profile_volatile_above_tolerance() -> None:
    profile = SensorProfile(
        series_id="1", equipment_class=EquipmentClass.PASTEURIZER,
        operational_range=(60.0, 85.0), setpoint_tolerance=0.5,
        noise_floor=0.3, maintenance_history_score=0.5,
    )
    r = classify_regime(
        noise_ratio=0.01, slope=0.0, std=1.5, mean=72.0, sensor_profile=profile
    )
    assert r.value == "VOLATILE"

# --- HampelFilter con profile ---

class _FakeProfile:
    def __init__(self, hampel_k: float) -> None:
        self.hampel_k = hampel_k

def test_hampel_filter_with_profile_uses_profile_k() -> None:
    """k=2.5 rejects a marginal outlier that k=3.0 accepts."""
    profile = _FakeProfile(hampel_k=2.5)
    perceptions = [_P("a", 10.0), _P("b", 20.0), _P("c", 30.0), _P("d", 65.0)]
    result_25 = hampel_filter_with_profile(perceptions, sensor_profile=profile)
    result_30 = hampel_filter_with_profile(perceptions, sensor_profile=None)
    assert len(result_25.rejected) == 1
    assert result_25.rejected[0][0] == "d"
    assert len(result_30.rejected) == 0

def test_hampel_filter_with_profile_none_uses_default_k() -> None:
    perceptions = [_P("a", 10.0), _P("b", 20.0), _P("c", 30.0)]
    result = hampel_filter_with_profile(perceptions, sensor_profile=None)
    result_30 = hampel_filter_with_profile(perceptions, sensor_profile=_FakeProfile(3.0))
    assert result.kept == result_30.kept
    assert result.rejected == result_30.rejected

class _P:
    """Minimal EnginePerception stand-in for tests."""
    def __init__(self, engine_name: str, predicted_value: float) -> None:
        self.engine_name = engine_name
        self.predicted_value = predicted_value
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _P):
            return NotImplemented
        return (self.engine_name, self.predicted_value) == (other.engine_name, other.predicted_value)
    def __repr__(self) -> str:
        return f"_P({self.engine_name!r}, {self.predicted_value})"

