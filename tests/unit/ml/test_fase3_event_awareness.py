"""Tests for Fase 3: event awareness."""

from __future__ import annotations

from iot_machine_learning.domain.entities.sensor_profile import SensorProfile
from iot_machine_learning.domain.value_objects.equipment_class import EquipmentClass
from iot_machine_learning.domain.value_objects.industrial_event import (
    EventContext,
    IndustrialEvent,
)
from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
    EnginePerception,
)
from iot_machine_learning.infrastructure.ml.cognitive.fusion.hampel_filter import (
    hampel_filter_with_profile,
)
from iot_machine_learning.infrastructure.ml.moe.events.industrial_event_detector import (
    detect_industrial_event,
)


class TestIndustrialEventDetector:
    def test_detect_startup_from_cusum_ramp_pasteurizer(self):
        values = [20.0] * 8 + [45.0, 72.0]
        profile = SensorProfile("s1", EquipmentClass.PASTEURIZER, (0, 100), 1.0, 0.3, 0.5)
        ctx = detect_industrial_event(values, ["cusum_ramp_detected"], profile)
        assert ctx.detected_event == IndustrialEvent.STARTUP
        assert ctx.event_confidence >= 0.7

    def test_detect_cip_from_ramp_cip_equipment(self):
        values = [72.0] * 8 + [73.0, 74.0]
        profile = SensorProfile("s1", EquipmentClass.CIP, (0, 100), 1.0, 1.5, 0.5)
        ctx = detect_industrial_event(values, ["cusum_ramp_detected"], profile)
        assert ctx.detected_event == IndustrialEvent.CIP_CYCLE

    def test_detect_none_when_no_flags(self):
        ctx = detect_industrial_event([72.0] * 10, [], None)
        assert ctx == EventContext.none()
        assert not ctx.is_active

    def test_detect_fault_transient_from_clamping(self):
        ctx = detect_industrial_event([1.0] * 10, ["value_clamped:4"], None)
        assert ctx.detected_event == IndustrialEvent.FAULT_TRANSIENT

    def test_detect_never_raises_on_empty_values(self):
        ctx = detect_industrial_event([], [], None)
        assert ctx == EventContext.none()


class TestEventContext:
    def test_is_active_true(self):
        assert EventContext(IndustrialEvent.STARTUP, 0, 0.8).is_active

    def test_is_active_false_low_confidence(self):
        assert not EventContext(IndustrialEvent.STARTUP, 0, 0.3).is_active

    def test_none_factory(self):
        assert not EventContext.none().is_active


class TestHampelFilterWithEventContext:
    def test_k_amplified_during_active_event(self):
        perceptions = [
            EnginePerception("a", 10.0, 0.9),
            EnginePerception("b", 10.5, 0.9),
            EnginePerception("c", 13.0, 0.9),
        ]
        profile = SensorProfile("s1", EquipmentClass.FILLER, (0, 100), 1.0, 1.0, 0.5, hampel_k=2.5)
        event = EventContext(IndustrialEvent.STARTUP, 0, 0.9)
        result = hampel_filter_with_profile(perceptions, sensor_profile=profile, event_context=event)
        assert len(result.rejected) == 0

    def test_k_not_amplified_without_event(self):
        perceptions = [
            EnginePerception("a", 10.0, 0.9),
            EnginePerception("b", 10.5, 0.9),
            EnginePerception("c", 13.0, 0.9),
        ]
        profile = SensorProfile("s1", EquipmentClass.FILLER, (0, 100), 1.0, 1.0, 0.5, hampel_k=2.5)
        result = hampel_filter_with_profile(perceptions, sensor_profile=profile, event_context=None)
        assert len(result.rejected) == 1
        assert result.rejected[0][0] == "c"
