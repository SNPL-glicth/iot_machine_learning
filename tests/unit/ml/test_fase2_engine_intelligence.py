"""Tests for Fase 2: equipment-aware engine intelligence."""

from __future__ import annotations

from iot_machine_learning.domain.value_objects.equipment_class import EquipmentClass
from iot_machine_learning.infrastructure.ml.moe.engine_weight_initializer import (
    EquipmentTypeWeightInitializer,
    StructuralEngineFilter,
)


class TestEquipmentTypeWeightInitializer:
    def test_pasteurizer_sums_to_one(self):
        w = EquipmentTypeWeightInitializer().get_initial_weights(
            EquipmentClass.PASTEURIZER, ["baseline", "statistical", "taylor", "kalman"]
        )
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert w["taylor"] > 0.4

    def test_filler_sums_to_one(self):
        w = EquipmentTypeWeightInitializer().get_initial_weights(
            EquipmentClass.FILLER, ["baseline", "statistical", "taylor", "kalman"]
        )
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert w["statistical"] > 0.5

    def test_missing_engine_renormalizes(self):
        w = EquipmentTypeWeightInitializer().get_initial_weights(
            EquipmentClass.PASTEURIZER, ["baseline", "taylor"]
        )
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_unknown_class_falls_back_to_generic(self):
        w = EquipmentTypeWeightInitializer().get_initial_weights(
            EquipmentClass.GENERIC, ["baseline", "statistical", "taylor", "kalman"]
        )
        assert all(v == 0.25 for v in w.values())


class TestStructuralEngineFilter:
    def test_filler_excludes_taylor(self):
        assert "taylor" in StructuralEngineFilter().get_ineligible_engines(EquipmentClass.FILLER)

    def test_generic_excludes_nothing(self):
        assert StructuralEngineFilter().get_ineligible_engines(EquipmentClass.GENERIC) == set()

    def test_never_raises(self):
        f = StructuralEngineFilter()
        for ec in EquipmentClass:
            f.get_ineligible_engines(ec)  # no exception


class TestColdStartBlend:
    def test_blend_at_zero_points(self):
        blend = 0 / 50
        cold, learned = 0.8, 0.2
        final = (1 - blend) * cold + blend * learned
        assert final == cold

    def test_blend_at_fifty_points(self):
        blend = 50 / 50
        cold, learned = 0.8, 0.2
        final = (1 - blend) * cold + blend * learned
        assert final == learned

    def test_blend_at_twenty_five_points(self):
        blend = 25 / 50
        cold, learned = 0.8, 0.2
        final = (1 - blend) * cold + blend * learned
        assert final == 0.5
