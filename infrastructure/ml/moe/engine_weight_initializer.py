"""Equipment-aware engine weight initialization and structural filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Set

if TYPE_CHECKING:
    from iot_machine_learning.domain.value_objects.equipment_class import EquipmentClass

EQUIPMENT_WEIGHT_MATRIX: Dict[str, Dict[str, float]] = {
    "PASTEURIZER": {"baseline": 0.15, "statistical": 0.10, "taylor": 0.55, "kalman": 0.20},
    "CIP": {"baseline": 0.60, "statistical": 0.20, "taylor": 0.10, "kalman": 0.10},
    "FILLER": {"baseline": 0.10, "statistical": 0.65, "taylor": 0.10, "kalman": 0.15},
    "PET_BLOWER": {"baseline": 0.15, "statistical": 0.20, "taylor": 0.20, "kalman": 0.45},
    "CONVEYOR": {"baseline": 0.20, "statistical": 0.25, "taylor": 0.10, "kalman": 0.45},
    "SILO": {"baseline": 0.10, "statistical": 0.15, "taylor": 0.60, "kalman": 0.15},
    "GENERIC": {"baseline": 0.25, "statistical": 0.25, "taylor": 0.25, "kalman": 0.25},
}

STRUCTURAL_INELIGIBLE: Dict[str, Set[str]] = {
    "FILLER": {"taylor"},
    "CIP": {"taylor"},
    "PASTEURIZER": {"baseline"},
    "GENERIC": set(),
}


class EquipmentTypeWeightInitializer:
    """Returns initial weights by equipment_class. Pure, no state."""

    def get_initial_weights(
        self, equipment_class: "EquipmentClass", available_engines: List[str]
    ) -> Dict[str, float]:
        """Return weights from EQUIPMENT_WEIGHT_MATRIX filtered to available engines.
        Renormalizes to sum 1.0 if engines are missing. Falls back to GENERIC.
        """
        key = equipment_class.value if hasattr(equipment_class, "value") else str(equipment_class)
        matrix = EQUIPMENT_WEIGHT_MATRIX.get(key, EQUIPMENT_WEIGHT_MATRIX["GENERIC"])
        filtered = {k: v for k, v in matrix.items() if k in available_engines}
        if not filtered:
            n = len(available_engines)
            return {k: 1.0 / n for k in available_engines} if n else {}
        total = sum(filtered.values())
        return {k: v / total for k, v in filtered.items()}


class StructuralEngineFilter:
    """Returns structurally ineligible engines for an equipment_class. Pure, no state."""

    def get_ineligible_engines(self, equipment_class: "EquipmentClass") -> Set[str]:
        """Return set of engine names structurally inappropriate. Never raises."""
        key = equipment_class.value if hasattr(equipment_class, "value") else str(equipment_class)
        return STRUCTURAL_INELIGIBLE.get(key, set())
