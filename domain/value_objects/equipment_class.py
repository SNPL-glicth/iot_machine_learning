"""Equipment class enum for industrial bottling equipment."""

from __future__ import annotations

from enum import Enum


class EquipmentClass(str, Enum):
    """Industrial equipment types in bottling plants."""

    PASTEURIZER = "PASTEURIZER"
    CIP = "CIP"
    FILLER = "FILLER"
    PET_BLOWER = "PET_BLOWER"
    CONVEYOR = "CONVEYOR"
    SILO = "SILO"
    GENERIC = "GENERIC"

    @classmethod
    def from_device_type(cls, raw: str) -> "EquipmentClass":
        """Map raw device_type string to EquipmentClass."""
        normalized = raw.upper().strip()
        for member in cls:
            if member.value == normalized:
                return member
        for member in cls:
            if member.value.startswith(normalized) or normalized.startswith(member.value):
                return member
        return cls.GENERIC
