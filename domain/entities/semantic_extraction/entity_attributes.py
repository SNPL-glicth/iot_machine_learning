"""Type-specific entity attributes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class MetricAttributes:
    """Attributes for METRIC entity type."""
    value: float
    unit: str
    metric_class: str  # pressure, temperature, flow, level, vibration
    is_out_of_range: bool = False
    reference_range: Optional[Tuple[float, float]] = None
    deviation_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "metric_class": self.metric_class,
            "is_out_of_range": self.is_out_of_range,
            "reference_range": self.reference_range,
            "deviation_percent": round(self.deviation_percent, 4),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricAttributes":
        """Create from dictionary."""
        range_tuple = None
        if data.get("reference_range"):
            r = data["reference_range"]
            if isinstance(r, (list, tuple)) and len(r) == 2:
                range_tuple = (float(r[0]), float(r[1]))
        
        return cls(
            value=float(data.get("value", 0)),
            unit=str(data.get("unit", "")),
            metric_class=str(data.get("metric_class", "unknown")),
            is_out_of_range=bool(data.get("is_out_of_range", False)),
            reference_range=range_tuple,
            deviation_percent=float(data.get("deviation_percent", 0)),
        )


@dataclass(frozen=True)
class EquipmentAttributes:
    """Attributes for EQUIPMENT entity type."""
    equipment_class: str  # compressor, valve, pump, motor
    equipment_id: str  # Numeric/alphanumeric ID
    parent_system: Optional[str] = None
    is_critical_path: bool = False
    redundancy_status: str = "unknown"  # primary, backup, redundant
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "equipment_class": self.equipment_class,
            "equipment_id": self.equipment_id,
            "parent_system": self.parent_system,
            "is_critical_path": self.is_critical_path,
            "redundancy_status": self.redundancy_status,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EquipmentAttributes":
        return cls(
            equipment_class=str(data.get("equipment_class", "unknown")),
            equipment_id=str(data.get("equipment_id", "")),
            parent_system=data.get("parent_system"),
            is_critical_path=bool(data.get("is_critical_path", False)),
            redundancy_status=str(data.get("redundancy_status", "unknown")),
        )


@dataclass(frozen=True)
class LocationAttributes:
    """Attributes for LOCATION entity type."""
    location_type: str  # sector, building, unit, area, floor
    parent_location: Optional[str] = None
    coordinates: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "location_type": self.location_type,
            "parent_location": self.parent_location,
            "coordinates": self.coordinates,
        }


@dataclass(frozen=True)
class TemporalAttributes:
    """Attributes for TEMPORAL entity type."""
    temporal_type: str  # date, time, duration, relative
    iso_format: Optional[str] = None
    is_future: bool = False
    is_past: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "temporal_type": self.temporal_type,
            "iso_format": self.iso_format,
            "is_future": self.is_future,
            "is_past": self.is_past,
        }


@dataclass(frozen=True)
class AlertAttributes:
    """Attributes for ALERT entity type."""
    alert_level: str  # critical, warning, info, error
    alert_category: str  # safety, operational, maintenance, security
    requires_ack: bool = False
    auto_escalate: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_level": self.alert_level,
            "alert_category": self.alert_category,
            "requires_ack": self.requires_ack,
            "auto_escalate": self.auto_escalate,
        }


# Union type for all attributes
EntityAttributes = MetricAttributes | EquipmentAttributes | LocationAttributes | TemporalAttributes | AlertAttributes | Dict[str, Any]
