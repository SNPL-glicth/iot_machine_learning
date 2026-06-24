"""FeatureContext — contexto enriquecido del pipeline para consumo MoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from iot_machine_learning.domain.entities.sensor_profile import SensorProfile
    from iot_machine_learning.domain.value_objects.industrial_event import EventContext


@dataclass(frozen=True)
class FeatureContext:
    """Features numéricos del pipeline para routing y fusión MoE."""

    regime: str
    mean: float
    std: float
    slope: float
    curvature: float
    noise_ratio: float
    stability: float
    hampel_outlier_mask: List[bool]
    spatial_correlation_score: float
    sensor_profile: Optional["SensorProfile"] = None
    relative_deviation: float = 0.0
    equipment_class: str = "GENERIC"
    event_context: Optional["EventContext"] = None
    seasonal_strength: float = 0.0
    dominant_period: int = 0
    regime_confidence: float = 0.5
    cross_regime_incoherence: bool = False
    regime_stability_score: float = 0.5

    @classmethod
    def from_structural_analysis(
        cls, regime: str, mean: float, std: float, slope: float,
        curvature: float, noise_ratio: float, stability: float,
        hampel_outlier_mask: List[bool] = None, spatial_correlation_score: float = 0.0,
        sensor_profile: Optional["SensorProfile"] = None,
        relative_deviation: float = 0.0, equipment_class: str = "GENERIC",
        event_context: Optional["EventContext"] = None,
        seasonal_strength: float = 0.0, dominant_period: int = 0,
        regime_confidence: float = 0.5, cross_regime_incoherence: bool = False,
        regime_stability_score: float = 0.5,
    ) -> "FeatureContext":
        if hampel_outlier_mask is None:
            hampel_outlier_mask = []
        return cls(
            regime=regime, mean=mean, std=std, slope=slope,
            curvature=curvature, noise_ratio=noise_ratio, stability=stability,
            hampel_outlier_mask=list(hampel_outlier_mask),
            spatial_correlation_score=spatial_correlation_score,
            sensor_profile=sensor_profile, relative_deviation=relative_deviation,
            equipment_class=equipment_class, event_context=event_context,
            seasonal_strength=seasonal_strength, dominant_period=dominant_period,
            regime_confidence=regime_confidence,
            cross_regime_incoherence=cross_regime_incoherence,
            regime_stability_score=regime_stability_score,
        )

    @classmethod
    def from_structural_analysis_with_profile(
        cls, regime: str, mean: float, std: float, slope: float,
        curvature: float, noise_ratio: float, stability: float,
        hampel_outlier_mask: List[bool] = None, spatial_correlation_score: float = 0.0,
        sensor_profile: Optional["SensorProfile"] = None,
        event_context: Optional["EventContext"] = None,
        seasonal_strength: float = 0.0, dominant_period: int = 0,
        regime_confidence: float = 0.5, cross_regime_incoherence: bool = False,
        regime_stability_score: float = 0.5,
    ) -> "FeatureContext":
        rd = 0.0
        ec = "GENERIC"
        if sensor_profile is not None:
            rd = sensor_profile.relative_deviation(std)
            ec = sensor_profile.equipment_class.value
        return cls.from_structural_analysis(
            regime=regime, mean=mean, std=std, slope=slope,
            curvature=curvature, noise_ratio=noise_ratio, stability=stability,
            hampel_outlier_mask=hampel_outlier_mask,
            spatial_correlation_score=spatial_correlation_score,
            sensor_profile=sensor_profile, relative_deviation=rd, equipment_class=ec,
            event_context=event_context,
            seasonal_strength=seasonal_strength, dominant_period=dominant_period,
            regime_confidence=regime_confidence,
            cross_regime_incoherence=cross_regime_incoherence,
            regime_stability_score=regime_stability_score,
        )

    @classmethod
    def empty(cls) -> "FeatureContext":
        return cls(
            regime="unknown", mean=0.0, std=0.0, slope=0.0, curvature=0.0,
            noise_ratio=0.0, stability=0.0, hampel_outlier_mask=[],
            spatial_correlation_score=0.0, sensor_profile=None,
            relative_deviation=0.0, equipment_class="GENERIC", event_context=None,
        )
