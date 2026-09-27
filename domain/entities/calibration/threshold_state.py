"""Domain entity defining adaptive calibration threshold states and quantiles.

Conforms to:
- ISO/IEC 25010:2023: Reliability, fault tolerance, and data integrity.
- ISO/IEC 22989:2022: AI trustworthiness and auditability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AdaptiveThresholdState:
    """Immutable state tracking real-time statistical threshold calibration."""

    sample_count: int
    running_mean: float
    running_variance: float
    current_threshold: float
    floor_threshold: float
    ceiling_threshold: float
    multiplier_k: float
    last_update_timestamp_ns: int

    def __post_init__(self) -> None:
        """Validate invariant constraints on thresholds and bounds."""
        if self.sample_count < 0:
            raise ValueError(f"sample_count must be non-negative, got {self.sample_count}")
        if self.floor_threshold <= 0.0:
            raise ValueError(f"floor_threshold must be positive, got {self.floor_threshold}")
        if self.ceiling_threshold < self.floor_threshold:
            raise ValueError(
                f"ceiling_threshold ({self.ceiling_threshold}) cannot be less than "
                f"floor_threshold ({self.floor_threshold})"
            )
        if not (self.floor_threshold <= self.current_threshold <= self.ceiling_threshold):
            raise ValueError(
                f"current_threshold ({self.current_threshold}) must lie in "
                f"[{self.floor_threshold}, {self.ceiling_threshold}]"
            )

    @property
    def running_std(self) -> float:
        """Return running standard deviation, guaranteeing non-negativity."""
        return float(self.running_variance ** 0.5) if self.running_variance > 0.0 else 0.0


@dataclass(frozen=True)
class DynamicConfidenceBand:
    """Adaptive confidence band for MoE action envelope modulation."""

    min_confidence: float
    max_confidence: float
    action_scale: float
    stop_ratio: float
    target_ratio: float
    horizon_steps: int


@dataclass(frozen=True)
class ThresholdAuditSnapshot:
    """Telemetry snapshot for ISO/IEC 22989 explainability audits."""

    metric_name: str
    observed_value: float
    threshold_applied: float
    is_admissible: bool
    state: AdaptiveThresholdState
    timestamp_ns: int
