"""RUL value objects — frozen dataclasses for estimator output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RULEstimate:
    """Immutable estimate of remaining useful life.

    Attributes:
        time_to_failure_hours: Estimated hours until failure (None if no deterioration).
        urgency: LOW, MEDIUM, or CRITICAL.
        confidence: LOW, MEDIUM, or HIGH.
        deterioration_rate: Normalized deterioration rate [0, 1].
        human_readable: Single-line summary for UI logs.
    """

    time_to_failure_hours: Optional[float]
    urgency: str
    confidence: str
    deterioration_rate: float
    human_readable: str
