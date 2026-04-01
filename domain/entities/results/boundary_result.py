"""Boundary check result entity.

Represents the outcome of a domain boundary check operation.
Identifies whether data is within the acceptable domain for processing,
with rejection reasons and quality scoring.

Pure data entity — immutable, no business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class BoundaryResult:
    """Result of domain boundary validation.

    Attributes:
        within_domain: True if data is acceptable for processing.
        rejection_reason: Why data was rejected (None if within_domain=True).
        data_quality_score: Quality score 0.0–1.0 (0.0 if rejected).
        warnings: List of non-fatal warnings about data quality.
    """

    within_domain: bool
    rejection_reason: Optional[str] = None
    data_quality_score: float = 1.0
    warnings: List[str] = field(default_factory=list)
