"""Types for numeric column analysis.

Data structures for analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class NumericColumnResult:
    """Result of analyzing a single numeric column.

    Attributes:
        stats: Basic statistics dict.
        structural: Structural analysis dict.
        anomaly_result: Anomaly detection result dict (or None).
        thresholds: Adaptive thresholds dict.
        triggers: List of trigger dicts.
        conclusion: Per-column conclusion string.
        confidence: Confidence score [0, 1].
    """

    stats: Dict[str, Any]
    structural: Dict[str, Any]
    anomaly_result: Optional[Dict[str, Any]]
    thresholds: Dict[str, float]
    triggers: List[Dict[str, Any]]
    conclusion: str
    confidence: float
