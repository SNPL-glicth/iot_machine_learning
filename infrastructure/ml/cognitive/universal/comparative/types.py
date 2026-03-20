"""Data types for UniversalComparativeEngine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..analysis.types import UniversalResult


@dataclass(frozen=True)
class ComparisonContext:
    """Input to comparative engine.

    Attributes:
        current_result: Result from UniversalAnalysisEngine
        series_id: For memory recall filtering
        tenant_id: Multi-tenant isolation
        cognitive_memory: CognitiveMemoryPort implementation
        domain: Classified domain (infrastructure, security, etc.)
    """
    current_result: UniversalResult
    series_id: str
    tenant_id: str = ""
    cognitive_memory: Optional[object] = None
    domain: str = "general"


@dataclass(frozen=True)
class ComparisonResult:
    """Output of comparative analysis.

    Attributes:
        severity_delta_pct: Severity change vs average of similar past incidents (%)
            Example: +60.0 means 60% more severe
        urgency_delta_pct: Urgency change (%)
        topic_overlap_pct: % of topics/keywords in common with historical matches
        top_similar: Top 3 most similar past analyses
            Each dict: {doc_id, score, summary, severity, timestamp, resolution_time}
        delta_conclusion: Human-readable comparison
        resolution_probability: Estimated resolution probability [0, 1]
        estimated_resolution_time: Human-readable time estimate
    """
    severity_delta_pct: float
    urgency_delta_pct: float
    topic_overlap_pct: float
    top_similar: List[Dict[str, Any]]
    delta_conclusion: str
    resolution_probability: Optional[float] = None
    estimated_resolution_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        return {
            "severity_delta_pct": round(self.severity_delta_pct, 2),
            "urgency_delta_pct": round(self.urgency_delta_pct, 2),
            "topic_overlap_pct": round(self.topic_overlap_pct, 2),
            "top_similar": self.top_similar,
            "delta_conclusion": self.delta_conclusion,
            "resolution_probability": (
                round(self.resolution_probability, 3)
                if self.resolution_probability is not None
                else None
            ),
            "estimated_resolution_time": self.estimated_resolution_time,
        }


@dataclass(frozen=True)
class ColdStartResult:
    """Returned when insufficient historical data for comparison.
    
    Indicates that comparative analysis cannot be performed yet because
    minimum number of similar documents has not been reached.
    
    Attributes:
        reason: Why comparative analysis unavailable ("insufficient_history")
        docs_found: Number of similar documents found
        docs_needed: Minimum number required for comparison
        message: Human-readable explanation
    """
    reason: str
    docs_found: int
    docs_needed: int
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        return {
            "reason": self.reason,
            "docs_found": self.docs_found,
            "docs_needed": self.docs_needed,
            "message": self.message,
        }
