"""Unified narrative result entity.

Represents a reconciled narrative from multiple sources (prediction explanation,
anomaly narrative, text cognitive narrative). Ensures consistency and detects
contradictions between different narrative sources.

Pure data entity — immutable, no business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class UnifiedNarrative:
    """Result of narrative unification.

    Attributes:
        primary_verdict: Single-sentence summary of what happened.
        severity: The highest severity among all sources.
        confidence: The lowest confidence among all sources.
        contradictions: List of detected conflicts between narratives.
        sources_used: Which sources contributed to the unified narrative.
        suppressed: Which sources were ignored and why.
    """

    primary_verdict: str
    severity: str
    confidence: float
    contradictions: List[str] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)
    suppressed: List[str] = field(default_factory=list)
