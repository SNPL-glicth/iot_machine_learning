"""Types for Pattern Interpreter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InterpretedPattern:
    """Human-interpretable pattern result.
    
    Transforms technical pattern detection into actionable insights.
    
    Attributes:
        pattern_type: Technical pattern identifier
        short_name: Human-friendly pattern name
        description: Context-aware explanation
        severity_hint: Severity classification
        domain_context: Domain-specific interpretation
        confidence: Confidence in interpretation (0-1)
        data_type: Type of data this pattern applies to
    """
    pattern_type: str          # technical: "cusum_drift", "delta_spike", "regime_change"
    short_name: str            # human: "Escalada crítica", "Spike anómalo", "Cambio de régimen"
    description: str           # full context-aware description
    severity_hint: str         # "info" | "warning" | "critical"
    domain_context: str        # how this pattern relates to the domain
    confidence: float          # [0, 1]
    data_type: str             # "text" | "numeric" | "universal"
