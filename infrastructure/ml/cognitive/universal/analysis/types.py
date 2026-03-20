"""Data types for UniversalAnalysisEngine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from iot_machine_learning.domain.entities.explainability.explanation import Explanation
from iot_machine_learning.domain.services.severity_rules import SeverityResult


class InputType(Enum):
    """Detected input data type."""
    TEXT = "text"
    NUMERIC = "numeric"
    TABULAR = "tabular"
    MIXED = "mixed"
    SPECIAL_CHARS = "special_chars"
    JSON = "json"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class UniversalInput:
    """Universal input container — any data type.

    Attributes:
        raw_data: Original input (str, List[float], Dict, etc.)
        detected_type: Auto-detected InputType enum
        metadata: Pre-computed metrics (word_count, n_rows, n_columns, etc.)
            Structure varies by type:
            - TEXT: word_count, paragraph_count, char_count, language
            - NUMERIC: n_points, mean, std, has_timestamps
            - TABULAR: n_rows, n_columns, column_names, numeric_columns
            - MIXED: combines above
        domain_hint: Optional domain override (e.g., "infrastructure")
        series_id: Identifier for tracking/plasticity
    """
    raw_data: Any
    detected_type: InputType
    metadata: Dict[str, Any] = field(default_factory=dict)
    domain_hint: str = ""
    series_id: str = "unknown"


@dataclass(frozen=True)
class UniversalResult:
    """Output of UniversalAnalysisEngine.

    Attributes:
        explanation: Domain Explanation object (same as MetaCognitiveOrchestrator)
        severity: Severity classification
        analysis: Detailed analysis dict (backward-compatible)
        confidence: Overall confidence [0, 1]
        domain: Detected/assigned domain (infrastructure, security, etc.)
        input_type: Detected InputType
        pipeline_timing: Per-phase timing (perceive, analyze, remember, reason, explain)
        recall_context: Memory enrichment if cognitive_memory was provided
    """
    explanation: Explanation
    severity: SeverityResult
    analysis: Dict[str, Any]
    confidence: float
    domain: str
    input_type: InputType
    pipeline_timing: Dict[str, float] = field(default_factory=dict)
    recall_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        return {
            "explanation": self.explanation.to_dict(),
            "severity": {
                "risk_level": self.severity.risk_level,
                "severity": self.severity.severity,
                "action_required": self.severity.action_required,
                "recommended_action": self.severity.recommended_action,
            },
            "analysis": self.analysis,
            "confidence": round(self.confidence, 4),
            "domain": self.domain,
            "input_type": self.input_type.value,
            "pipeline_timing": {k: round(v, 3) for k, v in self.pipeline_timing.items()},
            "recall_context": self.recall_context,
        }


@dataclass(frozen=True)
class UniversalContext:
    """Pipeline configuration and environment.

    Attributes:
        series_id: Identifier for plasticity tracking
        tenant_id: Multi-tenant isolation
        cognitive_memory: Optional CognitiveMemoryPort for semantic recall
        domain_hint: Override auto-detected domain
        budget_ms: Pipeline time budget
    """
    series_id: str
    tenant_id: str = ""
    cognitive_memory: Optional[object] = None
    domain_hint: str = ""
    budget_ms: float = 2000.0
