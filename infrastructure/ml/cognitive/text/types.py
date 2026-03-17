"""Data types for the TextCognitiveEngine pipeline.

Pure value objects — no I/O, no state, no side effects.
No imports from ml_service — only domain layer.

``TextAnalysisContext``  — pipeline configuration / environment.
``TextAnalysisInput``    — pre-computed analysis scores (from ml_service analyzers).
``TextCognitiveResult``  — full pipeline output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from iot_machine_learning.domain.entities.explainability.explanation import Explanation
from iot_machine_learning.domain.services.severity_rules import SeverityResult


@dataclass(frozen=True)
class TextAnalysisContext:
    """Pipeline configuration and environment.

    Attributes:
        document_id: Unique identifier for the document.
        tenant_id: Tenant identifier for multi-tenant isolation.
        filename: Original filename (for display/logging).
        weaviate_url: Weaviate base URL (enables embedding + recall).
            ``None`` disables Weaviate features gracefully.
        cognitive_memory: Optional ``CognitiveMemoryPort`` implementation.
            ``None`` disables memory recall gracefully.
        domain_hint: Override auto-detected domain classification.
            Empty string means auto-detect.
        budget_ms: Pipeline time budget in milliseconds.
    """

    document_id: str
    tenant_id: str = ""
    filename: str = ""
    weaviate_url: Optional[str] = None
    cognitive_memory: Optional[object] = None
    domain_hint: str = ""
    budget_ms: float = 2000.0


@dataclass(frozen=True)
class TextAnalysisInput:
    """Pre-computed text analysis scores.

    Produced by ml_service text analyzers, consumed by the engine.
    All fields are primitives — no ml_service type imports.
    """

    full_text: str
    word_count: int
    paragraph_count: int

    # Sentiment
    sentiment_score: float
    sentiment_label: str
    sentiment_positive_count: int = 0
    sentiment_negative_count: int = 0

    # Urgency
    urgency_score: float = 0.0
    urgency_severity: str = "info"
    urgency_total_hits: int = 0
    urgency_hits: Dict[str, int] = field(default_factory=dict)

    # Readability
    readability_avg_sentence_length: float = 0.0
    readability_n_sentences: int = 0
    readability_vocabulary_richness: float = 0.0
    readability_embedded_numeric_count: int = 0
    readability_sentences: List[str] = field(default_factory=list)

    # Structural
    structural_regime: str = "unknown"
    structural_trend: str = "stable"
    structural_stability: float = 0.0
    structural_noise: float = 0.0
    structural_available: bool = False

    # Patterns
    pattern_n_patterns: int = 0
    pattern_change_points: List[int] = field(default_factory=list)
    pattern_spikes: List[int] = field(default_factory=list)
    pattern_available: bool = False
    pattern_summary: str = ""


@dataclass(frozen=True)
class TextCognitiveResult:
    """Full output of the TextCognitiveEngine pipeline.

    Attributes:
        explanation: ``Explanation`` domain object (same type as
            ``MetaCognitiveOrchestrator`` produces).
        conclusion: Human-readable conclusion from the engine
            (basic severity+domain summary; caller may enrich).
        severity: Severity classification result.
        analysis: Detailed analysis scores dict (backward-compatible).
        confidence: Overall confidence score [0, 1].
        pipeline_timing: Per-phase timing in milliseconds.
        recall_context: Memory recall enrichment dict (if available).
        domain: Auto-detected or hinted document domain.
    """

    explanation: Explanation
    conclusion: str
    severity: SeverityResult
    analysis: Dict[str, Any]
    confidence: float
    domain: str = "general"
    pipeline_timing: Dict[str, float] = field(default_factory=dict)
    recall_context: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses or logging."""
        d: Dict[str, Any] = {
            "explanation": self.explanation.to_dict(),
            "conclusion": self.conclusion,
            "severity": {
                "risk_level": self.severity.risk_level,
                "severity": self.severity.severity,
                "action_required": self.severity.action_required,
                "recommended_action": self.severity.recommended_action,
            },
            "analysis": self.analysis,
            "confidence": round(self.confidence, 4),
            "domain": self.domain,
            "pipeline_timing": {
                k: round(v, 3) for k, v in self.pipeline_timing.items()
            },
        }
        if self.recall_context:
            d["recall_context"] = self.recall_context
        return d

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to the dict format expected by existing callers.

        Matches the return shape of ``analyze_text_document()``.
        """
        return {
            "analysis": self.analysis,
            "adaptive_thresholds": {
                "urgency_warning": 0.4,
                "urgency_critical": 0.7,
                "sentiment_negative": -0.2,
            },
            "conclusion": self.conclusion,
            "confidence": round(self.confidence, 3),
        }
