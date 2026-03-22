"""TextCognitiveEngine — deep analysis engine for text.

Reusable cognitive engine at the same layer as ``MetaCognitiveOrchestrator``.
Orchestrates a five-phase pipeline:

    1. **Perceive** — build signal profile from pre-computed text metrics.
    2. **Analyze**  — map sub-analyzer scores to ``EnginePerception[]``.
    3. **Remember** — recall similar past documents from cognitive memory.
    4. **Reason**   — inhibit unreliable engines, fuse, classify severity.
    5. **Explain**  — assemble ``Explanation`` domain object.

Design principles:
    - No imports from ml_service — receives pre-computed scores via
      ``TextAnalysisInput``.
    - Produces the same ``Explanation`` domain object as
      ``MetaCognitiveOrchestrator``.
    - Graceful-fail on every external dependency.
    - Domain-agnostic — works for logs, contracts, reports, trading notes.

Single entry point: ``TextCognitiveEngine.analyze()``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .types import TextAnalysisContext, TextAnalysisInput, TextCognitiveResult
from .pipeline import (
    TextPerceivePhase,
    TextAnalyzePhase,
    TextRememberPhase,
    TextReasonPhase,
    TextExplainPhase,
)
from .engine_helpers import compute_confidence, build_analysis_dict, build_basic_conclusion

logger = logging.getLogger(__name__)


class TextCognitiveEngine:
    """Deep analysis engine for text — thinks like a human.

    Perceives, analyzes, remembers, reasons, and explains.

    Args:
        enable_plasticity: Enable regime-contextual weight learning.
        budget_ms: Pipeline time budget in milliseconds.
    """

    def __init__(
        self,
        *,
        enable_plasticity: bool = True,
        budget_ms: float = 2000.0,
    ) -> None:
        self._perceive = TextPerceivePhase()
        self._analyze = TextAnalyzePhase()
        self._remember = TextRememberPhase()
        self._reason = TextReasonPhase(enable_plasticity=enable_plasticity)
        self._explain = TextExplainPhase()
        self._budget_ms = budget_ms

    def analyze(
        self,
        inp: TextAnalysisInput,
        ctx: TextAnalysisContext,
    ) -> TextCognitiveResult:
        """Run full cognitive pipeline for text analysis.

        Args:
            inp: Pre-computed text analysis scores (from ml_service
                 analyzers: sentiment, urgency, readability, etc.).
            ctx: Pipeline configuration and environment.

        Returns:
            ``TextCognitiveResult`` with ``Explanation`` domain object,
            severity, analysis dict, confidence, and timing.
        """
        timing: Dict[str, float] = {}
        phases: List[Dict[str, Any]] = []

        # ── 1. PERCEIVE ──
        domain, signal, impact_result, perceive_summary = self._perceive.execute(
            word_count=inp.word_count,
            readability_sentences=inp.readability_sentences,
            readability_avg_sentence_length=inp.readability_avg_sentence_length,
            readability_vocabulary_richness=inp.readability_vocabulary_richness,
            sentiment_score=inp.sentiment_score,
            urgency_score=inp.urgency_score,
            paragraph_count=inp.paragraph_count,
            embedded_numeric_count=inp.readability_embedded_numeric_count,
            pattern_summary=inp.pattern_summary,
            full_text=inp.full_text,
            domain_hint=ctx.domain_hint,
            timing=timing,
        )
        phases.append(perceive_summary)

        # ── 2. ANALYZE (PREDICT phase) ──
        perceptions, predict_summary = self._analyze.execute(inp, timing)
        phases.append(predict_summary)

        # ── 3. REMEMBER ──
        recall_ctx = self._remember.execute(
            full_text=inp.full_text,
            domain=domain,
            cognitive_memory=ctx.cognitive_memory,
            document_id=ctx.document_id,
            timing=timing,
        )

        # ── 4. REASON (ADAPT + INHIBIT + FUSE) ──
        (
            fused_val, fused_conf, fused_trend,
            final_weights, selected, reason, severity, reason_summaries,
        ) = self._reason.execute(
            perceptions=perceptions,
            domain=domain,
            document_id=ctx.document_id,
            urgency_score=inp.urgency_score,
            urgency_severity=inp.urgency_severity,
            sentiment_label=inp.sentiment_label,
            full_text=inp.full_text,
            impact_result=impact_result,
            timing=timing,
        )
        phases.extend(reason_summaries)

        # ── 5. EXPLAIN ──
        explanation = self._explain.execute(
            document_id=ctx.document_id,
            signal=signal,
            perceptions=perceptions,
            inhibition_states=None,  # Will be set by reason phase
            final_weights=final_weights,
            selected_engine=selected,
            selection_reason=reason,
            fusion_method="weighted_average" if len(perceptions) > 1 else "single_engine",
            fused_confidence=fused_conf,
            domain=domain,
            severity=severity,
            pipeline_phases=phases,
            timing=timing,
        )

        # ── Build confidence ──
        confidence = compute_confidence(inp, recall_ctx.has_context)

        # ── Build analysis dict (backward-compatible) ──
        analysis = build_analysis_dict(
            inp, signal, perceptions, final_weights, impact_result,
        )

        # ── Build basic conclusion (caller may enrich with build_semantic_conclusion) ──
        conclusion = build_basic_conclusion(
            domain, severity, inp, impact_result,
        )

        logger.debug(
            "text_cognitive_pipeline",
            extra={
                "document_id": ctx.document_id,
                "domain": domain,
                "severity": severity.severity,
                "confidence": round(confidence, 3),
                "total_ms": round(sum(timing.values()), 2),
            },
        )

        return TextCognitiveResult(
            explanation=explanation,
            conclusion=conclusion,
            severity=severity,
            analysis=analysis,
            confidence=confidence,
            domain=domain,
            pipeline_timing=timing,
            recall_context=recall_ctx.to_dict() if recall_ctx.has_context else None,
        )
