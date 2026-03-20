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
import re
import time
from typing import Any, Dict, List, Optional

from ..inhibition import InhibitionGate
from ..fusion import WeightedFusion

try:
    from ..plasticity import PlasticityTracker
except (ImportError, ModuleNotFoundError):
    PlasticityTracker = None  # type: ignore[assignment,misc]

from .types import TextAnalysisContext, TextAnalysisInput, TextCognitiveResult
from .signal_profiler import TextSignalProfiler
from .perception_collector import TextPerceptionCollector, DEFAULT_TEXT_WEIGHTS
from .severity_mapper import classify_text_severity
from .impact_detector import detect_impact_signals
from .memory_enricher import TextMemoryEnricher
from .explanation_assembler import TextExplanationAssembler

logger = logging.getLogger(__name__)

# Domain classification keywords (duplicated from conclusion_builder
# to avoid ml_service imports — lightweight, ~20 keywords)
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "infrastructure": [
        "server", "cpu", "memory", "disk", "network", "node",
        "cluster", "deploy", "container", "kubernetes", "latency",
    ],
    "security": [
        "vulnerability", "breach", "unauthorized", "firewall",
        "intrusion", "malware", "exploit", "authentication",
    ],
    "operations": [
        "incident", "outage", "downtime", "maintenance",
        "escalation", "sla", "recovery", "alert",
    ],
    "business": [
        "revenue", "cost", "budget", "forecast", "margin",
        "growth", "kpi", "target", "profit", "contract",
    ],
}


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
        self._profiler = TextSignalProfiler()
        self._collector = TextPerceptionCollector()
        self._enricher = TextMemoryEnricher()
        self._assembler = TextExplanationAssembler()
        self._inhibition = InhibitionGate()
        self._fusion = WeightedFusion()
        self._plasticity = (
            PlasticityTracker() if enable_plasticity and PlasticityTracker is not None
            else None
        )
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
        t0 = time.monotonic()
        domain = self._classify_domain(inp.full_text, ctx.domain_hint)
        n_chunks = len(inp.full_text) // 500 + 1  # approximate chunk count

        signal = self._profiler.profile(
            word_count=inp.word_count,
            sentences=inp.readability_sentences,
            avg_sentence_length=inp.readability_avg_sentence_length,
            vocabulary_richness=inp.readability_vocabulary_richness,
            sentiment_score=inp.sentiment_score,
            urgency_score=inp.urgency_score,
            domain=domain,
            paragraph_count=inp.paragraph_count,
            n_chunks=n_chunks,
            embedded_numeric_count=inp.readability_embedded_numeric_count,
            pattern_summary=inp.pattern_summary,
        )

        # Impact signal detection (scans text once, reused by severity_mapper)
        impact_result = detect_impact_signals(inp.full_text)

        perceive_ms = (time.monotonic() - t0) * 1000
        timing["perceive"] = perceive_ms
        phases.append({
            "kind": "perceive",
            "summary": {
                "word_count": inp.word_count,
                "domain": domain,
                "impact_score": impact_result.score,
                "impact_categories_hit": impact_result.n_categories_hit,
            },
            "duration_ms": perceive_ms,
        })

        # ── 2. ANALYZE (PREDICT phase) ──
        t0 = time.monotonic()
        perceptions = self._collector.collect(inp)
        predict_ms = (time.monotonic() - t0) * 1000
        timing["predict"] = predict_ms
        phases.append({
            "kind": "predict",
            "summary": {"n_engines": len(perceptions)},
            "duration_ms": predict_ms,
        })

        # ── 3. REMEMBER ──
        t0 = time.monotonic()
        recall_ctx = self._enricher.enrich(
            full_text=inp.full_text,
            domain=domain,
            cognitive_memory=ctx.cognitive_memory,
            document_id=ctx.document_id,
        )
        recall_ms = (time.monotonic() - t0) * 1000
        timing["remember"] = recall_ms

        # ── 4. REASON (ADAPT + INHIBIT + FUSE) ──

        # Adapt: get base weights from plasticity or defaults
        t0 = time.monotonic()
        engine_names = [p.engine_name for p in perceptions]
        base_weights = self._get_base_weights(domain, engine_names)
        adapted = self._plasticity is not None and self._plasticity.has_history(domain)
        adapt_ms = (time.monotonic() - t0) * 1000
        timing["adapt"] = adapt_ms
        phases.append({
            "kind": "adapt",
            "summary": {"regime": domain, "adapted": adapted},
            "duration_ms": adapt_ms,
        })

        # Inhibit: suppress unreliable sub-analyzers
        t0 = time.monotonic()
        inh_states = self._inhibition.compute(
            perceptions, base_weights, series_id=ctx.document_id,
        )
        inhibit_ms = (time.monotonic() - t0) * 1000
        timing["inhibit"] = inhibit_ms
        phases.append({
            "kind": "inhibit",
            "summary": {
                "n_inhibited": sum(
                    1 for s in inh_states if s.suppression_factor > 0.01
                ),
            },
            "duration_ms": inhibit_ms,
        })

        # Fuse: combine sub-analyzer scores
        t0 = time.monotonic()
        (
            fused_val, fused_conf, fused_trend,
            final_weights, selected, reason,
        ) = self._fusion.fuse(perceptions, inh_states)

        fusion_method = (
            "weighted_average" if len(perceptions) > 1 else "single_engine"
        )

        # Severity classification (3-axis: urgency + sentiment + impact)
        severity = classify_text_severity(
            urgency_score=inp.urgency_score,
            urgency_severity=inp.urgency_severity,
            sentiment_label=inp.sentiment_label,
            has_critical_keywords=inp.urgency_severity == "critical",
            domain=domain,
            full_text=inp.full_text,
            impact_result=impact_result,
        )

        fuse_ms = (time.monotonic() - t0) * 1000
        timing["fuse"] = fuse_ms
        phases.append({
            "kind": "fuse",
            "summary": {
                "selected_engine": selected,
                "fused_confidence": round(fused_conf, 4),
                "severity": severity.severity,
            },
            "duration_ms": fuse_ms,
        })

        # ── 5. EXPLAIN ──
        t0 = time.monotonic()
        explanation = self._assembler.assemble(
            document_id=ctx.document_id,
            signal=signal,
            perceptions=perceptions,
            inhibition_states=inh_states,
            final_weights=final_weights,
            selected_engine=selected,
            selection_reason=reason,
            fusion_method=fusion_method,
            fused_confidence=fused_conf,
            domain=domain,
            severity=severity,
            pipeline_phases=phases,
        )
        explain_ms = (time.monotonic() - t0) * 1000
        timing["explain"] = explain_ms

        # ── Build confidence ──
        confidence = self._compute_confidence(inp, recall_ctx.has_context)

        # ── Build analysis dict (backward-compatible) ──
        analysis = self._build_analysis_dict(
            inp, signal, perceptions, final_weights, impact_result,
        )

        # ── Build basic conclusion (caller may enrich with build_semantic_conclusion) ──
        conclusion = self._build_basic_conclusion(
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

    # ── Private helpers ──

    def _classify_domain(self, text: str, hint: str) -> str:
        """Auto-detect document domain from text content."""
        if hint:
            return hint

        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for domain_name, keywords in _DOMAIN_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                scores[domain_name] = count

        if not scores:
            return "general"
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def _get_base_weights(
        self, domain: str, engine_names: List[str],
    ) -> Dict[str, float]:
        """Get base weights from plasticity or defaults."""
        if self._plasticity is not None:
            try:
                pw = self._plasticity.get_weights(domain, engine_names)
                if pw:
                    return pw
            except Exception:
                pass
        return {name: DEFAULT_TEXT_WEIGHTS.get(name, 0.2) for name in engine_names}

    def _compute_confidence(
        self, inp: TextAnalysisInput, has_recall: bool,
    ) -> float:
        """Compute overall confidence from text metrics."""
        confidence = 0.75
        if inp.word_count > 100:
            confidence = 0.80
        if inp.word_count > 500:
            confidence = 0.85
        if inp.structural_available:
            confidence += 0.05
        if has_recall:
            confidence = min(0.95, confidence + 0.05)
        if inp.pattern_available:
            confidence = min(0.95, confidence + 0.02)
        return confidence

    def _build_analysis_dict(
        self,
        inp: TextAnalysisInput,
        signal: Any,
        perceptions: List[Any],
        final_weights: Dict[str, float],
        impact_result: Any = None,
    ) -> Dict[str, Any]:
        """Build backward-compatible analysis dict."""
        d: Dict[str, Any] = {
            "sentiment": inp.sentiment_label,
            "sentiment_score": inp.sentiment_score,
            "urgency_score": inp.urgency_score,
            "urgency_hits": inp.urgency_hits,
            "readability": {
                "avg_sentence_length": inp.readability_avg_sentence_length,
                "n_sentences": inp.readability_n_sentences,
                "vocabulary_richness": inp.readability_vocabulary_richness,
                "embedded_numeric_values": inp.readability_embedded_numeric_count,
            },
            "structural": {
                "sentence_length_regime": inp.structural_regime,
                "sentence_length_trend": inp.structural_trend,
                "sentence_length_stability": inp.structural_stability,
                "sentence_length_noise": inp.structural_noise,
            } if inp.structural_available else {},
            "patterns": {
                "n_patterns": inp.pattern_n_patterns,
                "change_points": inp.pattern_change_points,
                "spikes": inp.pattern_spikes,
                "summary": inp.pattern_summary,
            } if inp.pattern_available else {},
            "cognitive": {
                "engine_weights": {
                    k: round(v, 4) for k, v in final_weights.items()
                },
                "engine_perceptions": [
                    p.to_dict() for p in perceptions
                ],
                "signal_profile": signal.to_dict(),
            },
        }
        if impact_result is not None:
            d["impact"] = impact_result.to_dict()
        return d

    def _build_basic_conclusion(
        self,
        domain: str,
        severity: Any,
        inp: TextAnalysisInput,
        impact_result: Any = None,
    ) -> str:
        """Build a basic conclusion from severity and domain.

        The caller (ml_service text_analyzer) may replace this with the
        richer output of ``build_semantic_conclusion()``.
        """
        parts: List[str] = []

        # Domain line
        domain_label = domain.capitalize() if domain != "general" else "General"
        parts.append(f"Domain: {domain_label}")

        # Severity line
        parts.append(f"Severity: {severity.severity}")

        # Impact signals
        if impact_result is not None and impact_result.summary:
            parts.append(impact_result.summary)

        # Key signals
        if inp.urgency_severity in ("critical", "warning"):
            parts.append(f"Urgency: {inp.urgency_severity} (score: {inp.urgency_score:.2f})")
        if inp.sentiment_label != "neutral":
            parts.append(f"Sentiment: {inp.sentiment_label} (score: {inp.sentiment_score:.2f})")

        # Action
        if severity.action_required:
            parts.append(f"Recommended actions: {severity.recommended_action}")
        else:
            parts.append("No immediate action required.")

        return "\n".join(parts)
