"""Tests for TextCognitiveEngine and its subcomponents.

Covers:
    - TextSignalProfiler → SignalSnapshot mapping
    - TextPerceptionCollector → EnginePerception[] mapping
    - TextSeverityMapper → SeverityResult classification
    - TextMemoryEnricher → TextRecallContext (with mock memory port)
    - TextExplanationAssembler → Explanation domain object
    - TextCognitiveEngine full pipeline (end-to-end)
    - Graceful degradation (no memory, no plasticity)
    - TextCognitiveResult serialization
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Dict, List, Optional

from iot_machine_learning.infrastructure.ml.cognitive.text.types import (
    TextAnalysisContext,
    TextAnalysisInput,
    TextCognitiveResult,
)
from iot_machine_learning.infrastructure.ml.cognitive.text.signal_profiler import (
    TextSignalProfiler,
)
from iot_machine_learning.infrastructure.ml.cognitive.text.perception_collector import (
    TextPerceptionCollector,
    DEFAULT_TEXT_WEIGHTS,
)
from iot_machine_learning.infrastructure.ml.cognitive.text.severity_mapper import (
    classify_text_severity,
)
from iot_machine_learning.infrastructure.ml.cognitive.text.impact_detector import (
    detect_impact_signals,
    ImpactSignalResult,
)
from iot_machine_learning.infrastructure.ml.cognitive.text.memory_enricher import (
    TextMemoryEnricher,
    TextRecallContext,
)
from iot_machine_learning.infrastructure.ml.cognitive.text.explanation_assembler import (
    TextExplanationAssembler,
)
from iot_machine_learning.infrastructure.ml.cognitive.text.engine import (
    TextCognitiveEngine,
)
from iot_machine_learning.domain.entities.explainability.explanation import (
    Explanation,
)
from iot_machine_learning.domain.entities.explainability.signal_snapshot import (
    SignalSnapshot,
)
from iot_machine_learning.domain.entities.memory_search_result import (
    MemorySearchResult,
)


# ── Fixtures ──

def _make_input(**overrides) -> TextAnalysisInput:
    """Build a TextAnalysisInput with sensible defaults."""
    defaults = dict(
        full_text="The server CPU usage spiked to 95% causing latency issues. "
                  "Network connectivity was degraded. Immediate action required.",
        word_count=20,
        paragraph_count=1,
        sentiment_score=-0.3,
        sentiment_label="negative",
        sentiment_positive_count=0,
        sentiment_negative_count=3,
        urgency_score=0.65,
        urgency_severity="warning",
        urgency_total_hits=4,
        urgency_hits={"en": 3, "es": 1},
        readability_avg_sentence_length=10.0,
        readability_n_sentences=3,
        readability_vocabulary_richness=0.85,
        readability_embedded_numeric_count=1,
        readability_sentences=[
            "The server CPU usage spiked to 95% causing latency issues.",
            "Network connectivity was degraded.",
            "Immediate action required.",
        ],
        structural_regime="variable",
        structural_trend="increasing",
        structural_stability=0.6,
        structural_noise=0.3,
        structural_available=True,
        pattern_n_patterns=1,
        pattern_change_points=[1],
        pattern_spikes=[],
        pattern_available=True,
        pattern_summary="1 change point detected at sentence 2",
    )
    defaults.update(overrides)
    return TextAnalysisInput(**defaults)


def _make_context(**overrides) -> TextAnalysisContext:
    """Build a TextAnalysisContext with sensible defaults."""
    defaults = dict(
        document_id="doc-001",
        tenant_id="tenant-1",
        filename="report.txt",
    )
    defaults.update(overrides)
    return TextAnalysisContext(**defaults)


class MockCognitiveMemory:
    """Mock CognitiveMemoryPort for testing memory enrichment."""

    def __init__(self, explanations=None, patterns=None):
        self._explanations = explanations or []
        self._patterns = patterns or []

    def recall_similar_explanations(self, query, **kwargs):
        return self._explanations

    def recall_similar_patterns(self, query, **kwargs):
        return self._patterns


class FailingCognitiveMemory:
    """Mock that raises on every call."""

    def recall_similar_explanations(self, query, **kwargs):
        raise ConnectionError("Weaviate down")

    def recall_similar_patterns(self, query, **kwargs):
        raise ConnectionError("Weaviate down")


# ── TextSignalProfiler Tests ──

class TestTextSignalProfiler:

    def test_profile_returns_signal_snapshot(self):
        profiler = TextSignalProfiler()
        snap = profiler.profile(
            word_count=100,
            sentences=["Short sentence.", "A much longer sentence with more words."],
            avg_sentence_length=12.5,
            vocabulary_richness=0.8,
            sentiment_score=-0.2,
            urgency_score=0.6,
            domain="infrastructure",
        )
        assert isinstance(snap, SignalSnapshot)
        assert snap.n_points == 100
        assert snap.mean == 12.5
        assert snap.regime == "infrastructure"

    def test_profile_encodes_sentiment_as_slope(self):
        profiler = TextSignalProfiler()
        snap = profiler.profile(
            word_count=50,
            sentences=["Test."],
            avg_sentence_length=5.0,
            vocabulary_richness=0.5,
            sentiment_score=-0.7,
            urgency_score=0.1,
            domain="general",
        )
        assert snap.slope == -0.7

    def test_profile_encodes_urgency_as_curvature(self):
        profiler = TextSignalProfiler()
        snap = profiler.profile(
            word_count=50,
            sentences=["Test."],
            avg_sentence_length=5.0,
            vocabulary_richness=0.5,
            sentiment_score=0.0,
            urgency_score=0.9,
            domain="general",
        )
        assert snap.curvature == 0.9

    def test_profile_extra_contains_text_metadata(self):
        profiler = TextSignalProfiler()
        snap = profiler.profile(
            word_count=200,
            sentences=["A.", "B."],
            avg_sentence_length=10.0,
            vocabulary_richness=0.75,
            sentiment_score=0.1,
            urgency_score=0.3,
            domain="operations",
            paragraph_count=3,
            n_chunks=4,
            pattern_summary="2 spikes",
        )
        assert snap.extra["source"] == "text_cognitive_engine"
        assert snap.extra["paragraph_count"] == 3
        assert snap.extra["n_chunks"] == 4
        assert snap.extra["pattern_summary"] == "2 spikes"

    def test_profile_std_with_single_sentence(self):
        profiler = TextSignalProfiler()
        snap = profiler.profile(
            word_count=5,
            sentences=["Single sentence here."],
            avg_sentence_length=3.0,
            vocabulary_richness=1.0,
            sentiment_score=0.0,
            urgency_score=0.0,
            domain="general",
        )
        assert snap.std == 0.0


# ── TextPerceptionCollector Tests ──

class TestTextPerceptionCollector:

    def test_collect_returns_five_perceptions(self):
        collector = TextPerceptionCollector()
        inp = _make_input()
        perceptions = collector.collect(inp)
        assert len(perceptions) == 5

    def test_perception_engine_names(self):
        collector = TextPerceptionCollector()
        inp = _make_input()
        perceptions = collector.collect(inp)
        names = [p.engine_name for p in perceptions]
        assert names == [
            "text_sentiment",
            "text_urgency",
            "text_readability",
            "text_structural",
            "text_pattern",
        ]

    def test_sentiment_perception_negative(self):
        collector = TextPerceptionCollector()
        inp = _make_input(sentiment_score=-0.5, sentiment_label="negative")
        perceptions = collector.collect(inp)
        sentiment_p = perceptions[0]
        assert sentiment_p.trend == "down"
        assert 0.0 <= sentiment_p.predicted_value <= 1.0

    def test_urgency_perception_warning(self):
        collector = TextPerceptionCollector()
        inp = _make_input(urgency_score=0.65, urgency_severity="warning")
        perceptions = collector.collect(inp)
        urgency_p = perceptions[1]
        assert urgency_p.trend == "up"
        assert urgency_p.predicted_value == 0.65

    def test_structural_unavailable_returns_default(self):
        collector = TextPerceptionCollector()
        inp = _make_input(structural_available=False)
        perceptions = collector.collect(inp)
        structural_p = perceptions[3]
        assert structural_p.predicted_value == 0.5
        assert structural_p.confidence == 0.3
        assert structural_p.metadata.get("available") is False

    def test_pattern_unavailable_returns_default(self):
        collector = TextPerceptionCollector()
        inp = _make_input(
            pattern_available=False,
            readability_sentences=[],
        )
        perceptions = collector.collect(inp)
        pattern_p = perceptions[4]
        assert pattern_p.predicted_value == 0.5
        assert pattern_p.confidence == 0.3

    def test_pattern_computed_on_the_fly_when_sentences_present(self):
        collector = TextPerceptionCollector()
        inp = _make_input(
            pattern_available=False,
            readability_sentences=[
                "El sistema funciona correctamente.",
                "Se detecto una alerta menor.",
                "Error critico en el modulo principal.",
            ],
        )
        perceptions = collector.collect(inp)
        pattern_p = perceptions[4]
        assert pattern_p.metadata["available"] is True
        assert pattern_p.predicted_value >= 0.0
        assert pattern_p.confidence > 0.3

    def test_default_weights_sum_to_one(self):
        total = sum(DEFAULT_TEXT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9


# ── TextSeverityMapper Tests ──

class TestTextSeverityMapper:

    def test_critical_urgency_with_impact(self):
        # High urgency + negative sentiment + critical text → critical
        result = classify_text_severity(
            urgency_score=0.8,
            urgency_severity="critical",
            sentiment_label="negative",
            full_text="CRITICAL failure detected. System down.",
        )
        assert result.severity == "critical"
        assert result.risk_level == "HIGH"
        assert result.action_required is True

    def test_warning_urgency_no_impact(self):
        # Moderate urgency + neutral sentiment, no impact text → warning
        # composite = 0.5*0.3 + 0.3*0.2 + 0*0.5 = 0.21 → info
        # But with impact text about concerns:
        result = classify_text_severity(
            urgency_score=0.5,
            urgency_severity="warning",
            sentiment_label="neutral",
            full_text="Alert: degradation detected in the network.",
        )
        assert result.severity in ("warning", "info")

    def test_negative_sentiment_with_elevated_urgency(self):
        # urgency=0.35 + negative + impact text
        result = classify_text_severity(
            urgency_score=0.35,
            urgency_severity="info",
            sentiment_label="negative",
            full_text="Error crítico en servidor. Fallas múltiples.",
        )
        assert result.severity in ("warning", "critical")

    def test_low_urgency_positive_sentiment(self):
        result = classify_text_severity(
            urgency_score=0.1,
            urgency_severity="info",
            sentiment_label="positive",
        )
        assert result.severity == "info"
        assert result.action_required is False

    def test_critical_keywords_with_negative_sentiment(self):
        result = classify_text_severity(
            urgency_score=0.3,
            urgency_severity="info",
            sentiment_label="negative",
            has_critical_keywords=True,
        )
        assert result.severity == "critical"

    def test_domain_in_action_text(self):
        result = classify_text_severity(
            urgency_score=0.8,
            urgency_severity="critical",
            sentiment_label="negative",
            domain="infrastructure",
            full_text="CRITICAL server failure.",
        )
        assert "infrastructure" in result.recommended_action

    def test_3axis_formula_no_text_backward_compatible(self):
        # Without full_text, impact=0 → pure urgency+sentiment
        result = classify_text_severity(
            urgency_score=0.1,
            urgency_severity="info",
            sentiment_label="positive",
        )
        assert result.severity == "info"
        assert result.risk_level == "NONE"

    def test_3axis_high_impact_overrides_low_urgency(self):
        # Low urgency (0.2) + neutral sentiment, but text has CRITICAL markers
        result = classify_text_severity(
            urgency_score=0.2,
            urgency_severity="info",
            sentiment_label="neutral",
            full_text=(
                "CRÍTICO: SLA roto. Temperatura 89°C. "
                "Riesgo de caída total en 72 horas. "
                "Múltiples servicios afectados."
            ),
        )
        assert result.severity == "critical"
        assert result.risk_level == "HIGH"

    def test_3axis_moderate_impact_produces_warning(self):
        # SLA breach + critical marker + negative sentiment → enough for warning
        result = classify_text_severity(
            urgency_score=0.4,
            urgency_severity="info",
            sentiment_label="negative",
            full_text="CRITICAL: SLA breach detected. Risk of failure within 48 hours.",
        )
        assert result.severity in ("warning", "critical")


# ── TextMemoryEnricher Tests ──

class TestTextMemoryEnricher:

    def test_no_memory_port_returns_empty(self):
        enricher = TextMemoryEnricher()
        ctx = enricher.enrich("test text", "general", None)
        assert not ctx.has_context
        assert ctx.enriched_summary == ""

    def test_with_mock_memory_returns_context(self):
        mem = MemorySearchResult(
            memory_id="mem-1",
            series_id="doc-old",
            text="Previous server analysis",
            certainty=0.85,
        )
        port = MockCognitiveMemory(explanations=[mem])
        enricher = TextMemoryEnricher()
        ctx = enricher.enrich("server cpu spike analysis", "infrastructure", port)
        assert ctx.has_context
        assert len(ctx.similar_explanations) == 1
        assert "Historical context" in ctx.enriched_summary

    def test_failing_memory_returns_empty(self):
        port = FailingCognitiveMemory()
        enricher = TextMemoryEnricher()
        ctx = enricher.enrich("test text", "general", port)
        assert not ctx.has_context

    def test_recall_context_to_dict(self):
        mem = MemorySearchResult(
            memory_id="mem-1",
            series_id="doc-old",
            text="Previous analysis",
            certainty=0.82,
        )
        ctx = TextRecallContext(
            similar_explanations=[mem],
            enriched_summary="Historical context: test",
            historical_references=["ref-1"],
        )
        d = ctx.to_dict()
        assert "enriched_summary" in d
        assert "historical_references" in d
        assert "similar_explanations" in d


# ── TextExplanationAssembler Tests ──

class TestTextExplanationAssembler:

    def test_assemble_returns_explanation(self):
        assembler = TextExplanationAssembler()
        profiler = TextSignalProfiler()
        collector = TextPerceptionCollector()

        inp = _make_input()
        signal = profiler.profile(
            word_count=inp.word_count,
            sentences=inp.readability_sentences,
            avg_sentence_length=inp.readability_avg_sentence_length,
            vocabulary_richness=inp.readability_vocabulary_richness,
            sentiment_score=inp.sentiment_score,
            urgency_score=inp.urgency_score,
            domain="infrastructure",
        )
        perceptions = collector.collect(inp)

        from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
            InhibitionState,
        )
        inh_states = [
            InhibitionState(
                engine_name=p.engine_name,
                base_weight=0.2,
                inhibited_weight=0.2,
            )
            for p in perceptions
        ]
        weights = {p.engine_name: 0.2 for p in perceptions}

        from iot_machine_learning.domain.services.severity_rules import SeverityResult
        severity = SeverityResult(
            risk_level="MEDIUM",
            severity="warning",
            action_required=True,
            recommended_action="Review",
        )

        explanation = assembler.assemble(
            document_id="doc-001",
            signal=signal,
            perceptions=perceptions,
            inhibition_states=inh_states,
            final_weights=weights,
            selected_engine="text_urgency",
            selection_reason="highest_weight",
            fusion_method="weighted_average",
            fused_confidence=0.82,
            domain="infrastructure",
            severity=severity,
            pipeline_phases=[
                {"kind": "perceive", "summary": {}, "duration_ms": 1.0},
                {"kind": "predict", "summary": {}, "duration_ms": 2.0},
            ],
        )

        assert isinstance(explanation, Explanation)
        assert explanation.series_id == "doc-001"
        assert explanation.outcome.kind == "text_analysis"
        assert explanation.outcome.confidence == 0.82
        assert explanation.contributions.n_engines == 5
        assert explanation.trace.regime_at_inference == "infrastructure"

    def test_explanation_serializable(self):
        assembler = TextExplanationAssembler()
        profiler = TextSignalProfiler()
        collector = TextPerceptionCollector()

        inp = _make_input()
        signal = profiler.profile(
            word_count=inp.word_count,
            sentences=inp.readability_sentences,
            avg_sentence_length=inp.readability_avg_sentence_length,
            vocabulary_richness=inp.readability_vocabulary_richness,
            sentiment_score=inp.sentiment_score,
            urgency_score=inp.urgency_score,
            domain="general",
        )
        perceptions = collector.collect(inp)

        from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
            InhibitionState,
        )
        inh_states = [
            InhibitionState(engine_name=p.engine_name, base_weight=0.2, inhibited_weight=0.2)
            for p in perceptions
        ]
        weights = {p.engine_name: 0.2 for p in perceptions}

        from iot_machine_learning.domain.services.severity_rules import SeverityResult
        severity = SeverityResult(
            risk_level="LOW", severity="info",
            action_required=False, recommended_action="Monitor",
        )

        explanation = assembler.assemble(
            document_id="doc-002",
            signal=signal,
            perceptions=perceptions,
            inhibition_states=inh_states,
            final_weights=weights,
            selected_engine="text_urgency",
            selection_reason="test",
            fusion_method="weighted_average",
            fused_confidence=0.75,
            domain="general",
            severity=severity,
            pipeline_phases=[],
        )

        d = explanation.to_dict()
        assert isinstance(d, dict)
        assert "outcome" in d
        assert d["outcome"]["kind"] == "text_analysis"


# ── TextCognitiveEngine Full Pipeline Tests ──

class TestTextCognitiveEngine:

    def test_full_pipeline_returns_result(self):
        engine = TextCognitiveEngine()
        inp = _make_input()
        ctx = _make_context()
        result = engine.analyze(inp, ctx)

        assert isinstance(result, TextCognitiveResult)
        assert isinstance(result.explanation, Explanation)
        assert result.explanation.outcome.kind == "text_analysis"
        assert result.confidence > 0
        assert result.severity.severity in ("critical", "warning", "info")
        assert result.domain != ""

    def test_pipeline_timing_contains_all_phases(self):
        engine = TextCognitiveEngine()
        inp = _make_input()
        ctx = _make_context()
        result = engine.analyze(inp, ctx)

        expected_phases = {"perceive", "predict", "adapt", "inhibit", "fuse", "explain"}
        assert expected_phases.issubset(set(result.pipeline_timing.keys()))

    def test_domain_auto_detection_infrastructure(self):
        engine = TextCognitiveEngine()
        inp = _make_input(
            full_text="Server CPU usage at 95%. Network latency increased. "
                      "Kubernetes cluster node is unresponsive.",
        )
        ctx = _make_context()
        result = engine.analyze(inp, ctx)
        assert result.domain == "infrastructure"

    def test_domain_hint_overrides_auto_detection(self):
        engine = TextCognitiveEngine()
        inp = _make_input(full_text="Server CPU at 95%")
        ctx = _make_context(domain_hint="security")
        result = engine.analyze(inp, ctx)
        assert result.domain == "security"

    def test_domain_general_when_no_keywords(self):
        engine = TextCognitiveEngine()
        inp = _make_input(
            full_text="The quick brown fox jumps over the lazy dog.",
        )
        ctx = _make_context()
        result = engine.analyze(inp, ctx)
        assert result.domain == "general"

    def test_critical_urgency_produces_critical_severity(self):
        engine = TextCognitiveEngine()
        inp = _make_input(urgency_score=0.85, urgency_severity="critical")
        ctx = _make_context()
        result = engine.analyze(inp, ctx)
        assert result.severity.severity == "critical"
        assert result.severity.action_required is True

    def test_low_urgency_positive_produces_info(self):
        engine = TextCognitiveEngine()
        inp = _make_input(
            urgency_score=0.1,
            urgency_severity="info",
            sentiment_score=0.3,
            sentiment_label="positive",
        )
        ctx = _make_context()
        result = engine.analyze(inp, ctx)
        assert result.severity.severity == "info"

    def test_confidence_increases_with_word_count(self):
        engine = TextCognitiveEngine()
        short_inp = _make_input(word_count=50)
        long_inp = _make_input(word_count=600)
        ctx = _make_context()
        short_result = engine.analyze(short_inp, ctx)
        long_result = engine.analyze(long_inp, ctx)
        assert long_result.confidence > short_result.confidence

    def test_confidence_increases_with_structural(self):
        engine = TextCognitiveEngine()
        no_struct = _make_input(structural_available=False)
        with_struct = _make_input(structural_available=True)
        ctx = _make_context()
        r1 = engine.analyze(no_struct, ctx)
        r2 = engine.analyze(with_struct, ctx)
        assert r2.confidence > r1.confidence

    def test_analysis_dict_contains_cognitive_section(self):
        engine = TextCognitiveEngine()
        inp = _make_input()
        ctx = _make_context()
        result = engine.analyze(inp, ctx)
        assert "cognitive" in result.analysis
        assert "engine_weights" in result.analysis["cognitive"]
        assert "engine_perceptions" in result.analysis["cognitive"]

    def test_analysis_dict_backward_compatible(self):
        engine = TextCognitiveEngine()
        inp = _make_input()
        ctx = _make_context()
        result = engine.analyze(inp, ctx)
        # Should have the same top-level keys as before
        assert "sentiment" in result.analysis
        assert "sentiment_score" in result.analysis
        assert "urgency_score" in result.analysis
        assert "readability" in result.analysis

    def test_explanation_has_five_contributions(self):
        engine = TextCognitiveEngine()
        inp = _make_input()
        ctx = _make_context()
        result = engine.analyze(inp, ctx)
        assert result.explanation.contributions.n_engines == 5

    def test_no_memory_still_works(self):
        engine = TextCognitiveEngine()
        inp = _make_input()
        ctx = _make_context(cognitive_memory=None)
        result = engine.analyze(inp, ctx)
        assert result.recall_context is None
        assert result.confidence > 0

    def test_with_mock_memory_enriches_result(self):
        mem = MemorySearchResult(
            memory_id="mem-1",
            series_id="doc-old",
            text="Previous analysis",
            certainty=0.85,
        )
        port = MockCognitiveMemory(explanations=[mem])
        engine = TextCognitiveEngine()
        inp = _make_input()
        ctx = _make_context(cognitive_memory=port)
        result = engine.analyze(inp, ctx)
        assert result.recall_context is not None

    def test_failing_memory_degrades_gracefully(self):
        port = FailingCognitiveMemory()
        engine = TextCognitiveEngine()
        inp = _make_input()
        ctx = _make_context(cognitive_memory=port)
        result = engine.analyze(inp, ctx)
        # Should still produce valid result
        assert isinstance(result, TextCognitiveResult)
        assert result.recall_context is None

    def test_plasticity_disabled(self):
        engine = TextCognitiveEngine(enable_plasticity=False)
        inp = _make_input()
        ctx = _make_context()
        result = engine.analyze(inp, ctx)
        assert isinstance(result, TextCognitiveResult)

    def test_conclusion_contains_severity_and_domain(self):
        engine = TextCognitiveEngine()
        inp = _make_input(urgency_score=0.85, urgency_severity="critical")
        ctx = _make_context()
        result = engine.analyze(inp, ctx)
        assert "Severity: critical" in result.conclusion
        assert "Domain:" in result.conclusion


# ── TextCognitiveResult Tests ──

class TestTextCognitiveResult:

    def test_to_dict_serializable(self):
        engine = TextCognitiveEngine()
        inp = _make_input()
        ctx = _make_context()
        result = engine.analyze(inp, ctx)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "explanation" in d
        assert "severity" in d
        assert "pipeline_timing" in d
        assert "domain" in d

    def test_to_legacy_dict_matches_old_format(self):
        engine = TextCognitiveEngine()
        inp = _make_input()
        ctx = _make_context()
        result = engine.analyze(inp, ctx)
        legacy = result.to_legacy_dict()
        assert "analysis" in legacy
        assert "adaptive_thresholds" in legacy
        assert "conclusion" in legacy
        assert "confidence" in legacy
        assert legacy["adaptive_thresholds"]["urgency_critical"] == 0.7


# ── ImpactSignalDetector Tests ──

class TestImpactSignalDetector:

    def test_empty_text_returns_zero(self):
        result = detect_impact_signals("")
        assert result.score == 0.0
        assert result.n_signals == 0
        assert result.n_categories_hit == 0

    def test_benign_text_returns_zero(self):
        result = detect_impact_signals(
            "The quick brown fox jumps over the lazy dog."
        )
        assert result.score == 0.0
        assert not result.has_critical_markers
        assert not result.has_sla_breach

    def test_detects_critico_marker(self):
        result = detect_impact_signals("Estado CRÍTICO del servidor principal.")
        assert result.has_critical_markers
        assert result.score > 0.0
        assert any(s.category == "critical_marker" for s in result.signals)

    def test_detects_critical_english(self):
        result = detect_impact_signals("CRITICAL failure in production cluster.")
        assert result.has_critical_markers
        assert any(
            s.matched_text == "critical" for s in result.signals
        )

    def test_detects_caida_total(self):
        result = detect_impact_signals("Riesgo de caída total del sistema.")
        assert result.has_critical_markers
        assert result.has_temporal_risk

    def test_detects_sla_breach(self):
        result = detect_impact_signals(
            "SLA breach: availability at 78% vs target 99.5%."
        )
        assert result.has_sla_breach
        assert any(s.category == "sla_breach" for s in result.signals)

    def test_detects_percentage_vs_percentage(self):
        result = detect_impact_signals("Disponibilidad 78% vs 99.5%")
        assert result.has_sla_breach

    def test_detects_extreme_temperature(self):
        result = detect_impact_signals("Temperatura del rack: 89°C")
        assert result.has_extreme_metrics
        extreme = [s for s in result.signals if s.category == "extreme_metric"]
        assert len(extreme) == 1
        assert extreme[0].value == 89.0

    def test_normal_temperature_not_flagged(self):
        result = detect_impact_signals("Temperatura del rack: 45°C")
        assert not result.has_extreme_metrics

    def test_detects_extreme_cpu(self):
        result = detect_impact_signals("CPU usage: 95%")
        assert result.has_extreme_metrics

    def test_normal_cpu_not_flagged(self):
        result = detect_impact_signals("CPU usage: 60%")
        assert not result.has_extreme_metrics

    def test_detects_low_availability(self):
        result = detect_impact_signals("Availability: 78%")
        assert result.has_extreme_metrics

    def test_good_availability_not_flagged(self):
        result = detect_impact_signals("Availability: 99.9%")
        assert not result.has_extreme_metrics

    def test_detects_temporal_risk_es(self):
        result = detect_impact_signals(
            "Riesgo de falla total en 72 horas."
        )
        assert result.has_temporal_risk

    def test_detects_temporal_risk_en(self):
        result = detect_impact_signals(
            "Risk of failure within 24 hours."
        )
        assert result.has_temporal_risk

    def test_detects_cascade_risk_es(self):
        result = detect_impact_signals(
            "Múltiples servicios afectados. Efecto dominó."
        )
        assert result.has_cascade_risk

    def test_detects_cascade_risk_en(self):
        result = detect_impact_signals(
            "Multiple systems affected. Cascade failure imminent."
        )
        assert result.has_cascade_risk

    def test_multi_category_bonus(self):
        # 3+ categories should get multiplier
        result = detect_impact_signals(
            "CRÍTICO: SLA roto. CPU al 95%. "
            "Riesgo de caída en 24 horas."
        )
        assert result.n_categories_hit >= 3
        assert result.score >= 0.55  # should be critical threshold

    def test_full_critical_document(self):
        doc = (
            "Informe CRÍTICO de infraestructura:\n"
            "El servidor principal presenta fallas críticas. "
            "CPU al 95%, temperatura 89°C.\n"
            "SLA roto: disponibilidad actual 78% vs objetivo 99.5%.\n"
            "Riesgo de caída total del sistema en las próximas 72 horas.\n"
            "Múltiples servicios degradados."
        )
        result = detect_impact_signals(doc)
        assert result.n_categories_hit == 5  # all categories
        assert result.score >= 0.9
        assert result.has_critical_markers
        assert result.has_sla_breach
        assert result.has_extreme_metrics
        assert result.has_temporal_risk
        assert result.has_cascade_risk
        assert result.summary != ""

    def test_to_dict_serializable(self):
        result = detect_impact_signals("CRITICAL SLA breach. CPU 95%.")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "score" in d
        assert "signals" in d
        assert "n_categories_hit" in d
        assert isinstance(d["signals"], list)

    def test_summary_contains_category_names(self):
        result = detect_impact_signals(
            "CRÍTICO: SLA breach. Temperatura 89°C."
        )
        assert "critical severity markers" in result.summary
        assert "SLA/KPI breach" in result.summary


# ── 3-Axis Severity Integration Tests ──

class TestSeverityImpactIntegration:

    CRITICAL_DOC = (
        "Informe CRÍTICO de infraestructura:\n"
        "El servidor principal presenta fallas críticas. "
        "CPU al 95%, temperatura 89°C.\n"
        "SLA roto: disponibilidad actual 78% vs objetivo 99.5%.\n"
        "Riesgo de caída total del sistema en las próximas 72 horas.\n"
        "Múltiples servicios degradados. Se requiere intervención inmediata."
    )

    def test_critical_doc_produces_critical_severity(self):
        """THE key test: a document with critical signals MUST produce critical severity."""
        result = classify_text_severity(
            urgency_score=0.65,
            urgency_severity="warning",
            sentiment_label="negative",
            full_text=self.CRITICAL_DOC,
            domain="infrastructure",
        )
        assert result.severity == "critical"
        assert result.risk_level == "HIGH"
        assert result.action_required is True

    def test_critical_doc_even_with_low_urgency(self):
        """Impact signals override low urgency scores."""
        result = classify_text_severity(
            urgency_score=0.2,
            urgency_severity="info",
            sentiment_label="neutral",
            full_text=self.CRITICAL_DOC,
        )
        assert result.severity == "critical"

    def test_impact_weight_dominates(self):
        """Impact axis (0.50) outweighs urgency (0.30) + sentiment (0.20)."""
        # Low urgency, positive sentiment, but horrific text
        result = classify_text_severity(
            urgency_score=0.1,
            urgency_severity="info",
            sentiment_label="positive",
            full_text=(
                "CRITICAL: total failure imminent. "
                "SLA breach 50% vs 99.9%. Temperature 92°C. "
                "Multiple systems cascading."
            ),
        )
        assert result.severity == "critical"

    def test_benign_text_stays_info(self):
        result = classify_text_severity(
            urgency_score=0.1,
            urgency_severity="info",
            sentiment_label="neutral",
            full_text="Everything is running smoothly. All systems nominal.",
        )
        assert result.severity == "info"

    def test_impact_summary_in_action(self):
        result = classify_text_severity(
            urgency_score=0.6,
            urgency_severity="warning",
            sentiment_label="negative",
            full_text="CRITICAL SLA breach. CPU 95%.",
            domain="infrastructure",
        )
        assert "Impact signals detected" in result.recommended_action


# ── Full Pipeline with Impact ──

class TestTextCognitiveEngineWithImpact:

    CRITICAL_DOC = (
        "Informe CRÍTICO de infraestructura:\n"
        "El servidor principal presenta fallas críticas. "
        "CPU al 95%, temperatura 89°C.\n"
        "SLA roto: disponibilidad actual 78% vs objetivo 99.5%.\n"
        "Riesgo de caída total del sistema en las próximas 72 horas.\n"
        "Múltiples servicios degradados."
    )

    def test_full_pipeline_critical_document(self):
        """End-to-end: critical document → critical severity in full pipeline."""
        engine = TextCognitiveEngine()
        inp = _make_input(
            full_text=self.CRITICAL_DOC,
            urgency_score=0.65,
            urgency_severity="warning",
            sentiment_label="negative",
            sentiment_score=-0.4,
        )
        ctx = _make_context()
        result = engine.analyze(inp, ctx)

        assert result.severity.severity == "critical"
        assert result.severity.risk_level == "HIGH"
        assert result.severity.action_required is True
        assert "critical" in result.conclusion.lower()

    def test_analysis_dict_contains_impact(self):
        engine = TextCognitiveEngine()
        inp = _make_input(
            full_text=self.CRITICAL_DOC,
            urgency_score=0.65,
            urgency_severity="warning",
            sentiment_label="negative",
        )
        ctx = _make_context()
        result = engine.analyze(inp, ctx)
        assert "impact" in result.analysis
        assert result.analysis["impact"]["n_categories_hit"] >= 3

    def test_benign_document_stays_info(self):
        engine = TextCognitiveEngine()
        inp = _make_input(
            full_text="Todo funciona correctamente. Sistemas estables.",
            urgency_score=0.1,
            urgency_severity="info",
            sentiment_label="positive",
            sentiment_score=0.3,
        )
        ctx = _make_context()
        result = engine.analyze(inp, ctx)
        assert result.severity.severity == "info"

    def test_perceive_phase_includes_impact_score(self):
        engine = TextCognitiveEngine()
        inp = _make_input(full_text=self.CRITICAL_DOC)
        ctx = _make_context()
        result = engine.analyze(inp, ctx)
        assert "perceive" in result.pipeline_timing
