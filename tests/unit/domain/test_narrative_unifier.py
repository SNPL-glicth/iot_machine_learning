"""Tests for NarrativeUnifier domain service.

Covers narrative unification rules:
- severity: maximum among sources
- confidence: minimum among sources
- primary_verdict: from highest severity source (tie: prediction_explanation)
- contradiction detection between sources
- suppression of unavailable sources
"""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.services.narrative_unifier import (
    NarrativeUnifier,
    NarrativeSource,
)
from iot_machine_learning.domain.entities.results.unified_narrative import UnifiedNarrative


class TestNarrativeUnifierBasics:
    """Basic construction and sanity checks."""

    def test_unifier_construction(self):
        """NarrativeUnifier can be instantiated."""
        unifier = NarrativeUnifier()
        assert unifier is not None

    def test_unified_narrative_dataclass(self):
        """UnifiedNarrative is a proper frozen dataclass."""
        result = UnifiedNarrative(
            primary_verdict="system stable",
            severity="NORMAL",
            confidence=0.85,
            contradictions=[],
            sources_used=["prediction_explanation"],
            suppressed=[],
        )
        assert result.primary_verdict == "system stable"
        assert result.severity == "NORMAL"
        assert result.confidence == pytest.approx(0.85)
        assert result.contradictions == []
        assert result.sources_used == ["prediction_explanation"]


class TestNoSourcesAvailable:
    """When no sources available."""

    def test_all_sources_none(self):
        """All sources None → minimal result."""
        unifier = NarrativeUnifier()
        result = unifier.unify(
            prediction_explanation=None,
            anomaly_narrative=None,
            text_narrative=None,
        )
        
        assert result.primary_verdict == "no narrative sources available"
        assert result.severity == "UNKNOWN"
        assert result.confidence == pytest.approx(0.0)
        assert result.sources_used == []
        assert len(result.suppressed) == 3  # All three sources suppressed

    def test_all_sources_empty(self):
        """All sources empty/invalid → suppressed."""
        unifier = NarrativeUnifier()
        result = unifier.unify(
            prediction_explanation={},
            anomaly_narrative={"verdict": "", "severity": "WARNING"},
            text_narrative=None,
        )
        
        assert result.severity == "UNKNOWN"
        assert len(result.suppressed) >= 2


class TestContradictionDetection:
    """Contradiction detection between sources."""

    def test_prediction_stable_anomaly_critical_contradiction(self):
        """prediction stable + anomaly critical → contradiction detected."""
        unifier = NarrativeUnifier()
        result = unifier.unify(
            prediction_explanation={
                "verdict": "system stable",
                "severity": "NORMAL",
                "confidence": 0.9,
            },
            anomaly_narrative={
                "verdict": "critical anomaly detected",
                "severity": "CRITICAL",
                "confidence": 0.8,
            },
            text_narrative=None,
        )
        
        # Should detect contradiction
        assert len(result.contradictions) >= 1
        assert any("prediction" in c and "anomaly" in c for c in result.contradictions)
        # Severity should be max (CRITICAL)
        assert result.severity == "CRITICAL"

    def test_text_prediction_severity_gap_contradiction(self):
        """text severity differs from prediction by >1 level → contradiction."""
        unifier = NarrativeUnifier()
        result = unifier.unify(
            prediction_explanation={
                "verdict": "system stable",
                "severity": "NORMAL",
                "confidence": 0.9,
            },
            anomaly_narrative=None,
            text_narrative={
                "verdict": "text analysis critical",
                "severity": "CRITICAL",
                "confidence": 0.7,
            },
        )
        
        # Should detect gap contradiction
        assert len(result.contradictions) >= 1
        assert any("gap" in c for c in result.contradictions)


class TestCoherentSources:
    """All sources coherent → no contradictions."""

    def test_all_sources_agree(self):
        """All sources agree → contradictions empty."""
        unifier = NarrativeUnifier()
        result = unifier.unify(
            prediction_explanation={
                "verdict": "warning condition",
                "severity": "WARNING",
                "confidence": 0.85,
            },
            anomaly_narrative={
                "verdict": "anomaly warning",
                "severity": "WARNING",
                "confidence": 0.80,
            },
            text_narrative={
                "verdict": "text warning",
                "severity": "WARNING",
                "confidence": 0.75,
            },
        )
        
        assert len(result.contradictions) == 0
        assert result.severity == "WARNING"


class TestSourceSuppression:
    """Unavailable sources are suppressed."""

    def test_none_source_suppressed(self):
        """None source → suppressed with not_available."""
        unifier = NarrativeUnifier()
        result = unifier.unify(
            prediction_explanation={
                "verdict": "system stable",
                "severity": "NORMAL",
                "confidence": 0.9,
            },
            anomaly_narrative=None,
            text_narrative={
                "verdict": "text analysis",
                "severity": "INFO",
                "confidence": 0.7,
            },
        )
        
        assert any("anomaly_narrative:not_available" in s for s in result.suppressed)
        assert "prediction_explanation" in result.sources_used
        assert "text_narrative" in result.sources_used


class TestSeverityRules:
    """Severity unification rules."""

    def test_maximum_severity_propagates(self):
        """Maximum severity among sources is used."""
        unifier = NarrativeUnifier()
        result = unifier.unify(
            prediction_explanation={
                "verdict": "normal",
                "severity": "NORMAL",
                "confidence": 0.9,
            },
            anomaly_narrative={
                "verdict": "warning",
                "severity": "WARNING",
                "confidence": 0.8,
            },
            text_narrative={
                "verdict": "critical",
                "severity": "CRITICAL",
                "confidence": 0.7,
            },
        )
        
        # Should take max severity (CRITICAL)
        assert result.severity == "CRITICAL"

    def test_critical_higher_than_warning(self):
        """CRITICAL > WARNING."""
        unifier = NarrativeUnifier()
        result = unifier.unify(
            prediction_explanation={
                "verdict": "warning",
                "severity": "WARNING",
                "confidence": 0.9,
            },
            anomaly_narrative={
                "verdict": "critical",
                "severity": "CRITICAL",
                "confidence": 0.5,
            },
            text_narrative=None,
        )
        
        assert result.severity == "CRITICAL"


class TestConfidenceRules:
    """Confidence unification rules."""

    def test_minimum_confidence_used(self):
        """Minimum confidence among sources is used."""
        unifier = NarrativeUnifier()
        result = unifier.unify(
            prediction_explanation={
                "verdict": "normal",
                "severity": "NORMAL",
                "confidence": 0.9,
            },
            anomaly_narrative={
                "verdict": "warning",
                "severity": "WARNING",
                "confidence": 0.6,
            },
            text_narrative={
                "verdict": "info",
                "severity": "INFO",
                "confidence": 0.8,
            },
        )
        
        # Should take min confidence (0.6)
        assert result.confidence == pytest.approx(0.6)

    def test_single_source_confidence_preserved(self):
        """Single source → its confidence preserved."""
        unifier = NarrativeUnifier()
        result = unifier.unify(
            prediction_explanation={
                "verdict": "stable",
                "severity": "NORMAL",
                "confidence": 0.75,
            },
            anomaly_narrative=None,
            text_narrative=None,
        )
        
        assert result.confidence == pytest.approx(0.75)


class TestPrimaryVerdictSelection:
    """Primary verdict selection rules."""

    def test_highest_severity_source_wins(self):
        """Source with highest severity provides verdict."""
        unifier = NarrativeUnifier()
        result = unifier.unify(
            prediction_explanation={
                "verdict": "prediction says normal",
                "severity": "NORMAL",
                "confidence": 0.9,
            },
            anomaly_narrative={
                "verdict": "anomaly says critical",
                "severity": "CRITICAL",
                "confidence": 0.5,
            },
            text_narrative=None,
        )
        
        # anomaly_narrative has CRITICAL, so it wins
        assert result.primary_verdict == "anomaly says critical"

    def test_tie_prediction_wins(self):
        """Tie in severity → prediction_explanation wins."""
        unifier = NarrativeUnifier()
        result = unifier.unify(
            prediction_explanation={
                "verdict": "prediction warning",
                "severity": "WARNING",
                "confidence": 0.9,
            },
            anomaly_narrative={
                "verdict": "anomaly warning",
                "severity": "WARNING",
                "confidence": 0.8,
            },
            text_narrative=None,
        )
        
        # Tie in severity, prediction_explanation wins
        assert result.primary_verdict == "prediction warning"


class TestSingleSource:
    """Single source available scenarios."""

    def test_only_prediction_available(self):
        """Only prediction available → unifies with that."""
        unifier = NarrativeUnifier()
        result = unifier.unify(
            prediction_explanation={
                "verdict": "single prediction",
                "severity": "WARNING",
                "confidence": 0.85,
            },
            anomaly_narrative=None,
            text_narrative=None,
        )
        
        assert result.primary_verdict == "single prediction"
        assert result.severity == "WARNING"
        assert result.confidence == pytest.approx(0.85)
        assert result.sources_used == ["prediction_explanation"]
        assert len(result.suppressed) == 2

    def test_only_anomaly_available(self):
        """Only anomaly available → unifies with that."""
        unifier = NarrativeUnifier()
        result = unifier.unify(
            prediction_explanation=None,
            anomaly_narrative={
                "verdict": "single anomaly",
                "severity": "CRITICAL",
                "confidence": 0.75,
            },
            text_narrative=None,
        )
        
        assert result.primary_verdict == "single anomaly"
        assert result.severity == "CRITICAL"
        assert result.sources_used == ["anomaly_narrative"]
