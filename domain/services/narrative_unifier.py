"""Narrative unifier — reconciles multiple narrative sources.

Combines outputs from ExplanationBuilder, anomaly_narrator, and TextCognitiveEngine
into a single coherent narrative. Detects contradictions and applies severity/confidence
unification rules.

Pure domain logic — stateless, no I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ...domain.entities.results.unified_narrative import UnifiedNarrative


class NarrativeSource:
    """Wrapper for a narrative source input."""

    def __init__(
        self,
        name: str,
        verdict: Optional[str],
        severity: str,
        confidence: float,
    ):
        self.name = name
        self.verdict = verdict
        self.severity = severity
        self.confidence = confidence
        self.available = verdict is not None and verdict.strip() != ""


class NarrativeUnifier:
    """Unifies multiple narrative sources into a single coherent narrative.

    Stateless domain service — reconciles potentially conflicting narratives
    from different cognitive components.

    Unification rules:
    - severity: maximum among all sources (CRITICAL > WARNING > INFO > UNKNOWN)
    - confidence: minimum among all sources (system is as weak as weakest link)
    - primary_verdict: from source with highest severity,
                       tie-breaker: prediction_explanation
    - contradictions detected when:
      - prediction says "stable" and anomaly says "critical"
      - text severity differs from prediction by more than one level
    - unavailable sources added to suppressed list
    """

    # Severity ranking (higher = more severe)
    SEVERITY_RANK: Dict[str, int] = {
        "UNKNOWN": 0,
        "INFO": 1,
        "NORMAL": 1,
        "OK": 1,
        "WARNING": 2,
        "WARN": 2,
        "CRITICAL": 3,
        "ERROR": 3,
        "ALERT": 3,
    }

    def unify(
        self,
        prediction_explanation: Optional[Dict],
        anomaly_narrative: Optional[Dict],
        text_narrative: Optional[Dict],
    ) -> "UnifiedNarrative":
        """Unify multiple narrative sources.

        Args:
            prediction_explanation: Dict with 'verdict', 'severity', 'confidence'
            anomaly_narrative: Dict with 'verdict', 'severity', 'confidence'
            text_narrative: Dict with 'verdict', 'severity', 'confidence'

        Returns:
            UnifiedNarrative with reconciled information
        """
        from ...domain.entities.results.unified_narrative import UnifiedNarrative

        # Create source wrappers
        sources = [
            NarrativeSource(
                "prediction_explanation",
                self._extract_verdict(prediction_explanation),
                self._extract_severity(prediction_explanation),
                self._extract_confidence(prediction_explanation),
            ),
            NarrativeSource(
                "anomaly_narrative",
                self._extract_verdict(anomaly_narrative),
                self._extract_severity(anomaly_narrative),
                self._extract_confidence(anomaly_narrative),
            ),
            NarrativeSource(
                "text_narrative",
                self._extract_verdict(text_narrative),
                self._extract_severity(text_narrative),
                self._extract_confidence(text_narrative),
            ),
        ]

        # Separate available and suppressed sources
        available_sources = [s for s in sources if s.available]
        suppressed = [
            f"{s.name}:not_available" for s in sources if not s.available
        ]

        # If no sources available, return minimal result
        if not available_sources:
            return UnifiedNarrative(
                primary_verdict="no narrative sources available",
                severity="UNKNOWN",
                confidence=0.0,
                contradictions=[],
                sources_used=[],
                suppressed=suppressed,
            )

        # Compute unified severity (max) and confidence (min)
        unified_severity = self._max_severity(
            [s.severity for s in available_sources]
        )
        unified_confidence = min(s.confidence for s in available_sources)

        # Select primary verdict source (highest severity, tie: prediction_explanation)
        primary_source = self._select_primary_source(
            available_sources, unified_severity
        )
        primary_verdict = primary_source.verdict if primary_source else "unknown"

        # Detect contradictions
        contradictions = self._detect_contradictions(available_sources)

        # Build sources_used list
        sources_used = [s.name for s in available_sources]

        return UnifiedNarrative(
            primary_verdict=primary_verdict or "unknown",
            severity=unified_severity,
            confidence=unified_confidence,
            contradictions=contradictions,
            sources_used=sources_used,
            suppressed=suppressed,
        )

    def _extract_verdict(self, source: Optional[Dict]) -> Optional[str]:
        """Extract verdict string from source dict."""
        if source is None:
            return None
        verdict = source.get("verdict") or source.get("primary_verdict")
        if verdict and str(verdict).strip():
            return str(verdict).strip()
        return None

    def _extract_severity(self, source: Optional[Dict]) -> str:
        """Extract severity from source dict, default UNKNOWN."""
        if source is None:
            return "UNKNOWN"
        severity = source.get("severity") or source.get("level")
        if severity:
            sev_str = str(severity).upper().strip()
            if sev_str in self.SEVERITY_RANK:
                return sev_str
        return "UNKNOWN"

    def _extract_confidence(self, source: Optional[Dict]) -> float:
        """Extract confidence from source dict, default 0.0."""
        if source is None:
            return 0.0
        confidence = source.get("confidence")
        if confidence is not None:
            try:
                return float(confidence)
            except (ValueError, TypeError):
                pass
        return 0.0

    def _severity_rank(self, severity: str) -> int:
        """Get numeric rank for severity level."""
        return self.SEVERITY_RANK.get(severity.upper(), 0)

    def _max_severity(self, severities: List[str]) -> str:
        """Get maximum severity from list."""
        if not severities:
            return "UNKNOWN"
        return max(severities, key=lambda s: self._severity_rank(s))

    def _select_primary_source(
        self,
        sources: List[NarrativeSource],
        target_severity: str,
    ) -> Optional[NarrativeSource]:
        """Select primary source with highest severity (tie: prediction_explanation)."""
        target_rank = self._severity_rank(target_severity)

        # Filter sources at target severity
        candidates = [
            s for s in sources
            if self._severity_rank(s.severity) == target_rank
        ]

        if not candidates:
            return sources[0] if sources else None

        # Tie-breaker: prediction_explanation preferred
        for c in candidates:
            if c.name == "prediction_explanation":
                return c

        return candidates[0]

    def _detect_contradictions(
        self,
        sources: List[NarrativeSource],
    ) -> List[str]:
        """Detect contradictions between sources."""
        contradictions: List[str] = []

        # Find prediction and anomaly sources
        pred = next((s for s in sources if s.name == "prediction_explanation"), None)
        anomaly = next((s for s in sources if s.name == "anomaly_narrative"), None)
        text = next((s for s in sources if s.name == "text_narrative"), None)

        # Contradiction 1: prediction stable + anomaly critical
        if pred and anomaly:
            pred_rank = self._severity_rank(pred.severity)
            anomaly_rank = self._severity_rank(anomaly.severity)

            if pred_rank <= 1 and anomaly_rank >= 3:  # stable/normal vs critical
                contradictions.append(
                    f"prediction_{pred.severity.lower()}_vs_anomaly_{anomaly.severity.lower()}"
                )

        # Contradiction 2: text severity differs from prediction by > 1 level
        if pred and text:
            pred_rank = self._severity_rank(pred.severity)
            text_rank = self._severity_rank(text.severity)

            if abs(pred_rank - text_rank) > 1:
                contradictions.append(
                    f"text_prediction_severity_gap:{pred.severity.lower()}_vs_{text.severity.lower()}"
                )

        # Contradiction 3: anomaly and text disagree significantly
        if anomaly and text:
            anomaly_rank = self._severity_rank(anomaly.severity)
            text_rank = self._severity_rank(text.severity)

            if abs(anomaly_rank - text_rank) > 1:
                contradictions.append(
                    f"anomaly_text_severity_gap:{anomaly.severity.lower()}_vs_{text.severity.lower()}"
                )

        return contradictions
