"""Text perceive phase: build signal profile from pre-computed metrics."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from iot_machine_learning.infrastructure.ml.cognitive.text.signal_profiler import TextSignalProfiler
from iot_machine_learning.infrastructure.ml.cognitive.text.impact_detector import detect_impact_signals


# Domain classification keywords (lightweight, ~20 keywords)
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


class TextPerceivePhase:
    """Phase 1: Build signal profile from pre-computed text metrics."""

    def __init__(self) -> None:
        self._profiler = TextSignalProfiler()

    def execute(
        self,
        word_count: int,
        readability_sentences: int,
        readability_avg_sentence_length: float,
        readability_vocabulary_richness: float,
        sentiment_score: float,
        urgency_score: float,
        paragraph_count: int,
        embedded_numeric_count: int,
        pattern_summary: str,
        full_text: str,
        domain_hint: str,
        timing: Dict[str, float],
    ) -> Tuple[str, Any, Any, Dict[str, Any]]:
        """Execute perceive phase.

        Args:
            All text metrics from pre-computed analysis
            full_text: Complete document text for domain classification
            domain_hint: Optional domain hint
            timing: Pipeline timing dict

        Returns:
            Tuple of (domain, signal, impact_result, phase_summary)
        """
        t0 = time.monotonic()
        
        # Domain classification
        domain = self._classify_domain(full_text, domain_hint)
        
        # Approximate chunk count
        n_chunks = len(full_text) // 500 + 1
        
        # Signal profiling
        signal = self._profiler.profile(
            word_count=word_count,
            sentences=readability_sentences,
            avg_sentence_length=readability_avg_sentence_length,
            vocabulary_richness=readability_vocabulary_richness,
            sentiment_score=sentiment_score,
            urgency_score=urgency_score,
            domain=domain,
            paragraph_count=paragraph_count,
            n_chunks=n_chunks,
            embedded_numeric_count=embedded_numeric_count,
            pattern_summary=pattern_summary,
        )

        # Impact signal detection
        impact_result = detect_impact_signals(full_text)

        perceive_ms = (time.monotonic() - t0) * 1000
        timing["perceive"] = perceive_ms
        
        phase_summary = {
            "kind": "perceive",
            "summary": {
                "word_count": word_count,
                "domain": domain,
                "impact_score": impact_result.score,
                "impact_categories_hit": impact_result.n_categories_hit,
            },
            "duration_ms": perceive_ms,
        }

        return domain, signal, impact_result, phase_summary

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
