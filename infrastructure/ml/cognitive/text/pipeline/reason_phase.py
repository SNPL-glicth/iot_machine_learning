"""Text reason phase: inhibit unreliable engines, fuse, classify severity."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

from iot_machine_learning.infrastructure.ml.cognitive.inhibition import InhibitionGate
from iot_machine_learning.infrastructure.ml.cognitive.fusion import WeightedFusion
from iot_machine_learning.infrastructure.ml.cognitive.text.perception_collector import DEFAULT_TEXT_WEIGHTS
from iot_machine_learning.infrastructure.ml.cognitive.text.severity_mapper import classify_text_severity

logger = logging.getLogger(__name__)

_PLASTICITY_AVAILABLE = True
try:
    from iot_machine_learning.infrastructure.ml.cognitive.plasticity import PlasticityTracker
except (ImportError, ModuleNotFoundError):
    PlasticityTracker = None  # type: ignore[assignment,misc]


class TextReasonPhase:
    """Phase 4: Inhibit unreliable engines, fuse, classify severity."""

    def __init__(self, enable_plasticity: bool = True) -> None:
        self._inhibition = InhibitionGate()
        self._fusion = WeightedFusion()
        self._plasticity = (
            PlasticityTracker() if enable_plasticity and PlasticityTracker is not None
            else None
        )

    def execute(
        self,
        perceptions: List[Any],
        domain: str,
        document_id: str,
        urgency_score: float,
        urgency_severity: str,
        sentiment_label: str,
        full_text: str,
        impact_result: Any,
        timing: Dict[str, float],
    ) -> Tuple[float, float, str, Dict[str, float], str, str, Any, Dict[str, Any]]:
        """Execute reason phase.

        Args:
            perceptions: List of EnginePerception objects
            domain: Classified domain
            document_id: Document identifier
            urgency_score: Urgency score
            urgency_severity: Urgency severity
            sentiment_label: Sentiment label
            full_text: Complete document text
            impact_result: Impact detection result
            timing: Pipeline timing dict

        Returns:
            Tuple of (fused_val, fused_conf, fused_trend, final_weights, selected, reason, severity, phase_summaries)
        """
        phase_summaries: List[Dict[str, Any]] = []

        # Adapt: get base weights from plasticity or defaults
        t0 = time.monotonic()
        engine_names = [p.engine_name for p in perceptions]
        base_weights = self._get_base_weights(domain, engine_names)
        adapted = self._plasticity is not None and self._plasticity.has_history(domain)
        adapt_ms = (time.monotonic() - t0) * 1000
        timing["adapt"] = adapt_ms
        phase_summaries.append({
            "kind": "adapt",
            "summary": {"regime": domain, "adapted": adapted},
            "duration_ms": adapt_ms,
        })

        # Inhibit: suppress unreliable sub-analyzers
        t0 = time.monotonic()
        inh_states = self._inhibition.compute(
            perceptions, base_weights, series_id=document_id,
        )
        inhibit_ms = (time.monotonic() - t0) * 1000
        timing["inhibit"] = inhibit_ms
        phase_summaries.append({
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
            urgency_score=urgency_score,
            urgency_severity=urgency_severity,
            sentiment_label=sentiment_label,
            has_critical_keywords=urgency_severity == "critical",
            domain=domain,
            full_text=full_text,
            impact_result=impact_result,
        )

        fuse_ms = (time.monotonic() - t0) * 1000
        timing["fuse"] = fuse_ms
        phase_summaries.append({
            "kind": "fuse",
            "summary": {
                "selected_engine": selected,
                "fused_confidence": round(fused_conf, 4),
                "severity": severity.severity,
            },
            "duration_ms": fuse_ms,
        })

        return fused_val, fused_conf, fused_trend, final_weights, selected, reason, severity, phase_summaries

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
