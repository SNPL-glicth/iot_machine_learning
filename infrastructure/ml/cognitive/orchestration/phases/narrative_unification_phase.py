"""Narrative Unification Phase — 3 real sources with agreement/contradiction.

Sources:
  1. ExplainPhase (prediction narrative from ctx.explanation)
  2. CausalPhase   (causal narrative from ctx.causal_events)
  3. AnomalyDomainService integration (anomaly narrative)

When source 1 and 3 agree on severity → confidence boost +0.1.
When they contradict → confidence penalty -0.1 and contradiction flag.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from . import PipelineContext

from iot_machine_learning.domain.services.cognitive.narrative_unifier import NarrativeUnifier

logger = logging.getLogger(__name__)


def _severity_from_z(z_score: float) -> str:
    """Map z‑score anomaly magnitude to severity."""
    if abs(z_score) > 4.0:
        return "CRITICAL"
    if abs(z_score) > 3.0:
        return "WARNING"
    if abs(z_score) > 2.5:
        return "WARN"
    return "NORMAL"


def _anomaly_narrative_from_context(ctx: PipelineContext) -> Optional[Dict[str, Any]]:
    """Build anomaly narrative from the AnomalyDomainService integration.

    Uses ctx.effective values: z‑score, consecutive_anomalies,
    and max_action to produce a severity assessment.
    """
    z = getattr(ctx.profile, "z_score", 0.0) if ctx.profile else 0.0
    severity = _severity_from_z(z)
    cons = getattr(ctx, "consecutive_anomalies", 0)
    if cons >= 3:
        severity = "CRITICAL"
    elif cons >= 2 and severity == "NORMAL":
        severity = "WARNING"
    return {
        "verdict": f"anomaly_z_{abs(z):.2f}_consecutive_{cons}",
        "severity": severity,
        "confidence": ctx.fused_confidence or 0.5,
    }


def _causal_narrative_from_events(
    events: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build causal narrative from CausalPhase events."""
    if not events:
        return None
    # Determine max severity from event count and recency
    n = len(events)
    if n >= 3:
        sev = "CRITICAL"
    elif n >= 2:
        sev = "WARNING"
    else:
        sev = "WARN"
    # Confidence decays with count but stays meaningful
    conf = max(0.3, 0.9 - n * 0.1)
    return {
        "verdict": f"causal_chain:{n}_events_from_"
                   f"{events[0]['preceding_param']}_to_{events[-1]['current_param']}",
        "severity": sev,
        "confidence": round(conf, 3),
    }


def _causal_severity(events: List[Dict[str, Any]]) -> str:
    n = len(events)
    if n >= 3:
        return "CRITICAL"
    if n >= 2:
        return "WARNING"
    return "WARN"


def _anomaly_severity(z: float, consecutive: int) -> str:
    if consecutive >= 3 or abs(z) > 4.0:
        return "CRITICAL"
    if consecutive >= 2 or abs(z) > 3.0:
        return "WARNING"
    if abs(z) > 2.5:
        return "WARN"
    return "NORMAL"


class NarrativeUnificationPhase:
    """Phase: Unify 3 real narrative sources with agreement/contradiction detection."""

    @property
    def name(self) -> str:
        return "narrative_unification"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            unifier = NarrativeUnifier()

            # ── Source 1: prediction (from ExplainPhase) ────────────
            prediction_source: Optional[Dict[str, Any]] = None
            if ctx.explanation:
                narrative_list = ctx.explanation.get("narratives", [])
                verdict = "; ".join(narrative_list[:3]) if narrative_list else None
                max_action = getattr(ctx, "max_action", "PREDICT")
                sev_map = {
                    "ESCALATE": "CRITICAL",
                    "INVESTIGATE": "WARNING",
                    "MONITOR": "WARN",
                    "PREDICT": "NORMAL",
                    "LOG_ONLY": "INFO",
                }
                prediction_source = {
                    "verdict": verdict or "predicción normal",
                    "severity": sev_map.get(max_action, "NORMAL"),
                    "confidence": ctx.fused_confidence or 0.5,
                }

            # ── Source 2: causal (from CausalPhase) ─────────────────
            causal_events = getattr(ctx, "causal_events", []) or []
            causal_source = _causal_narrative_from_events(causal_events)

            # ── Source 3: anomaly (from AnomalyDomainService) ───────
            anomaly_source = _anomaly_narrative_from_context(ctx)

            # ── Unified narrative ───────────────────────────────────
            unified = unifier.unify(
                prediction_explanation=prediction_source,
                anomaly_narrative=anomaly_source,
                text_narrative=causal_source,
            )

            # ── Agreement/contradiction logic ───────────────────────
            contradictions = list(unified.contradictions) if unified.contradictions else []
            confidence_boost = 0.0

            if prediction_source and anomaly_source:
                pred_sev = prediction_source.get("severity", "NORMAL")
                anom_sev = anomaly_source.get("severity", "NORMAL")
                if pred_sev == anom_sev:
                    # Agreement → boost
                    confidence_boost = 0.1
                elif abs(
                    _sev_rank(pred_sev) - _sev_rank(anom_sev)
                ) >= 2:
                    contradictions.append(
                        f"prediction_{pred_sev.lower()}_vs_anomaly_{anom_sev.lower()}"
                    )

            if prediction_source and causal_source:
                pred_sev = prediction_source.get("severity", "NORMAL")
                caus_sev = causal_source.get("severity", "NORMAL")
                if pred_sev == caus_sev:
                    confidence_boost = max(confidence_boost, 0.1)
                elif abs(
                    _sev_rank(pred_sev) - _sev_rank(caus_sev)
                ) >= 2:
                    contradictions.append(
                        f"prediction_{pred_sev.lower()}_vs_causal_{caus_sev.lower()}"
                    )

            final_conf = unified.confidence
            if contradictions:
                final_conf = max(0.1, final_conf - 0.1)
                logger.warning(
                    "narrative_contradictions_detected",
                    extra={
                        "series_id": ctx.series_id,
                        "contradictions": contradictions,
                        "boost": confidence_boost,
                        "final_confidence": round(final_conf, 3),
                    },
                )
            elif confidence_boost > 0:
                final_conf = min(0.95, final_conf + confidence_boost)

            # ── Build final narrative ───────────────────────────────
            sources_used = list(unified.sources_used) if unified.sources_used else []
            suppressed = list(unified.suppressed) if unified.suppressed else []

            from iot_machine_learning.domain.entities.results.unified_narrative import (
                UnifiedNarrative,
            )

            unified_narrative = UnifiedNarrative(
                primary_verdict=unified.primary_verdict or "unknown",
                severity=unified.severity,
                confidence=round(final_conf, 3),
                contradictions=contradictions,
                sources_used=sources_used,
                suppressed=suppressed,
            )

            return ctx.with_field(unified_narrative=unified_narrative)

        except Exception as e:
            logger.debug(f"narrative_unification_skipped: {e}")
            return ctx


def _sev_rank(sev: str) -> int:
    return {"UNKNOWN": 0, "INFO": 1, "NORMAL": 1, "WARN": 2, "WARNING": 2, "CRITICAL": 3, "ERROR": 3}.get(
        sev.upper(), 0
    )
