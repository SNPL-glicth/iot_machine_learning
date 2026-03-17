"""TextExplanationAssembler — builds Explanation domain object.

Constructs the same ``Explanation`` domain object produced by
``MetaCognitiveOrchestrator`` — full interoperability with
``ExplanationRenderer``, Weaviate storage, and any downstream consumer.

Human-readable conclusion rendering is NOT done here — that is the
caller's responsibility (ml_service layer calls ``build_semantic_conclusion``).

No imports from ml_service — only domain + sibling infrastructure.
Single entry point: ``TextExplanationAssembler.assemble()``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from iot_machine_learning.domain.entities.explainability.explanation import (
    Explanation,
    Outcome,
)
from iot_machine_learning.domain.entities.explainability.signal_snapshot import (
    SignalSnapshot,
)
from iot_machine_learning.domain.entities.explainability.reasoning_trace import (
    PhaseKind,
    ReasoningPhase,
    ReasoningTrace,
)
from iot_machine_learning.domain.entities.explainability.contribution_breakdown import (
    ContributionBreakdown,
    EngineContribution,
)
from iot_machine_learning.domain.services.severity_rules import SeverityResult

from ..analysis.types import EnginePerception, InhibitionState


class TextExplanationAssembler:
    """Builds ``Explanation`` domain object from text pipeline outputs.

    Stateless — safe to reuse across documents.
    """

    def assemble(
        self,
        *,
        document_id: str,
        signal: SignalSnapshot,
        perceptions: List[EnginePerception],
        inhibition_states: List[InhibitionState],
        final_weights: Dict[str, float],
        selected_engine: str,
        selection_reason: str,
        fusion_method: str,
        fused_confidence: float,
        domain: str,
        severity: SeverityResult,
        pipeline_phases: List[Dict[str, Any]],
    ) -> Explanation:
        """Assemble the Explanation domain object.

        Args:
            document_id: Document identifier (becomes series_id).
            signal: Text signal profile as SignalSnapshot.
            perceptions: EnginePerception list from sub-analyzers.
            inhibition_states: Inhibition results per engine.
            final_weights: Post-inhibition weights per engine.
            selected_engine: Engine with highest final weight.
            selection_reason: Reason for selection.
            fusion_method: Fusion method name.
            fused_confidence: Fused overall confidence.
            domain: Document domain classification.
            severity: Severity classification result.
            pipeline_phases: Phase timing/summary dicts.

        Returns:
            ``Explanation`` domain object.
        """
        # ── Contribution breakdown ──
        contributions = []
        for p in perceptions:
            inh = _find_inhibition(p.engine_name, inhibition_states)
            base_w = inh.base_weight if inh else final_weights.get(p.engine_name, 0.0)
            inhibited = inh.suppression_factor > 0.01 if inh else False
            inh_reason = inh.inhibition_reason if inh else "none"

            contributions.append(EngineContribution(
                engine_name=p.engine_name,
                predicted_value=p.predicted_value,
                confidence=p.confidence,
                trend=p.trend,
                base_weight=base_w,
                final_weight=final_weights.get(p.engine_name, 0.0),
                inhibited=inhibited,
                inhibition_reason=inh_reason,
                local_fit_error=p.local_fit_error,
                stability=p.stability,
                metadata=p.metadata,
            ))

        breakdown = ContributionBreakdown(
            contributions=contributions,
            fusion_method=fusion_method,
            selected_engine=selected_engine,
            selection_reason=selection_reason,
        )

        # ── Reasoning trace ──
        phases = []
        for phase_data in pipeline_phases:
            kind_str = phase_data.get("kind", "perceive")
            try:
                kind = PhaseKind(kind_str)
            except ValueError:
                continue
            phases.append(ReasoningPhase(
                kind=kind,
                summary=phase_data.get("summary", {}),
                duration_ms=phase_data.get("duration_ms"),
            ))

        trace = ReasoningTrace(
            phases=phases,
            regime_at_inference=domain,
            n_engines_available=len(perceptions),
            n_engines_active=sum(
                1 for w in final_weights.values() if w > 0.01
            ),
        )

        # ── Outcome ──
        outcome = Outcome(
            kind="text_analysis",
            predicted_value=None,
            confidence=fused_confidence,
            trend="stable",
            is_anomaly=severity.severity == "critical",
            anomaly_score=None,
            extra={
                "severity": severity.severity,
                "risk_level": severity.risk_level,
                "action_required": severity.action_required,
                "domain": domain,
            },
        )

        return Explanation(
            series_id=document_id,
            signal=signal,
            contributions=breakdown,
            trace=trace,
            outcome=outcome,
        )


def _find_inhibition(
    engine_name: str,
    states: List[InhibitionState],
) -> Optional[InhibitionState]:
    """Find InhibitionState for a given engine."""
    for s in states:
        if s.engine_name == engine_name:
            return s
    return None
