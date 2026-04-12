"""ConservativeStrategy — Prefers false positives over false negatives.

Conservative approach: When in doubt, escalate.
Always errs on the side of caution, prioritizing safety over efficiency.
Uses worst-case Monte Carlo scenarios to justify decisions.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ......domain.entities.decision import Decision, DecisionContext, SimulatedOutcome
from ......domain.ports.decision_port import DecisionEnginePort
from ......domain.services.cognitive_constants import (
    get_confidence_ceiling,
    get_confidence_floor,
    get_confidence_reduction_sparse,
)

from .decision_rules import apply_decision_hierarchy
from .outcome_builder import build_simulated_outcomes


class ConservativeStrategy(DecisionEnginePort):
    """Conservative decision strategy.

    Decision hierarchy (first match wins):
    1. severity=critical OR pattern contains "escalation" → escalate
    2. confidence > 0.8 → intervene
    3. default → investigate

    This strategy:
    - Prefers false positives over false negatives (better safe than sorry)
    - Uses Monte Carlo worst-case scenario to justify decisions
    - Always considers high-confidence anomalies as actionable
    - Adds safety margin to all risk assessments

    Attributes:
        _version: Strategy version for audit trails
        _confidence_threshold: Minimum confidence for "intervene" (default 0.8)
        _safety_margin: Risk multiplier for worst-case analysis (default 1.2)
    """

    def __init__(
        self,
        version: str = "1.0.0",
        confidence_threshold: float | None = None,
        safety_margin: float | None = None,
    ) -> None:
        """Initialize ConservativeStrategy.

        Args:
            version: Version string for audit trails
            confidence_threshold: Confidence threshold for "intervene" action (None = read from flags)
            safety_margin: Multiplier applied to risk in worst-case analysis (None = read from flags)
        """
        self._version = version
        self._confidence_threshold = confidence_threshold
        self._safety_margin = safety_margin

    def _get_confidence_threshold(self) -> float:
        """Get confidence threshold from flags or fallback (hot-reload)."""
        if self._confidence_threshold is not None:
            return self._confidence_threshold
        try:
            from ......ml_service.config.feature_flags import get_feature_flags
            return get_feature_flags().ML_DECISION_CONSERVATIVE_THRESHOLD
        except Exception:
            return 0.8  # fallback

    def _get_safety_margin(self) -> float:
        """Get safety margin from flags or fallback (hot-reload)."""
        if self._safety_margin is not None:
            return self._safety_margin
        try:
            from ......ml_service.config.feature_flags import get_feature_flags
            return get_feature_flags().ML_DECISION_CONSERVATIVE_SAFETY_MARGIN
        except Exception:
            return 1.2  # fallback

    @property
    def strategy_name(self) -> str:
        """Unique identifier: 'conservative'."""
        return "conservative"

    @property
    def version(self) -> str:
        """Semantic version of this strategy."""
        return self._version

    def can_decide(self, context: DecisionContext) -> bool:
        """Check if context has minimum required fields.

        Conservative strategy requires at least severity or patterns
        to make a risk assessment.

        Args:
            context: Decision context to validate

        Returns:
            True if severity or patterns are present
        """
        return context.severity is not None or len(context.patterns) > 0

    def decide(self, context: DecisionContext) -> Decision:
        """Make conservative decision.

        Applies decision hierarchy:
        1. Check for critical conditions requiring escalation
        2. Check for high-confidence anomalies requiring intervention
        3. Default to investigation for all other cases

        Args:
            context: Aggregated ML outputs

        Returns:
            Decision with conservative action selection
        """
        # Build simulated outcomes from Monte Carlo or generate conservative scenarios
        simulated = build_simulated_outcomes(context, self._get_safety_margin())

        # Decision hierarchy: escalate → intervene → investigate
        action, priority, reason = apply_decision_hierarchy(
            context, simulated, self._get_confidence_threshold()
        )

        # Calculate confidence based on evidence strength
        confidence = self._calculate_confidence(context)

        # Build source ML outputs reference
        source_outputs = self._build_source_outputs(context)

        return Decision(
            action=action,
            priority=priority,
            confidence=confidence,
            reason=reason,
            strategy_used=self.strategy_name,
            simulated_outcomes=simulated,
            source_ml_outputs=source_outputs,
            audit_trace_id=context.audit_trace_id,
        )

    def _calculate_confidence(self, context: DecisionContext) -> float:
        """Calculate decision confidence.

        Conservative confidence calculation:
        - Never exceeds input confidence
        - Reduced if evidence is sparse
        - Minimum 0.6 (we're confident in being conservative)

        Args:
            context: Decision context

        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = context.confidence

        # Evidence quality modifier (leer de flags en cada llamada para hot-reload)
        evidence_count = len(context.patterns) + (1 if context.severity else 0)
        if evidence_count < 2:
            base_confidence *= get_confidence_reduction_sparse()  # Reduce confidence with sparse evidence

        # Conservative floor: we're always at least 60% confident
        # in conservative decisions (by design) - valores desde flags
        return max(get_confidence_floor(), min(get_confidence_ceiling(), base_confidence))

    def _build_source_outputs(self, context: DecisionContext) -> Dict[str, Any]:
        """Build reference to source ML outputs.

        Args:
            context: Decision context

        Returns:
            Dictionary of source references
        """
        return {
            "series_id": context.series_id,
            "confidence": context.confidence,
            "is_anomaly": context.is_anomaly,
            "anomaly_score": context.anomaly_score,
            "trend": context.trend,
            "domain": context.domain,
            "pattern_count": len(context.patterns),
            "has_monte_carlo": context.has_monte_carlo,
            "strategy_config": {
                "confidence_threshold": self._get_confidence_threshold(),
                "safety_margin": self._get_safety_margin(),
            },
        }
