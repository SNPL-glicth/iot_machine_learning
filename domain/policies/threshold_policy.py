"""ThresholdPolicy — single source of truth for severity classification.

Unifies:
- AnomalySeverity.from_score()   (numeric score → severity enum)
- severity_rules.compute_severity() (anomaly + risk → severity string)
- adaptive_thresholds percentile logic
- severity_mapper 3-axis text classification

All severity decisions MUST go through this policy.
Design: frozen dataclass with configurable thresholds, pure logic, no I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from ..entities.results.anomaly import AnomalySeverity
from .policy_result import SeverityPolicyResult

logger = logging.getLogger(__name__)
def _load_defaults() -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    """Load defaults from centralized config if available, else hardcoded fallbacks.

    Keeps domain layer loosely coupled to config (import is lazy).
    """
    try:
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        cfg = FeatureFlags()
        score = (
            cfg.ML_SEVERITY_NONE_MAX,
            cfg.ML_SEVERITY_LOW_MAX,
            cfg.ML_SEVERITY_MEDIUM_MAX,
            cfg.ML_SEVERITY_HIGH_MAX,
        )
        text_w = (cfg.ML_TEXT_WEIGHT_URGENCY, cfg.ML_TEXT_WEIGHT_SENTIMENT, cfg.ML_TEXT_WEIGHT_IMPACT)
        text_t = (cfg.ML_TEXT_THRESHOLD_INFO, cfg.ML_TEXT_THRESHOLD_WARNING, cfg.ML_TEXT_THRESHOLD_CRITICAL)
        return score, text_w, text_t
    except Exception:
        # Fallbacks matching original hardcoded values
        return (0.3, 0.5, 0.7, 0.9), (0.45, 0.20, 0.35), (0.15, 0.35, 0.55)

# Default thresholds tuned for IoT sensor anomaly detection.
_DEFAULT_SCORE_THRESHOLDS: Tuple[float, float, float, float] = _load_defaults()[0]
_DEFAULT_TEXT_WEIGHTS: Tuple[float, float, float] = _load_defaults()[1]
_DEFAULT_TEXT_THRESHOLDS: Tuple[float, float, float] = _load_defaults()[2]

@dataclass(frozen=True)
class ThresholdPolicy:
    """Single policy for all severity classification.

    Args:
        score_thresholds: (none_max, low_max, medium_max, high_max)
            Scores below each bound map to the next severity tier.
        text_weights: (urgency, sentiment, impact) weights for 3-axis text severity.
        text_thresholds: (info_max, warning_min, critical_min) for composite text score.
        regime_overrides: Optional per-regime policy overrides.
    """

    score_thresholds: Tuple[float, float, float, float] = field(
        default_factory=lambda: _DEFAULT_SCORE_THRESHOLDS
    )
    text_weights: Tuple[float, float, float] = field(
        default_factory=lambda: _DEFAULT_TEXT_WEIGHTS
    )
    text_thresholds: Tuple[float, float, float] = field(
        default_factory=lambda: _DEFAULT_TEXT_THRESHOLDS
    )
    regime_overrides: Optional[Dict[str, "ThresholdPolicy"]] = None

    def classify_score(self, score: float) -> AnomalySeverity:
        """Map normalized score [0, 1] to AnomalySeverity.

        Replaces duplicated logic in:
        - AnomalySeverity.from_score()
        - AdaptiveThresholdManager.classify_severity()
        - voting_anomaly_detector severity assignment
        """
        none_max, low_max, medium_max, high_max = self.score_thresholds

        if score < none_max:
            return AnomalySeverity.NONE
        if score < low_max:
            return AnomalySeverity.LOW
        if score < medium_max:
            return AnomalySeverity.MEDIUM
        if score < high_max:
            return AnomalySeverity.HIGH
        return AnomalySeverity.CRITICAL

    def classify_score_label(self, score: float) -> str:
        """Return severity label string for UI/API consumption."""
        severity = self.classify_score(score)
        mapping = {
            AnomalySeverity.NONE: "info",
            AnomalySeverity.LOW: "info",
            AnomalySeverity.MEDIUM: "warning",
            AnomalySeverity.HIGH: "warning",
            AnomalySeverity.CRITICAL: "critical",
        }
        return mapping[severity]

    def classify_with_context(
        self,
        *,
        score: float,
        is_anomaly: bool = False,
        risk_level: str = "NONE",
        out_of_physical_range: bool = False,
        regime: Optional[str] = None,
        label: str = "",
    ) -> SeverityPolicyResult:
        """Full severity classification with all context."""
        from .context_policy import classify_with_context
        return classify_with_context(
            self,
            score=score,
            is_anomaly=is_anomaly,
            risk_level=risk_level,
            out_of_physical_range=out_of_physical_range,
            regime=regime,
            label=label,
        )

    def classify_text(
        self,
        *,
        urgency_score: float,
        sentiment_weight: float,
        impact_score: float,
        domain: str = "general",
        n_categories_hit: int = 0,
        urgency_override: bool = False,
    ) -> SeverityPolicyResult:
        """3-axis composite severity for text/document analysis."""
        from .text_policy import classify_text
        return classify_text(
            self,
            urgency_score=urgency_score,
            sentiment_weight=sentiment_weight,
            impact_score=impact_score,
            domain=domain,
            n_categories_hit=n_categories_hit,
            urgency_override=urgency_override,
        )

    def _regime_policy(self, regime: Optional[str]) -> "ThresholdPolicy":
        """Return regime-specific policy override if configured."""
        if regime and self.regime_overrides and regime in self.regime_overrides:
            return self.regime_overrides[regime]
        return self

    @staticmethod
    def _severity_to_risk_level(severity: AnomalySeverity) -> str:
        from .policy_helpers import _severity_to_risk_level
        return _severity_to_risk_level(severity)

    @classmethod
    def default(cls) -> "ThresholdPolicy":
        """Return the default policy (same thresholds as legacy code)."""
        from .policy_compat import default_policy
        return default_policy()

    @classmethod
    def from_score_thresholds(
        cls,
        none_max: float = 0.3,
        low_max: float = 0.5,
        medium_max: float = 0.7,
        high_max: float = 0.9,
    ) -> "ThresholdPolicy":
        """Factory matching legacy AnomalySeverity.from_score() signature."""
        from .policy_compat import from_score_thresholds
        return from_score_thresholds(none_max, low_max, medium_max, high_max)
