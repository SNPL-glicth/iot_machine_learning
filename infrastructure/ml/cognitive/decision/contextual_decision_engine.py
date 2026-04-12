"""ContextualDecisionEngine — scoring determinista con factores contextuales.

Calcula score [0.0, 1.0] basado en severity + amplificadores/atenuadores.
Mapea score final a acción con prioridad.

Configuración vía feature flags (leídos en cada llamada para hot-reload):
  ML_DECISION_AMP_*: Amplificadores multiplicativos
  ML_DECISION_ATT_*: Atenuadores multiplicativos
  ML_DECISION_THRESHOLD_*: Umbrales de mapeo
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .....domain.entities.decision import Decision, DecisionContext
from .....domain.ports.decision_port import DecisionEnginePort
from .....ml_service.config.flags import FeatureFlags

logger = logging.getLogger(__name__)


def _get_flags() -> FeatureFlags:
    """Obtener flags actuales (lazy import para evitar circular)."""
    try:
        from .....ml_service.config.feature_flags import get_feature_flags
        return get_feature_flags()
    except Exception:
        # Fallback a defaults si no hay flags configurados
        return FeatureFlags()


class ContextualDecisionEngine(DecisionEnginePort):
    """Engine de decisión con scoring contextual.
    
    Score base según severity:
      CRITICAL → 0.90
      HIGH     → 0.70
      MEDIUM   → 0.45
      LOW      → 0.25
      NONE     → 0.05
    
    Amplificadores (acumulativos, techo en 1.0) - valores desde flags:
      consecutive_anomalies >= 5 → ×flags.ML_DECISION_AMP_CONSECUTIVE_5
      consecutive_anomalies >= 3 → ×flags.ML_DECISION_AMP_CONSECUTIVE_3
      recent_anomaly_rate > 0.60  → ×flags.ML_DECISION_AMP_RATE_HIGH
      recent_anomaly_rate > 0.30  → ×flags.ML_DECISION_AMP_RATE_MED
      current_regime == "VOLATILE" → ×flags.ML_DECISION_AMP_VOLATILE
      current_regime == "NOISY"    → ×flags.ML_DECISION_AMP_NOISY
      drift_score > 0.70            → ×flags.ML_DECISION_AMP_DRIFT_HIGH
      drift_score > 0.40            → ×flags.ML_DECISION_AMP_DRIFT_MED
    
    Atenuadores (aplicados después) - valores desde flags:
      current_regime == "STABLE" and drift < 0.10 → ×flags.ML_DECISION_ATT_STABLE
      series_criticality == "LOW"  → ×flags.ML_DECISION_ATT_LOW_CRITICALITY
      recent_anomaly_count == 0  → ×flags.ML_DECISION_ATT_NO_CONTEXT
    
    Mapeo score → acción - umbrales desde flags:
      >= flags.ML_DECISION_THRESHOLD_ESCALATE   → ESCALATE,   priority=1
      >= flags.ML_DECISION_THRESHOLD_INVESTIGATE → INVESTIGATE, priority=2
      >= flags.ML_DECISION_THRESHOLD_MONITOR     → MONITOR,     priority=3
      < flags.ML_DECISION_THRESHOLD_MONITOR      → LOG_ONLY,    priority=5
    """
    
    def __init__(self, version: str = "1.0.0") -> None:
        """Initialize contextual engine.
        
        Args:
            version: Version string for audit
        """
        self._version = version
    
    @property
    def strategy_name(self) -> str:
        return "contextual"
    
    @property
    def version(self) -> str:
        return self._version
    
    def can_decide(self, context: DecisionContext) -> bool:
        """Siempre puede decidir con severity (tiene default)."""
        return context.severity is not None
    
    def _get_base_scores(self, flags: FeatureFlags) -> Dict[str, float]:
        """Leer base_scores desde flags JSON (hot-reload)."""
        try:
            return json.loads(flags.ML_DECISION_BASE_SCORES)
        except json.JSONDecodeError:
            # Fallback seguro
            return {
                "critical": 0.90,
                "high": 0.70,
                "medium": 0.45,
                "low": 0.25,
                "info": 0.05,
                "warning": 0.45,
            }

    def _get_amp_thresholds(self, flags: FeatureFlags) -> Dict[str, float]:
        """Leer amplifier thresholds desde flags JSON (hot-reload)."""
        try:
            return json.loads(flags.ML_DECISION_AMP_THRESHOLDS)
        except json.JSONDecodeError:
            # Fallback seguro
            return {
                "count_high": 5,
                "count_medium": 3,
                "ratio_high": 0.60,
                "ratio_low": 0.30,
                "drift_high": 0.70,
                "drift_low": 0.40,
            }

    def _get_stable_drift_threshold(self, flags: FeatureFlags) -> float:
        """Leer umbral de drift para atenuador STABLE desde flags."""
        try:
            return float(flags.ML_DECISION_ATT_STABLE_DRIFT_THRESHOLD)
        except Exception:
            return 0.10  # fallback

    def _compute_base_score(self, context: DecisionContext, flags: FeatureFlags) -> float:
        """Score base según severity."""
        severity = context.severity.severity if context.severity else "info"
        base_scores = self._get_base_scores(flags)
        return base_scores.get(severity.lower(), 0.05)
    
    def _apply_amplifiers(
        self,
        score: float,
        context: DecisionContext,
        flags: FeatureFlags,
    ) -> tuple[float, List[str]]:
        """Aplicar amplificadores desde flags, retornar score y lista aplicada."""
        amplifiers: List[str] = []
        thresholds = self._get_amp_thresholds(flags)

        # Consecutive anomalies
        if context.consecutive_anomalies >= thresholds.get("count_high", 5):
            multiplier = flags.ML_DECISION_AMP_CONSECUTIVE_5
            score = min(1.0, score * multiplier)
            amplifiers.append(f"consecutive_5×{multiplier}")
        elif context.consecutive_anomalies >= thresholds.get("count_medium", 3):
            multiplier = flags.ML_DECISION_AMP_CONSECUTIVE_3
            score = min(1.0, score * multiplier)
            amplifiers.append(f"consecutive_3×{multiplier}")

        # Anomaly rate
        if context.recent_anomaly_rate > thresholds.get("ratio_high", 0.60):
            multiplier = flags.ML_DECISION_AMP_RATE_HIGH
            score = min(1.0, score * multiplier)
            amplifiers.append(f"rate_high×{multiplier}")
        elif context.recent_anomaly_rate > thresholds.get("ratio_low", 0.30):
            multiplier = flags.ML_DECISION_AMP_RATE_MED
            score = min(1.0, score * multiplier)
            amplifiers.append(f"rate_med×{multiplier}")

        # Regime
        regime = context.current_regime.upper()
        if regime == "VOLATILE":
            multiplier = flags.ML_DECISION_AMP_VOLATILE
            score = min(1.0, score * multiplier)
            amplifiers.append(f"volatile×{multiplier}")
        elif regime == "NOISY":
            multiplier = flags.ML_DECISION_AMP_NOISY
            score = min(1.0, score * multiplier)
            amplifiers.append(f"noisy×{multiplier}")

        # Drift score
        if context.drift_score > thresholds.get("drift_high", 0.70):
            multiplier = flags.ML_DECISION_AMP_DRIFT_HIGH
            score = min(1.0, score * multiplier)
            amplifiers.append(f"drift_high×{multiplier}")
        elif context.drift_score > thresholds.get("drift_low", 0.40):
            multiplier = flags.ML_DECISION_AMP_DRIFT_MED
            score = min(1.0, score * multiplier)
            amplifiers.append(f"drift_med×{multiplier}")

        return score, amplifiers
    
    def _apply_attenuators(
        self,
        score: float,
        context: DecisionContext,
        flags: FeatureFlags,
    ) -> tuple[float, List[str]]:
        """Aplicar atenuadores desde flags, retornar score y lista aplicada."""
        attenuators: List[str] = []

        # Stable regime with low drift
        drift_threshold = self._get_stable_drift_threshold(flags)
        if context.current_regime.upper() == "STABLE" and context.drift_score < drift_threshold:
            multiplier = flags.ML_DECISION_ATT_STABLE
            score *= multiplier
            attenuators.append(f"stable×{multiplier}")
        
        # Low criticality
        if context.series_criticality.upper() == "LOW":
            multiplier = flags.ML_DECISION_ATT_LOW_CRITICALITY
            score *= multiplier
            attenuators.append(f"low_criticality×{multiplier}")
        
        # No recent context (primera anomalía)
        if context.recent_anomaly_count == 0:
            multiplier = flags.ML_DECISION_ATT_NO_CONTEXT
            score *= multiplier
            attenuators.append(f"no_context×{multiplier}")
        
        return score, attenuators
    
    def _score_to_action(self, score: float) -> tuple[str, int, str]:
        """Mapear score a (action, priority, reason) usando umbrales de flags."""
        flags = _get_flags()
        
        if score >= flags.ML_DECISION_THRESHOLD_ESCALATE:
            return ("ESCALATE", 1, "patrón persistente detectado")
        elif score >= flags.ML_DECISION_THRESHOLD_INVESTIGATE:
            return ("INVESTIGATE", 2, "anomalía contextual confirmada")
        elif score >= flags.ML_DECISION_THRESHOLD_MONITOR:
            return ("MONITOR", 3, "anomalía moderada")
        else:
            return ("LOG_ONLY", 5, "señal débil o aislada")
    
    def decide(self, context: DecisionContext) -> Decision:
        """Calcular decisión con scoring contextual."""
        # Snapshot de flags para logging auditable
        flags = _get_flags()
        
        # Score base
        score_base = self._compute_base_score(context, flags)

        # Amplificadores
        score_after_amp, amplifiers = self._apply_amplifiers(score_base, context, flags)

        # Atenuadores
        score_final, attenuators = self._apply_attenuators(score_after_amp, context, flags)
        
        # Asegurar rango [0, 1]
        score_final = max(0.0, min(1.0, score_final))
        
        # Mapear a acción
        action, priority, reason = self._score_to_action(score_final)
        
        # Logging estructurado con snapshot de flags para auditabilidad
        logger.info(
            "contextual_decision",
            extra={
                "engine": "contextual",
                "series_id": context.series_id,
                "score_base": round(score_base, 4),
                "amplifiers": amplifiers,
                "attenuators": attenuators,
                "score_final": round(score_final, 4),
                "action": action,
                "priority": priority,
                "reason": reason,
                "flags_snapshot": {
                    "amp_consecutive_5": flags.ML_DECISION_AMP_CONSECUTIVE_5,
                    "amp_consecutive_3": flags.ML_DECISION_AMP_CONSECUTIVE_3,
                    "amp_rate_high": flags.ML_DECISION_AMP_RATE_HIGH,
                    "amp_rate_med": flags.ML_DECISION_AMP_RATE_MED,
                    "amp_volatile": flags.ML_DECISION_AMP_VOLATILE,
                    "amp_noisy": flags.ML_DECISION_AMP_NOISY,
                    "amp_drift_high": flags.ML_DECISION_AMP_DRIFT_HIGH,
                    "amp_drift_med": flags.ML_DECISION_AMP_DRIFT_MED,
                    "att_stable": flags.ML_DECISION_ATT_STABLE,
                    "att_low_criticality": flags.ML_DECISION_ATT_LOW_CRITICALITY,
                    "att_no_context": flags.ML_DECISION_ATT_NO_CONTEXT,
                    "threshold_escalate": flags.ML_DECISION_THRESHOLD_ESCALATE,
                    "threshold_investigate": flags.ML_DECISION_THRESHOLD_INVESTIGATE,
                    "threshold_monitor": flags.ML_DECISION_THRESHOLD_MONITOR,
                },
            },
        )
        
        # Construir decisión
        return Decision(
            action=action,
            priority=priority,
            confidence=round(score_final, 4),
            reason=reason,
            strategy_used=self.strategy_name,
            simulated_outcomes=None,
            source_ml_outputs={
                "score_base": score_base,
                "amplifiers": amplifiers,
                "attenuators": attenuators,
                "score_final": score_final,
                "regime": context.current_regime,
                "consecutive": context.consecutive_anomalies,
                "anomaly_rate": context.recent_anomaly_rate,
            },
            audit_trace_id=context.audit_trace_id,
        )
