"""ContextualDecisionEngine — scoring determinista con factores contextuales.

Calcula score [0.0, 1.0] basado en severity + amplificadores/atenuadores.
Mapea score final a acción con prioridad.

Configuración vía ContextualDecisionConfig (inyectable):
  - Scores base por severidad
  - Amplificadores aditivos
  - Atenuadores sustractivos
  - Umbrales de decisión

Applies DIP: Engine depende de config abstraction, no de env vars.
Applies SRP: Configuración separada de lógica de decisión.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

from iot_machine_learning.domain.entities.decision import Decision, DecisionContext
from iot_machine_learning.domain.ports.decision_port import DecisionEnginePort
from .contextual_decision_config import ContextualDecisionConfig

logger = logging.getLogger(__name__)


@dataclass
class Amplifier:
    """Amplifier rule for contextual scoring.
    
    Attributes:
        name: Descriptive name for logging.
        condition: Callable that evaluates context and returns bool.
        value: Amplification value to add if condition is True.
        priority: Evaluation order (lower = evaluated first).
        exclusive: If True, stops evaluation chain when applied.
    
    Applies SRP: Each amplifier is a self-contained rule.
    """
    name: str
    condition: Callable[[DecisionContext], bool]
    value: float
    priority: int
    exclusive: bool = False


@dataclass
class Attenuator:
    """Attenuator rule for contextual scoring.
    
    Attributes:
        name: Descriptive name for logging.
        condition: Callable that evaluates context and returns bool.
        value: Attenuation value to subtract if condition is True.
        priority: Evaluation order (lower = evaluated first).
        exclusive: If True, stops evaluation chain when applied.
    
    Applies SRP: Each attenuator is a self-contained rule.
    """
    name: str
    condition: Callable[[DecisionContext], bool]
    value: float
    priority: int
    exclusive: bool = False


class ContextualDecisionEngine(DecisionEnginePort):
    """Engine de decisión con scoring contextual.
    
    Usa ContextualDecisionConfig como única fuente de verdad para:
    - Scores base por severidad (CRITICAL, HIGH, MEDIUM, LOW, NONE, WARNING)
    - Amplificadores aditivos (consecutive, rate, regime, drift)
    - Atenuadores sustractivos (stable, low_criticality, no_context)
    - Umbrales de decisión (escalate, investigate, monitor)
    
    Applies DIP: Depende de config abstraction inyectable.
    Applies OCP: Agregar amplificador solo requiere extender config.
    """
    
    def __init__(
        self,
        config: Optional[ContextualDecisionConfig] = None,
    ) -> None:
        """Initialize contextual engine with injectable configuration.
        
        Args:
            config: ContextualDecisionConfig with all parameters.
                   Defaults to standard config if not provided.
        
        Raises:
            ValueError: If config validation fails.
        
        Applies DIP: Configuration is injected, not read from env vars.
        Applies OCP: Adding amplifier/attenuator only requires extending tables.
        """
        self._config = config or ContextualDecisionConfig()
        self._config.validate()  # Fail fast on invalid config
        
        # Build amplifier and attenuator tables (OCP: extend here)
        self._amplifiers = self._build_amplifiers()
        self._attenuators = self._build_attenuators()
    
    @property
    def strategy_name(self) -> str:
        return self._config.engine_name
    
    @property
    def version(self) -> str:
        return self._config.engine_version
    
    def can_decide(self, context: DecisionContext) -> bool:
        """Siempre puede decidir con severity (tiene default)."""
        return context.severity is not None
    
    def _build_amplifiers(self) -> List[Amplifier]:
        """Build amplifier table from config.
        
        Returns:
            List of amplifiers sorted by priority.
        
        Applies OCP: To add new amplifier, add entry here.
        Priorities are consecutive (1-8) without gaps.
        
        NOTE: Priorities were renumbered from original (which had gaps).
        Original gap (1,2,3,5) was unintentional per audit.
        """
        cfg = self._config
        
        amplifiers = [
            # Priority 1-2: Consecutive anomalies (mutually exclusive via elif logic)
            Amplifier(
                name="consecutive_5",
                condition=lambda ctx: ctx.consecutive_anomalies >= cfg.amp_consecutive_high_count,
                value=cfg.amp_consecutive_5,
                priority=1,
                exclusive=True,  # If >=5, don't check >=3
            ),
            Amplifier(
                name="consecutive_3",
                condition=lambda ctx: ctx.consecutive_anomalies >= cfg.amp_consecutive_med_count,
                value=cfg.amp_consecutive_3,
                priority=2,
                exclusive=False,
            ),
            
            # Priority 3-4: Anomaly rate (mutually exclusive via elif logic)
            Amplifier(
                name="rate_high",
                condition=lambda ctx: ctx.recent_anomaly_rate > cfg.amp_rate_high_threshold,
                value=cfg.amp_rate_high,
                priority=3,
                exclusive=True,  # If >60%, don't check >30%
            ),
            Amplifier(
                name="rate_med",
                condition=lambda ctx: ctx.recent_anomaly_rate > cfg.amp_rate_med_threshold,
                value=cfg.amp_rate_med,
                priority=4,
                exclusive=False,
            ),
            
            # Priority 5-6: Regime (mutually exclusive - series has one regime)
            Amplifier(
                name="volatile",
                condition=lambda ctx: ctx.current_regime.upper() == "VOLATILE",
                value=cfg.amp_volatile,
                priority=5,
                exclusive=True,  # Only one regime applies
            ),
            Amplifier(
                name="noisy",
                condition=lambda ctx: ctx.current_regime.upper() == "NOISY",
                value=cfg.amp_noisy,
                priority=6,
                exclusive=False,
            ),
            
            # Priority 7-8: Drift score (mutually exclusive via elif logic)
            Amplifier(
                name="drift_high",
                condition=lambda ctx: ctx.drift_score > cfg.amp_drift_high_threshold,
                value=cfg.amp_drift_high,
                priority=7,
                exclusive=True,  # If >70%, don't check >40%
            ),
            Amplifier(
                name="drift_med",
                condition=lambda ctx: ctx.drift_score > cfg.amp_drift_med_threshold,
                value=cfg.amp_drift_med,
                priority=8,
                exclusive=False,
            ),
        ]
        
        # Sort by priority (defensive - already in order)
        return sorted(amplifiers, key=lambda a: a.priority)
    
    def _build_attenuators(self) -> List[Attenuator]:
        """Build attenuator table from config.
        
        Returns:
            List of attenuators sorted by priority.
        
        Applies OCP: To add new attenuator, add entry here.
        Priorities are consecutive (1-3) without gaps.
        
        NOTE: All attenuators are independent (can stack).
        """
        cfg = self._config
        
        attenuators = [
            # Priority 1: Stable regime with low drift
            Attenuator(
                name="stable",
                condition=lambda ctx: (
                    ctx.current_regime.upper() == "STABLE" and
                    ctx.drift_score < cfg.att_stable_drift_max
                ),
                value=cfg.att_stable,
                priority=1,
                exclusive=False,
            ),
            
            # Priority 2: Low criticality series
            Attenuator(
                name="low_criticality",
                condition=lambda ctx: ctx.series_criticality.upper() == "LOW",
                value=cfg.att_low_criticality,
                priority=2,
                exclusive=False,
            ),
            
            # Priority 3: No recent context (first anomaly)
            Attenuator(
                name="no_context",
                condition=lambda ctx: ctx.recent_anomaly_count == 0,
                value=cfg.att_no_context,
                priority=3,
                exclusive=False,
            ),
        ]
        
        # Sort by priority (defensive - already in order)
        return sorted(attenuators, key=lambda a: a.priority)
    
    def _compute_base_score(self, context: DecisionContext) -> float:
        """Score base según severity.
        
        Args:
            context: Decision context with severity.
        
        Returns:
            Base score from config based on severity level.
        
        Uses ContextualDecisionConfig as single source of truth.
        """
        if not context.severity:
            return self._config.score_none
        
        severity = context.severity.severity.lower()
        
        # Map severity to config score
        severity_map = {
            "critical": self._config.score_critical,
            "high": self._config.score_high,
            "medium": self._config.score_medium,
            "low": self._config.score_low,
            "info": self._config.score_none,
            "none": self._config.score_none,
            "warning": self._config.score_warning,
        }
        
        return severity_map.get(severity, self._config.score_none)
    
    def _apply_amplifiers(
        self,
        score: float,
        context: DecisionContext,
    ) -> tuple[float, List[str]]:
        """Aplicar amplificadores aditivos desde tabla.
        
        Args:
            score: Base score to amplify.
            context: Decision context with anomaly metrics.
        
        Returns:
            Tuple of (amplified_score, list_of_applied_amplifiers).
        
        Applies OCP: Evaluation logic is generic, rules are in table.
        Exclusive amplifiers stop evaluation of lower-priority rules in same category.
        
        NOTE: Amplifiers are ADDITIVE (not multiplicative) per PASO 1 spec.
        """
        applied: List[str] = []
        total_amplification = 0.0
        
        # Track which exclusive categories have been satisfied
        exclusive_triggered = set()

        for amp in self._amplifiers:
            # Skip if exclusive rule in same category already triggered
            # Categories: consecutive (1-2), rate (3-4), regime (5-6), drift (7-8)
            category = (amp.priority - 1) // 2  # 0,0,1,1,2,2,3,3
            if category in exclusive_triggered:
                continue
            
            # Evaluate condition
            if amp.condition(context):
                total_amplification += amp.value
                applied.append(f"{amp.name}+{amp.value}")
                
                # Mark category as satisfied if exclusive
                if amp.exclusive:
                    exclusive_triggered.add(category)

        # Apply total amplification with clamp to [0, 1]
        amplified_score = min(1.0, score + total_amplification)
        
        return amplified_score, applied
    
    def _apply_attenuators(
        self,
        score: float,
        context: DecisionContext,
    ) -> tuple[float, List[str]]:
        """Aplicar atenuadores sustractivos desde tabla.
        
        Args:
            score: Amplified score to attenuate.
            context: Decision context with stability metrics.
        
        Returns:
            Tuple of (attenuated_score, list_of_applied_attenuators).
        
        Applies OCP: Evaluation logic is generic, rules are in table.
        All attenuators are independent (can stack).
        
        NOTE: Attenuators are SUBTRACTIVE (not multiplicative) per PASO 1 spec.
        Includes floor clamp to prevent negative scores (BUG FIX from audit).
        """
        # PASO 4 FIX: Clamp score before attenuation to prevent negative
        score = max(0.0, score)
        
        applied: List[str] = []
        total_attenuation = 0.0

        for att in self._attenuators:
            # Evaluate condition
            if att.condition(context):
                total_attenuation += att.value
                applied.append(f"{att.name}-{att.value}")
                
                # Stop if exclusive (currently none are exclusive)
                if att.exclusive:
                    break
        
        # Apply total attenuation with floor clamp (PASO 4 FIX)
        attenuated_score = max(0.0, score - total_attenuation)
        
        return attenuated_score, applied
    
    def _score_to_action(self, score: float) -> tuple[str, int, str]:
        """Mapear score a (action, priority, reason) usando umbrales de config.
        
        Args:
            score: Final score after amplifiers and attenuators.
        
        Returns:
            Tuple of (action, priority, reason).
        
        Uses config thresholds as single source of truth.
        Priorities are consecutive (1,2,3,4) per PASO 1 fix.
        """
        if score >= self._config.threshold_escalate:
            return ("ESCALATE", self._config.priority_escalate, "patrón persistente detectado")
        elif score >= self._config.threshold_investigate:
            return ("INVESTIGATE", self._config.priority_investigate, "anomalía contextual confirmada")
        elif score >= self._config.threshold_monitor:
            return ("MONITOR", self._config.priority_monitor, "anomalía moderada")
        else:
            return ("LOG_ONLY", self._config.priority_log_only, "señal débil o aislada")
    
    def decide(self, context: DecisionContext) -> Decision:
        """Calcular decisión con scoring contextual.
        
        Args:
            context: Decision context with severity and metrics.
        
        Returns:
            Decision with action, priority, and confidence.
        
        Pipeline:
            1. Compute base score from severity
            2. Apply amplifiers (additive)
            3. Apply attenuators (subtractive)
            4. Clamp to [0, 1]
            5. Map to action via thresholds
        """
        # Score base
        score_base = self._compute_base_score(context)

        # Amplificadores (additive)
        score_after_amp, amplifiers = self._apply_amplifiers(score_base, context)

        # Atenuadores (subtractive with floor clamp)
        score_final, attenuators = self._apply_attenuators(score_after_amp, context)
        
        # Asegurar rango [0, 1] (redundant but defensive)
        score_final = max(0.0, min(1.0, score_final))
        
        # Mapear a acción
        action, priority, reason = self._score_to_action(score_final)
        
        # Logging estructurado con snapshot de config para auditabilidad
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
                "config_snapshot": {
                    "threshold_escalate": self._config.threshold_escalate,
                    "threshold_investigate": self._config.threshold_investigate,
                    "threshold_monitor": self._config.threshold_monitor,
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
