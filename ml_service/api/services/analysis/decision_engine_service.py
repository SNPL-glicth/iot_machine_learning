"""Decision engine service for document analysis.

Extracted from document_analyzer.py as part of refactoring Paso A.
Handles decision engine lifecycle and execution.

Includes coherence validation to prevent DecisionEngine from contradicting
the main analysis engine.

FASE 3: Added decision outcomes tracking, feedback loop, and confidence
calibration based on historical precision.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class DecisionEngineService:
    """Service for running decision engine on analysis results.
    
    Handles lazy initialization, strategy selection, and safe execution
    with graceful fallback on errors.
    
    Args:
        feature_flags: Optional feature flags for configuration
        decision_engine: Optional pre-configured decision engine
    """
    
    def __init__(
        self,
        feature_flags: Optional[Any] = None,
        decision_engine: Optional[Any] = None,
        enable_outcomes_tracking: bool = True,
    ) -> None:
        """Initialize service with optional dependencies."""
        self._feature_flags = feature_flags
        self._decision_engine = decision_engine
        self._strategy_map: Optional[Dict[str, Any]] = None
        self._enable_outcomes = enable_outcomes_tracking
        self._outcomes_repo = None
        
        # Lazy init outcomes repository
        if enable_outcomes_tracking:
            try:
                from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.decision_outcomes_repository import (
                    DecisionOutcomesRepository,
                )
                self._outcomes_repo = DecisionOutcomesRepository()
            except Exception as exc:
                logger.warning(
                    "decision_outcomes_repo_init_failed",
                    extra={"error": str(exc)},
                )
                self._enable_outcomes = False
    
    def run(
        self,
        universal_result: object,
        document_id: str,
        build_context_fn: callable,
    ) -> Optional[Dict[str, Any]]:
        """Run decision engine if enabled and available.
        
        Args:
            universal_result: UniversalResult from ML pipeline
            document_id: Document identifier
            build_context_fn: Function to build DecisionContext
            
        Returns:
            Decision dict or None if disabled/unavailable
        """
        # Check feature flag (default: disabled)
        if self._feature_flags is None:
            return None
        if not getattr(self._feature_flags, "ML_ENABLE_DECISION_ENGINE", False):
            return None
        
        # Lazy init decision engine if not provided
        if self._decision_engine is None:
            self._decision_engine = self._create_engine()
        
        if self._decision_engine is None:
            return None
        
        try:
            # Build DecisionContext from UniversalResult
            context = build_context_fn(universal_result, document_id)
            
            # Get decision (fail-safe via decide_safe)
            decision = self._decision_engine.decide_safe(
                context,
                fallback_reason="Decision engine processing failed",
            )
            
            # Validate decision against analysis result
            decision_dict = decision.to_dict()
            validated_decision = self.validate_against_analysis(
                decision_dict, universal_result
            )
            
            # Extract domain and regime for precision calibration
            domain = getattr(universal_result, 'domain', 'unknown')
            regime = self._extract_regime(universal_result)
            
            # Calibrate confidence based on historical precision
            validated_decision = self._calibrate_confidence(
                validated_decision, domain, regime
            )
            
            # Generate decision_id and save to DB
            decision_id = str(uuid4())
            validated_decision["decision_id"] = decision_id
            
            if self._enable_outcomes and self._outcomes_repo:
                self._outcomes_repo.save_decision(
                    decision_id=decision_id,
                    domain=domain,
                    regime=regime,
                    action_taken=validated_decision.get("action", decision.action),
                    severity_declared=validated_decision.get("severity", "unknown"),
                    confidence_declared=validated_decision.get("confidence", 0.5),
                )
            
            # Log decision outcome
            logger.info(
                "decision_engine_completed",
                extra={
                    "document_id": document_id,
                    "decision_id": decision_id,
                    "action": validated_decision.get("action", decision.action),
                    "priority": validated_decision.get("priority", decision.priority),
                    "strategy": decision.strategy_used,
                    "was_overridden": validated_decision.get("decision_override_reason") is not None,
                    "confidence": round(validated_decision.get("confidence", 0.5), 3),
                },
            )
            
            return validated_decision
            
        except Exception as exc:
            # Graceful fail: log but don't break pipeline
            logger.warning(
                "decision_engine_failed_gracefully",
                extra={"document_id": document_id, "error": str(exc)},
            )
            return None
    
    def validate_against_analysis(
        self,
        decision: Dict[str, Any],
        analysis_result: Any,
    ) -> Dict[str, Any]:
        """Validate decision against analysis result to prevent contradictions.
        
        Rules:
        1. DecisionEngine NEVER lowers a critical alert
        2. If decision suggests "critical" but severity="low" → use analysis severity
        3. If decision suggests "stable" but severity="critical" → keep critical
        
        Args:
            decision: Decision dict from decision engine
            analysis_result: UniversalResult from analysis
            
        Returns:
            Validated decision dict with optional override_reason
        """
        # Extract severity from analysis
        analysis_severity = "unknown"
        if hasattr(analysis_result, 'severity'):
            severity_obj = analysis_result.severity
            if hasattr(severity_obj, 'severity'):
                analysis_severity = severity_obj.severity
            elif isinstance(severity_obj, str):
                analysis_severity = severity_obj
        
        # Extract action from decision
        decision_action = decision.get("action", "monitor")
        
        # Rule 1: Never lower a critical/high alert
        if analysis_severity in ["critical", "high"]:
            if decision_action in ["ignore", "defer", "stable"]:
                logger.warning(
                    "decision_engine_override_prevented_downgrade",
                    extra={
                        "analysis_severity": analysis_severity,
                        "decision_action": decision_action,
                    }
                )
                decision["action"] = "escalate"  # Force escalation
                decision["decision_override_reason"] = (
                    f"DecisionEngine suggested '{decision_action}' but analysis is {analysis_severity}. "
                    f"Overridden to 'escalate' (conservative)."
                )
        
        # Rule 2: If decision is critical but analysis is low, use analysis
        if decision_action in ["escalate", "critical"] and analysis_severity in ["info", "low"]:
            logger.warning(
                "decision_engine_override_prevented_upgrade",
                extra={
                    "analysis_severity": analysis_severity,
                    "decision_action": decision_action,
                }
            )
            decision["action"] = "monitor"  # Use analysis severity
            decision["decision_override_reason"] = (
                f"DecisionEngine suggested '{decision_action}' but analysis is {analysis_severity}. "
                f"Overridden to 'monitor' (align with analysis)."
            )
        
        # Rule 3: If decision is stable but severity is critical, keep critical
        if decision_action == "stable" and analysis_severity in ["critical", "high", "warning"]:
            logger.warning(
                "decision_engine_override_stable_vs_critical",
                extra={
                    "analysis_severity": analysis_severity,
                    "decision_action": decision_action,
                }
            )
            decision["action"] = "investigate"  # Conservative middle ground
            decision["decision_override_reason"] = (
                f"DecisionEngine suggested 'stable' but analysis is {analysis_severity}. "
                f"Overridden to 'investigate' (conservative)."
            )
        
        # Rule 4: Low confidence + critical severity → downgrade severity
        decision_confidence = decision.get("confidence", 0.5)
        if decision_confidence < 0.4 and analysis_severity == "critical":
            logger.warning(
                "decision_engine_low_confidence_critical",
                extra={
                    "confidence": decision_confidence,
                    "severity": analysis_severity,
                }
            )
            decision["severity"] = "warning"
            decision["coherence_warning"] = (
                f"Low confidence ({decision_confidence:.2f}) with critical severity. "
                f"Downgraded to warning."
            )
        
        # Rule 5: Domain-specific urgency escalation
        if hasattr(analysis_result, 'domain'):
            domain = analysis_result.domain
            if hasattr(analysis_result, 'analysis'):
                analysis_dict = analysis_result.analysis
                if isinstance(analysis_dict, dict):
                    urgency = analysis_dict.get('urgency_score', 0.0)
                    
                    if domain == "operations" and decision_action == "monitor" and urgency > 0.8:
                        logger.info(
                            "decision_engine_urgency_escalation",
                            extra={
                                "domain": domain,
                                "urgency": urgency,
                                "original_action": decision_action,
                            }
                        )
                        decision["action"] = "investigate"
                        decision["escalation_reason"] = (
                            f"High urgency ({urgency:.2f}) in operations domain. "
                            f"Escalated from monitor to investigate."
                        )
        
        # Rule 6: Integrate coherence warnings from CoherenceValidator
        if hasattr(analysis_result, 'analysis'):
            analysis_dict = analysis_result.analysis
            if isinstance(analysis_dict, dict):
                coherence_warnings = analysis_dict.get('coherence_warnings', [])
                if coherence_warnings:
                    # Reduce confidence by 15% if coherence warnings exist
                    original_conf = decision.get("confidence", 0.5)
                    decision["confidence"] = max(0.1, original_conf * 0.85)
                    decision["coherence_penalty"] = f"Reduced confidence by 15% due to {len(coherence_warnings)} coherence warnings"
                    
                    logger.debug(
                        "decision_confidence_reduced_coherence",
                        extra={
                            "n_warnings": len(coherence_warnings),
                            "original_confidence": round(original_conf, 3),
                            "adjusted_confidence": round(decision["confidence"], 3),
                        }
                    )
        
        return decision
    
    def record_feedback(
        self,
        decision_id: str,
        was_correct: bool,
    ) -> None:
        """Record feedback for a decision.
        
        Args:
            decision_id: Decision identifier
            was_correct: Whether the decision was correct
        """
        if not self._enable_outcomes or not self._outcomes_repo:
            logger.debug(
                "decision_feedback_skipped",
                extra={"decision_id": decision_id, "reason": "outcomes_tracking_disabled"},
            )
            return
        
        self._outcomes_repo.record_feedback(decision_id, was_correct)
        
        # Log incorrect decisions for review
        if not was_correct:
            logger.warning(
                "decision_incorrect",
                extra={"decision_id": decision_id},
            )
    
    def _extract_regime(self, universal_result: Any) -> str:
        """Extract regime from universal result."""
        try:
            if hasattr(universal_result, 'analysis'):
                analysis = universal_result.analysis
                if isinstance(analysis, dict):
                    # Try to get regime from cognitive metadata
                    cognitive = analysis.get('cognitive', {})
                    if isinstance(cognitive, dict):
                        signal_profile = cognitive.get('signal_profile', {})
                        if isinstance(signal_profile, dict):
                            return signal_profile.get('regime', 'unknown')
            return 'unknown'
        except Exception:
            return 'unknown'
    
    def _calibrate_confidence(
        self,
        decision: Dict[str, Any],
        domain: str,
        regime: str,
    ) -> Dict[str, Any]:
        """Calibrate confidence based on historical precision.
        
        Args:
            decision: Decision dict
            domain: Domain of decision
            regime: Regime/context
            
        Returns:
            Decision with calibrated confidence
        """
        if not self._enable_outcomes or not self._outcomes_repo:
            return decision
        
        try:
            # Get recent precision for this domain/regime
            precision = self._outcomes_repo.get_recent_precision(domain, regime, limit=20)
            
            if precision is not None:
                original_confidence = decision.get("confidence", 0.5)
                
                # Calibrate confidence based on precision
                if precision < 0.5:
                    # Low precision: reduce confidence by 20%
                    calibrated = original_confidence * 0.8
                    decision["confidence"] = max(0.1, calibrated)
                    decision["confidence_calibration"] = "reduced_low_precision"
                elif precision > 0.8:
                    # High precision: allow slight increase (max 10%)
                    calibrated = min(0.95, original_confidence * 1.1)
                    decision["confidence"] = calibrated
                    decision["confidence_calibration"] = "increased_high_precision"
                else:
                    decision["confidence_calibration"] = "no_adjustment"
                
                decision["historical_precision"] = round(precision, 3)
                
                logger.debug(
                    "decision_confidence_calibrated",
                    extra={
                        "domain": domain,
                        "regime": regime,
                        "precision": round(precision, 3),
                        "original_confidence": round(original_confidence, 3),
                        "calibrated_confidence": round(decision["confidence"], 3),
                    },
                )
        except Exception as exc:
            logger.warning(
                "decision_confidence_calibration_failed",
                extra={"error": str(exc)},
            )
        
        return decision
    
    def _create_engine(self) -> Optional[Any]:
        """Create decision engine based on strategy from feature flags.
        
        Returns:
            Configured decision engine or None if not available
        """
        try:
            strategy_name = getattr(
                self._feature_flags, 
                "ML_DECISION_ENGINE_STRATEGY", 
                "simple"
            )
            
            # Import strategies
            from iot_machine_learning.infrastructure.ml.cognitive.decision import (
                SimpleDecisionEngine,
                ConservativeStrategy,
                AggressiveStrategy,
                CostOptimizedStrategy,
            )
            
            # Strategy map
            strategy_map = {
                "simple": SimpleDecisionEngine,
                "conservative": ConservativeStrategy,
                "aggressive": AggressiveStrategy,
                "cost_optimized": CostOptimizedStrategy,
            }
            
            engine_class = strategy_map.get(strategy_name, SimpleDecisionEngine)
            engine = engine_class()
            logger.info(f"decision_engine_lazy_initialized: strategy={strategy_name}")
            return engine
            
        except ImportError:
            logger.debug("decision_engine_unavailable: cognitive.decision not installed")
            return None
        except Exception as e:
            logger.warning(f"decision_engine_creation_failed: {e}")
            return None
