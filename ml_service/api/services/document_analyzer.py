"""Document Analysis Service — Thin orchestrator.

Delegates to modular analysis pipeline components.
Includes optional Decision Engine for recommendations.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from .analysis import (
    analyze_with_universal,
    analyze_with_legacy,
    build_output_dict,
    extract_raw_data,
    analyze_with_neural,
    arbitrate_results,
    extract_analysis_scores,
)

# Import domain classifier for online learning
from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.domain_classifier import fit_online

# Graceful import - feature flags
try:
    from iot_machine_learning.ml_service.config.feature_flags import get_feature_flags
    _FEATURE_FLAGS_AVAILABLE = True
except Exception:
    _FEATURE_FLAGS_AVAILABLE = False

# Graceful import - decision engine optional
_DECISION_AVAILABLE = False
try:
    from iot_machine_learning.infrastructure.ml.cognitive.decision import SimpleDecisionEngine
    from iot_machine_learning.domain.entities.decision import DecisionContext
    _DECISION_AVAILABLE = True
    logger.info("decision_engine_available")
except Exception as e:
    logger.warning(f"decision_engine_unavailable: {e}")
    _DECISION_AVAILABLE = False

# Graceful import - fall back to legacy analyzers if universal engines unavailable
_UNIVERSAL_AVAILABLE = False
try:
    from iot_machine_learning.infrastructure.ml.cognitive.universal import (
        UniversalAnalysisEngine,
        UniversalComparativeEngine,
    )
    _UNIVERSAL_AVAILABLE = True
    logger.info("universal_engines_available")
except ImportError as e:
    logger.error(f"UNIVERSAL_ANALYSIS_UNAVAILABLE: Missing dependencies. Install with: pip install numpy scipy scikit-learn")
    _UNIVERSAL_AVAILABLE = False
except Exception as e:
    logger.warning(f"universal_engines_unavailable_using_legacy_fallback: {e}")
    _UNIVERSAL_AVAILABLE = False

# Graceful import - neural engine optional
_NEURAL_AVAILABLE = False
try:
    from iot_machine_learning.infrastructure.ml.cognitive.neural import (
        HybridNeuralEngine,
    )
    from iot_machine_learning.infrastructure.ml.cognitive.neural.competition import (
        NeuralArbiter,
    )
    _NEURAL_AVAILABLE = True
    logger.info("neural_engine_available")
except Exception as e:
    logger.warning(f"neural_engine_unavailable: {e}")


class DocumentAnalyzer:
    """Universal document analyzer backed by real ML engines.
    
    Automatically detects input type and routes to appropriate analysis.
    Produces Explanation domain objects with comparative context.
    """

    def __init__(
        self,
        cognitive_memory: Optional[object] = None,
        decision_engine: Optional[object] = None,
        feature_flags: Optional[object] = None,
    ):
        """Initialize with optional cognitive memory and decision engine.
        
        Args:
            cognitive_memory: Optional CognitiveMemoryPort for semantic recall
            decision_engine: Optional DecisionEnginePort (auto-created if None and enabled)
            feature_flags: Optional FeatureFlags for decision engine enablement
        """
        self._cognitive_memory = cognitive_memory
        self._analysis_engine = UniversalAnalysisEngine() if _UNIVERSAL_AVAILABLE else None
        self._comparative_engine = UniversalComparativeEngine() if _UNIVERSAL_AVAILABLE else None
        self._neural_engine = HybridNeuralEngine() if _NEURAL_AVAILABLE else None
        self._neural_arbiter = None  # Default seguro
        try:
            self._neural_arbiter = NeuralArbiter() if _NEURAL_AVAILABLE else None
        except Exception:
            pass  # Graceful fallback si NeuralArbiter falla al inicializar
        
        # Auto-load feature flags if not provided
        if feature_flags is None and _FEATURE_FLAGS_AVAILABLE:
            try:
                feature_flags = get_feature_flags()
            except Exception:
                feature_flags = None
        
        # Decision Engine (lazy init based on feature flags)
        self._decision_engine = decision_engine
        self._feature_flags = feature_flags

    def analyze(
        self,
        document_id: str,
        content_type: str,
        normalized_payload: Dict[str, Any],
        tenant_id: str = "",
    ) -> Dict[str, Any]:
        """Analyze document and return structured result.
        
        Args:
            document_id: Unique identifier for tracking
            content_type: Hint for content type (text, tabular, mixed, etc.)
            normalized_payload: Pre-processed document payload
            tenant_id: Multi-tenant isolation
            
        Returns:
            Dict with analysis, conclusion, confidence, comparative context
        """
        start = time.time()
        
        # DEBUG: Log input parameters
        logger.info(f"[STAGE-4] analyze called, content_type={content_type}")
        
        try:
            if _UNIVERSAL_AVAILABLE and self._analysis_engine:
                logger.info(f"[DEBUG] Using universal analysis path")
                # Extract raw data for logging
                raw_data = extract_raw_data(normalized_payload, content_type)
                logger.info(f"[STAGE-5] raw_data type={type(raw_data)}, length={len(str(raw_data))}")
                
                # Step 1: Run universal analysis
                universal_result, comparison_result, semantic_conclusion = analyze_with_universal(
                    document_id=document_id,
                    content_type=content_type,
                    payload=normalized_payload,
                    tenant_id=tenant_id,
                    analysis_engine=self._analysis_engine,
                    comparative_engine=self._comparative_engine,
                    cognitive_memory=self._cognitive_memory,
                )
                
                # Step 2: Run neural analysis (if GPU available)
                neural_result = None
                if _NEURAL_AVAILABLE and self._neural_engine:
                    # Only run neural on GPU - too slow on CPU
                    import os
                    has_gpu = os.environ.get('ZENIN_GPU_AVAILABLE', 'false').lower() == 'true'
                    if has_gpu:
                        # Extract scores from universal result
                        analysis_scores = extract_analysis_scores(universal_result)
                        domain = getattr(universal_result, 'domain', 'unknown')
                        
                        neural_result = analyze_with_neural(
                            analysis_scores=analysis_scores,
                            input_type=content_type,
                            domain=domain,
                            neural_engine=self._neural_engine,
                        )
                    else:
                        logger.info("neural_skipped_no_gpu: CPU-only mode, using universal engine")
                
                # Step 3: Neural arbitration (if both results available)
                if neural_result is not None and getattr(self, '_neural_arbiter', None):
                    winner_result, winner_engine, arbiter_reason = arbitrate_results(
                        neural_result=neural_result,
                        universal_result=universal_result,
                        domain=getattr(universal_result, 'domain', 'unknown'),
                        arbiter=self._neural_arbiter,
                    )
                else:
                    # Neural not available, use universal
                    winner_result = universal_result
                    winner_engine = "universal"
                    arbiter_reason = "neural_unavailable_or_disabled"
                
                # Step 4: Build output from winner
                raw_data = extract_raw_data(normalized_payload, content_type)
                result = build_output_dict(winner_result, comparison_result, raw_data, semantic_conclusion)
                
                # Add arbitration metadata
                result["engine_used"] = winner_engine
                result["arbitration_reason"] = arbiter_reason
                
                # Include neural metrics if available
                if neural_result is not None:
                    result["neural_metrics"] = {
                        "energy_consumed": neural_result.energy_consumed,
                        "active_neurons": neural_result.active_neurons,
                        "silent_neurons": neural_result.silent_neurons,
                        "energy_efficiency": neural_result.energy_efficiency,
                    }
                
                # Step 5: Decision Engine (optional, feature-flagged)
                decision_recommendation = self._run_decision_engine(
                    universal_result, document_id
                )
                if decision_recommendation is not None:
                    result["decision_recommendation"] = decision_recommendation
                
                # Step 5.5: Plasticity feedback loop - record actual outcome for learning
                try:
                    from iot_machine_learning.infrastructure.ml.cognitive.plasticity.base import PlasticityTracker
                    plasticity = PlasticityTracker()
                    plasticity.record_actual(
                        actual_value=1.0 if getattr(winner_result, 'severity', None) and getattr(winner_result.severity, 'severity', None) == 'critical' else 0.0,
                        perceptions=getattr(winner_result, 'explanation', None).engine_contributions if hasattr(getattr(winner_result, 'explanation', None), 'engine_contributions') else [],
                        regime=getattr(winner_result, 'domain', 'general'),
                    )
                    logger.debug(f"plasticity_feedback_recorded: {document_id}")
                except Exception as e:
                    logger.warning(f"plasticity_feedback_failed: {e}")
                
                # Step 6: Online learning - update NaiveBayes with confirmed domain
                try:
                    fit_online(
                        pre_computed_scores=pre_computed_scores if 'pre_computed_scores' in dir() else {},
                        label=winner_result.domain if hasattr(winner_result, 'domain') else 'general',
                    )
                except Exception:
                    # Graceful fail - online learning errors shouldn't break pipeline
                    pass
            else:
                # Delegate to legacy pipeline
                result = analyze_with_legacy(
                    document_id, content_type, normalized_payload
                )

            return {
                "document_id": document_id,
                "content_type": content_type,
                **result,
                "processing_time_ms": round((time.time() - start) * 1000, 2),
            }
        except Exception as exc:
            logger.exception(
                "document_analysis_failed",
                extra={"document_id": document_id, "error": str(exc)},
            )
            raise

    def _run_decision_engine(
        self,
        universal_result: object,
        document_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Run Decision Engine if enabled and available.
        
        Args:
            universal_result: UniversalResult from ML pipeline
            document_id: Document identifier
            
        Returns:
            Decision dict or None if disabled/unavailable
        """
        # Check feature flag (default: disabled)
        if self._feature_flags is None:
            return None
        if not getattr(self._feature_flags, "ML_ENABLE_DECISION_ENGINE", False):
            return None
        
        # Lazy init decision engine if not provided
        if self._decision_engine is None and _DECISION_AVAILABLE:
            strategy_name = getattr(self._feature_flags, "ML_DECISION_ENGINE_STRATEGY", "simple")
            
            # Import all strategies
            from iot_machine_learning.infrastructure.ml.cognitive.decision import (
                SimpleDecisionEngine,
                ConservativeStrategy,
                AggressiveStrategy,
                CostOptimizedStrategy,
            )
            
            # Create engine based on strategy
            strategy_map = {
                "simple": SimpleDecisionEngine,
                "conservative": ConservativeStrategy,
                "aggressive": AggressiveStrategy,
                "cost_optimized": CostOptimizedStrategy,
            }
            
            engine_class = strategy_map.get(strategy_name, SimpleDecisionEngine)
            self._decision_engine = engine_class()
            logger.info(f"decision_engine_lazy_initialized: strategy={strategy_name}")
        
        if self._decision_engine is None:
            return None
        
        try:
            # Build DecisionContext from UniversalResult
            context = self._build_decision_context(universal_result, document_id)
            
            # Get decision (fail-safe via decide_safe)
            decision = self._decision_engine.decide_safe(
                context,
                fallback_reason="Decision engine processing failed",
            )
            
            # Log decision outcome
            logger.info(
                "decision_engine_completed",
                extra={
                    "document_id": document_id,
                    "action": decision.action,
                    "priority": decision.priority,
                    "strategy": decision.strategy_used,
                },
            )
            
            return decision.to_dict()
            
        except Exception as exc:
            # Graceful fail: log but don't break pipeline
            logger.warning(
                "decision_engine_failed_gracefully",
                extra={"document_id": document_id, "error": str(exc)},
            )
            return None
    
    def _build_decision_context(
        self,
        universal_result: object,
        document_id: str,
    ) -> "DecisionContext":
        """Build DecisionContext from UniversalResult.
        
        Args:
            universal_result: UniversalResult from ML pipeline
            document_id: Document identifier
            
        Returns:
            DecisionContext for decision engine
        """
        # Extract fields safely with defaults
        severity = getattr(universal_result, "severity", None)
        confidence = getattr(universal_result, "confidence", 0.0)
        domain = getattr(universal_result, "domain", "")
        patterns = getattr(universal_result, "patterns", [])
        
        # Convert patterns to dict format
        pattern_dicts = []
        for p in patterns:
            if hasattr(p, "to_dict"):
                pattern_dicts.append(p.to_dict())
            elif isinstance(p, dict):
                pattern_dicts.append(p)
            else:
                pattern_dicts.append({
                    "pattern_type": getattr(p, "pattern_type", "unknown"),
                    "severity_hint": getattr(p, "severity_hint", "info"),
                    "confidence": getattr(p, "confidence", 0.0),
                })
        
        # Extract outcome from explanation if available
        explanation = getattr(universal_result, "explanation", None)
        is_anomaly = False
        anomaly_score = 0.0
        predicted_value = None
        trend = "stable"
        audit_trace_id = None
        
        if explanation is not None:
            outcome = getattr(explanation, "outcome", None)
            if outcome is not None:
                is_anomaly = getattr(outcome, "is_anomaly", False)
                anomaly_score = getattr(outcome, "anomaly_score", 0.0)
                predicted_value = getattr(outcome, "predicted_value", None)
                trend = getattr(outcome, "trend", "stable")
            audit_trace_id = getattr(explanation, "audit_trace_id", None)
        
        return DecisionContext(
            series_id=document_id,
            severity=severity,
            confidence=confidence,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score or 0.0,
            patterns=pattern_dicts,
            predicted_value=predicted_value,
            trend=trend,
            domain=domain,
            audit_trace_id=audit_trace_id,
        )
