"""Document Analysis Service — Thin orchestrator.

Delegates to modular analysis pipeline components.
Includes optional Decision Engine for recommendations.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Module-level cache: hash(content) + content_type -> result
# Max 100 entries to prevent memory leak
_analysis_cache: Dict[str, Dict[str, Any]] = {}
_MAX_CACHE_ENTRIES = 100


def _compute_content_hash(content: str) -> str:
    """Compute MD5 hash of content for cache key."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]


def _get_cache_key(content_hash: str, content_type: str) -> str:
    """Build cache key from content hash and type."""
    return f"{content_hash}:{content_type}"


def _get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached result if exists."""
    return _analysis_cache.get(cache_key)


def _set_cached_result(cache_key: str, result: Dict[str, Any]) -> None:
    """Store result in cache with LRU eviction."""
    global _analysis_cache
    
    # Evict oldest if at capacity (simple LRU: clear half if full)
    if len(_analysis_cache) >= _MAX_CACHE_ENTRIES:
        # Remove oldest 50% of entries
        keys_to_remove = list(_analysis_cache.keys())[:_MAX_CACHE_ENTRIES // 2]
        for key in keys_to_remove:
            del _analysis_cache[key]
        logger.info(f"analysis_cache_evicted: removed {len(keys_to_remove)} entries")
    
    _analysis_cache[cache_key] = result
    logger.debug(f"analysis_cache_stored: key={cache_key[:20]}...", extra={"cache_size": len(_analysis_cache)})


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

# Graceful import - anomaly tracker optional
_ANOMALY_TRACKER_AVAILABLE = False
try:
    from iot_machine_learning.domain.ports import NullAnomalyTracker
    _ANOMALY_TRACKER_AVAILABLE = True
except Exception as e:
    logger.warning(f"anomaly_tracker_import_failed: {e}")

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

# Import plasticity components for proper feedback loop
_PLASTICITY_AVAILABLE = False
try:
    from iot_machine_learning.infrastructure.ml.cognitive.perception.record_actual_handler import (
        record_actual_dispatch,
    )
    from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
        EnginePerception,
    )
    from iot_machine_learning.infrastructure.ml.cognitive.plasticity.base import (
        PlasticityTracker,
    )
    from iot_machine_learning.infrastructure.ml.cognitive.plasticity.contextual_plasticity_tracker import (
        PlasticityContext,
    )
    from iot_machine_learning.domain.entities.plasticity.plasticity_context import (
        RegimeType,
    )
    _PLASTICITY_AVAILABLE = True
    logger.info("plasticity_components_available")
except Exception as e:
    logger.warning(f"plasticity_components_unavailable: {e}")


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
        anomaly_tracker: Optional[object] = None,
    ):
        """Initialize with optional cognitive memory and decision engine.
        
        Args:
            cognitive_memory: Optional CognitiveMemoryPort for semantic recall
            decision_engine: Optional DecisionEnginePort (auto-created if None and enabled)
            feature_flags: Optional FeatureFlags for decision engine enablement
            anomaly_tracker: Optional RecentAnomalyTrackerPort for contextual decisions
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
        
        # Anomaly tracker for contextual decisions (uses Null if None)
        self._anomaly_tracker = anomaly_tracker

    def analyze(
        self,
        document_id: str,
        content_type: str,
        normalized_payload: Dict[str, Any],
        tenant_id: str = "",
    ) -> Dict[str, Any]:
        """Analyze document and return structured result.
        
        Uses content-based caching to ensure deterministic results for
        identical document content.
        
        Args:
            document_id: Unique identifier for tracking
            content_type: Hint for content type (text, tabular, mixed, etc.)
            normalized_payload: Pre-processed document payload
            tenant_id: Multi-tenant isolation
            
        Returns:
            Dict with analysis, conclusion, confidence, comparative context
        """
        start = time.time()
        
        # Extract raw data for cache key
        raw_data = extract_raw_data(normalized_payload, content_type)
        content_str = str(raw_data) if not isinstance(raw_data, str) else raw_data
        content_hash = _compute_content_hash(content_str)
        cache_key = _get_cache_key(content_hash, content_type)
        
        # Check cache first
        cached = _get_cached_result(cache_key)
        if cached is not None:
            logger.info(f"analysis_cache_hit: document_id={document_id}, key={cache_key[:20]}...")
            # Return cached result with updated document_id and processing time
            result = cached.copy()
            result["document_id"] = document_id
            result["processing_time_ms"] = round((time.time() - start) * 1000, 2)
            result["cached"] = True
            return result
        
        logger.info(f"analysis_cache_miss: document_id={document_id}, key={cache_key[:20]}...")
        
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
                if _PLASTICITY_AVAILABLE:
                    try:
                        # Build EnginePerception objects from analysis results
                        perceptions = _build_analysis_perceptions(
                            winner_result=winner_result,
                            universal_result=universal_result,
                        )
                        
                        # Determine regime from domain and structural analysis
                        regime = _determine_regime_for_analysis(winner_result, universal_result)
                        
                        # Create plasticity context
                        plasticity_context = _build_plasticity_context(
                            winner_result=winner_result,
                            regime=regime,
                        )
                        
                        # Initialize plasticity tracker with persistence
                        plasticity_tracker = PlasticityTracker()
                        
                        # Build error history manager (series-scoped for isolation)
                        from iot_machine_learning.infrastructure.ml.cognitive.monitoring.error_history import (
                            create_error_history_manager,
                        )
                        error_history = create_error_history_manager(max_history=50)
                        
                        # Record actual outcome using proper dispatch
                        # Actual value: 1.0 for critical severity (feedback signal)
                        actual_value = 1.0 if (
                            getattr(winner_result, 'severity', None) and 
                            getattr(winner_result.severity, 'severity', None) == 'critical'
                        ) else 0.0
                        
                        record_actual_dispatch(
                            actual_value=actual_value,
                            last_regime=regime,
                            last_perceptions=perceptions,
                            last_plasticity_context=plasticity_context,
                            enable_advanced_plasticity=False,  # Use legacy path for text analysis
                            plasticity_coordinator=None,
                            plasticity_tracker=plasticity_tracker,
                            error_history=error_history,
                            storage=None,  # No storage adapter in this context
                            series_id=document_id,
                            series_context=None,
                        )
                        
                        logger.info(
                            "plasticity_feedback_recorded",
                            extra={
                                "document_id": document_id,
                                "regime": regime,
                                "n_perceptions": len(perceptions),
                                "actual_value": actual_value,
                            }
                        )
                    except AttributeError as e:
                        logger.error(
                            "plasticity_interface_mismatch",
                            extra={
                                "document_id": document_id,
                                "error": str(e),
                                "expected_method": "record_actual_dispatch",
                                "fix_required": "Ensure all plasticity components are properly imported",
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            "plasticity_feedback_failed",
                            extra={"document_id": document_id, "error": str(e)}
                        )
                else:
                    logger.debug(f"plasticity_unavailable_skip_feedback: {document_id}")
                
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

            # Store in cache before returning
            result_to_cache = {
                "document_id": document_id,
                "content_type": content_type,
                **result,
                "processing_time_ms": round((time.time() - start) * 1000, 2),
            }
            _set_cached_result(cache_key, result_to_cache)
            
            return result_to_cache
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
        
        Enriched with contextual data from RecentAnomalyTracker and
        SignalProfile for contextual decision scoring.
        
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
        
        # Contextual enrichment fields (Paso 4)
        current_regime = "STABLE"
        drift_score = 0.0
        
        if explanation is not None:
            outcome = getattr(explanation, "outcome", None)
            if outcome is not None:
                is_anomaly = getattr(outcome, "is_anomaly", False)
                anomaly_score = getattr(outcome, "anomaly_score", 0.0)
                predicted_value = getattr(outcome, "predicted_value", None)
                trend = getattr(outcome, "trend", "stable")
            audit_trace_id = getattr(explanation, "audit_trace_id", None)
            
            # Extract SignalProfile data for contextual enrichment
            signal_profile = getattr(explanation, "signal", None)
            if signal_profile is not None:
                # Get regime from signal profile (could be enum or string)
                regime_attr = getattr(signal_profile, "regime", None)
                if regime_attr is not None:
                    current_regime = getattr(regime_attr, "value", str(regime_attr))
                # Get drift score if available
                drift_score = getattr(signal_profile, "drift_score", 0.0)
        
        # Get anomaly tracker (use Null if not configured)
        tracker = self._anomaly_tracker
        if tracker is None and _ANOMALY_TRACKER_AVAILABLE:
            tracker = NullAnomalyTracker()
        
        # Query anomaly statistics from tracker
        recent_anomaly_count = 0
        consecutive_anomalies = 0
        recent_anomaly_rate = 0.0
        
        if tracker is not None:
            recent_anomaly_count = tracker.get_count_last_n_minutes(document_id, 120)
            consecutive_anomalies = tracker.get_consecutive_count(document_id)
            recent_anomaly_rate = tracker.get_anomaly_rate(document_id, 120)
        
        # Record anomaly or normal to tracker (after building context, before decision)
        if tracker is not None:
            if is_anomaly:
                tracker.record_anomaly(document_id, anomaly_score, regime=current_regime)
            else:
                tracker.record_normal(document_id)
        
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
            # Contextual enrichment (Paso 4)
            recent_anomaly_count=recent_anomaly_count,
            recent_anomaly_rate=recent_anomaly_rate,
            consecutive_anomalies=consecutive_anomalies,
            current_regime=current_regime,
            drift_score=drift_score,
        )
    
    def _build_analysis_perceptions(
        self,
        winner_result: object,
        universal_result: object,
    ) -> list:
        """Build EnginePerception objects from text analysis results.
        
        Args:
            winner_result: The winning analysis result
            universal_result: Universal analysis result with all sub-analyses
            
        Returns:
            List of EnginePerception objects for plasticity tracking
        """
        perceptions = []
        
        if not _PLASTICITY_AVAILABLE:
            return perceptions
        
        # Extract scores from universal_result or winner_result
        analysis = getattr(universal_result, 'analysis', {}) or {}
        
        # Urgency perception
        urgency_score = analysis.get('urgency', {}).get('score', 0.0)
        urgency_severity = analysis.get('urgency', {}).get('severity', 'info')
        if urgency_score > 0:
            perceptions.append(
                EnginePerception(
                    engine_name="text_urgency",
                    predicted_value=urgency_score,
                    confidence=0.7,  # Urgency has moderate confidence
                    trend="stable",
                    metadata={
                        "severity": urgency_severity,
                        "type": "urgency",
                    }
                )
            )
        
        # Sentiment perception
        sentiment_score = analysis.get('sentiment', {}).get('score', 0.0)
        sentiment_label = analysis.get('sentiment', {}).get('label', 'neutral')
        if sentiment_score != 0.0:
            perceptions.append(
                EnginePerception(
                    engine_name="text_sentiment",
                    predicted_value=abs(sentiment_score),  # Use absolute for severity correlation
                    confidence=0.6,
                    trend="stable",
                    metadata={
                        "label": sentiment_label,
                        "type": "sentiment",
                    }
                )
            )
        
        # Pattern perception (if available)
        patterns = analysis.get('patterns', [])
        if patterns:
            pattern_confidence = 0.5
            if isinstance(patterns, list) and len(patterns) > 0:
                pattern_confidence = min(0.9, 0.5 + (len(patterns) * 0.1))
            perceptions.append(
                EnginePerception(
                    engine_name="text_pattern",
                    predicted_value=1.0 if patterns else 0.0,
                    confidence=pattern_confidence,
                    trend="stable",
                    metadata={
                        "n_patterns": len(patterns) if isinstance(patterns, list) else 0,
                        "type": "pattern",
                    }
                )
            )
        
        # Domain/context perception
        domain = getattr(universal_result, 'domain', 'general')
        perceptions.append(
            EnginePerception(
                engine_name="text_domain",
                predicted_value=0.5,  # Neutral baseline
                confidence=0.5,
                trend="stable",
                metadata={
                    "domain": domain,
                    "type": "context",
                }
            )
        )
        
        logger.debug(
            "analysis_perceptions_built",
            extra={
                "n_perceptions": len(perceptions),
                "engines": [p.engine_name for p in perceptions],
            }
        )
        
        return perceptions
    
    def _determine_regime_for_analysis(
        self,
        winner_result: object,
        universal_result: object,
    ) -> str:
        """Determine regime from analysis results.
        
        Args:
            winner_result: The winning analysis result
            universal_result: Universal analysis result
            
        Returns:
            Regime string for plasticity tracking
        """
        # Default to domain-based regime
        domain = getattr(universal_result, 'domain', 'general')
        
        # Map domain to regime
        domain_regime_map = {
            'infrastructure': 'STABLE',
            'security': 'VOLATILE',
            'operations': 'TRENDING',
            'business': 'STABLE',
            'general': 'STABLE',
        }
        
        regime = domain_regime_map.get(domain, 'STABLE')
        
        # Override based on structural analysis if available
        structural = getattr(universal_result, 'structural', None)
        if structural and hasattr(structural, 'regime'):
            structural_regime = structural.regime
            if structural_regime:
                regime = structural_regime.upper()
        
        return regime
    
    def _build_plasticity_context(
        self,
        winner_result: object,
        regime: str,
    ) -> Optional[object]:
        """Build PlasticityContext for advanced plasticity tracking.
        
        Args:
            winner_result: The winning analysis result
            regime: Detected regime string
            
        Returns:
            PlasticityContext object or None if unavailable
        """
        if not _PLASTICITY_AVAILABLE:
            return None
        
        try:
            # Map regime string to RegimeType enum
            regime_map = {
                'STABLE': RegimeType.STABLE,
                'TRENDING': RegimeType.TRENDING,
                'VOLATILE': RegimeType.VOLATILE,
                'NOISY': RegimeType.NOISY,
            }
            regime_type = regime_map.get(regime, RegimeType.STABLE)
            
            # Extract volatility from analysis if available
            volatility = 0.5  # Default medium volatility
            structural = getattr(winner_result, 'structural', None)
            if structural and hasattr(structural, 'volatility'):
                volatility = structural.volatility
            
            # Build context
            from datetime import datetime
            context = PlasticityContext(
                regime=regime_type,
                noise_ratio=0.3,  # Default for text analysis
                volatility=volatility,
                time_of_day=datetime.now().hour,
                consecutive_failures=0,
                error_magnitude=0.0,
                is_critical_zone=False,
                timestamp=datetime.now(),
            )
            
            return context
        except Exception as e:
            logger.warning(f"plasticity_context_build_failed: {e}")
            return None


# ─── DEPRECADO ────────────────────────────────────────────────────────────────
# Este archivo está en proceso de refactoring.
# Usar: application.analyze_document.AnalyzeDocumentUseCase
# Fecha estimada de eliminación: Semana 2 del refactor
# ──────────────────────────────────────────────────────────────────────────────
import warnings
warnings.warn(
    "DocumentAnalyzer está deprecado. Usa AnalyzeDocumentUseCase.",
    DeprecationWarning,
    stacklevel=2,
)
