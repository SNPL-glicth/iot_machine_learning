"""Plasticity feedback loop for recording analysis outcomes.

Extracted from document_analyzer.py as part of refactoring Paso 3.
Responsible for recording actual outcomes and updating plasticity weights.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PlasticityFeedbackLoop:
    """Handles plasticity feedback for document analysis.
    
    Builds perceptions, determines regime, creates plasticity context,
    and dispatches to the actual recording system.
    
    Args:
        plasticity_tracker: Optional tracker for weight updates.
        plasticity_coordinator: Optional coordinator for advanced tracking.
        error_history: Optional error history manager.
        storage: Optional storage adapter for persistence.
        enable_advanced: Whether to use advanced plasticity features.
    """
    
    def __init__(
        self,
        plasticity_tracker: Optional[Any] = None,
        plasticity_coordinator: Optional[Any] = None,
        error_history: Optional[Any] = None,
        storage: Optional[Any] = None,
        enable_advanced: bool = False,
        deterministic_mode: bool = False,
    ) -> None:
        """Initialize feedback loop with optional components."""
        self._plasticity_tracker = plasticity_tracker
        self._plasticity_coordinator = plasticity_coordinator
        self._error_history = error_history
        self._storage = storage
        self._enable_advanced = enable_advanced
        self._deterministic_mode = deterministic_mode
    
    def record_feedback(
        self,
        document_id: str,
        winner_result: Any,
        universal_result: Any,
    ) -> bool:
        """Record feedback from analysis result.
        
        Args:
            document_id: Document identifier
            winner_result: The winning analysis result
            universal_result: Universal analysis result
            
        Returns:
            True if feedback was recorded successfully, False otherwise.
        """
        try:
            # Skip plasticity update if deterministic mode is enabled
            if self._deterministic_mode:
                logger.debug(
                    "plasticity_update_skipped_deterministic_mode",
                    extra={"document_id": document_id}
                )
                return True
            
            # Build perceptions from results
            perceptions = self._build_perceptions(winner_result, universal_result)
            
            # Determine regime
            regime = self._determine_regime(winner_result, universal_result)
            
            # Create plasticity context
            context = self._build_context(winner_result, regime)
            
            # Calculate actual value (1.0 for critical severity)
            actual_value = self._calculate_actual_value(winner_result)
            
            # Dispatch to recording system
            self._dispatch_record(
                actual_value=actual_value,
                regime=regime,
                perceptions=perceptions,
                context=context,
                document_id=document_id,
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
            return True
            
        except AttributeError as e:
            logger.error(
                "plasticity_interface_mismatch",
                extra={
                    "document_id": document_id,
                    "error": str(e),
                    "expected_method": "record_actual_dispatch",
                }
            )
            return False
            
        except Exception as e:
            logger.warning(
                "plasticity_feedback_failed",
                extra={"document_id": document_id, "error": str(e)}
            )
            return False
    
    def _build_perceptions(
        self,
        winner_result: Any,
        universal_result: Any,
    ) -> List[Dict[str, Any]]:
        """Build perception dicts from analysis results."""
        perceptions = []
        
        # Extract analysis data
        analysis = getattr(universal_result, 'analysis', {}) or {}
        if not isinstance(analysis, dict):
            analysis = {}
        
        # Urgency perception
        urgency = analysis.get('urgency', {})
        urgency_score = urgency.get('score', 0.0)
        if urgency_score > 0:
            perceptions.append({
                "engine_name": "text_urgency",
                "predicted_value": urgency_score,
                "confidence": 0.7,
                "trend": "stable",
                "metadata": {
                    "severity": urgency.get('severity', 'info'),
                    "type": "urgency",
                }
            })
        
        # Sentiment perception
        sentiment = analysis.get('sentiment', {})
        sentiment_score = sentiment.get('score', 0.0)
        if sentiment_score != 0.0:
            perceptions.append({
                "engine_name": "text_sentiment",
                "predicted_value": abs(sentiment_score),
                "confidence": 0.6,
                "trend": "stable",
                "metadata": {
                    "label": sentiment.get('label', 'neutral'),
                    "type": "sentiment",
                }
            })
        
        # Pattern perception
        patterns = analysis.get('patterns', [])
        if patterns:
            pattern_confidence = min(0.9, 0.5 + (len(patterns) * 0.1))
            perceptions.append({
                "engine_name": "text_pattern",
                "predicted_value": 1.0 if patterns else 0.0,
                "confidence": pattern_confidence,
                "trend": "stable",
                "metadata": {
                    "n_patterns": len(patterns) if isinstance(patterns, list) else 0,
                    "type": "pattern",
                }
            })
        
        # Domain perception
        domain = getattr(universal_result, 'domain', 'general')
        perceptions.append({
            "engine_name": "text_domain",
            "predicted_value": 0.5,
            "confidence": 0.5,
            "trend": "stable",
            "metadata": {
                "domain": domain,
                "type": "context",
            }
        })
        
        return perceptions
    
    def _determine_regime(self, winner_result: Any, universal_result: Any) -> str:
        """Determine regime from analysis results."""
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
                regime = str(structural_regime).upper()
        
        return regime
    
    def _build_context(self, winner_result: Any, regime: str) -> Optional[Dict[str, Any]]:
        """Build plasticity context dict."""
        try:
            # Extract volatility
            volatility = 0.5  # Default
            structural = getattr(winner_result, 'structural', None)
            if structural and hasattr(structural, 'volatility'):
                volatility = structural.volatility
            
            from datetime import datetime
            
            return {
                "regime": regime,
                "noise_ratio": 0.3,
                "volatility": volatility,
                "time_of_day": datetime.now().hour,
                "consecutive_failures": 0,
                "error_magnitude": 0.0,
                "is_critical_zone": False,
                "timestamp": datetime.now(),
            }
        except Exception as e:
            logger.warning(f"plasticity_context_build_failed: {e}")
            return None
    
    def _calculate_actual_value(self, winner_result: Any) -> float:
        """Calculate actual value from result (1.0 for critical)."""
        severity = getattr(winner_result, 'severity', None)
        if severity and hasattr(severity, 'severity'):
            if severity.severity == 'critical':
                return 1.0
        return 0.0
    
    def _dispatch_record(
        self,
        actual_value: float,
        regime: str,
        perceptions: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
        document_id: str,
    ) -> None:
        """Dispatch to recording system."""
        # Try to import and use record_actual_dispatch if available
        try:
            from iot_machine_learning.infrastructure.ml.cognitive.perception.record_actual_handler import (
                record_actual_dispatch,
            )
            from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
                EnginePerception,
            )
            from iot_machine_learning.infrastructure.ml.cognitive.plasticity.contextual_plasticity_tracker import (
                SignalContext,
            )
            from iot_machine_learning.domain.entities.plasticity.signal_context import (
                RegimeType,
            )
            
            # Convert perceptions to EnginePerception objects
            engine_perceptions = []
            for p in perceptions:
                engine_perceptions.append(
                    EnginePerception(
                        engine_name=p["engine_name"],
                        predicted_value=p["predicted_value"],
                        confidence=p["confidence"],
                        trend=p["trend"],
                        metadata=p.get("metadata", {}),
                    )
                )
            
            # Map regime string to RegimeType
            regime_map = {
                'STABLE': RegimeType.STABLE,
                'TRENDING': RegimeType.TRENDING,
                'VOLATILE': RegimeType.VOLATILE,
                'NOISY': RegimeType.NOISY,
            }
            regime_type = regime_map.get(regime, RegimeType.STABLE)
            
            # Build SignalContext if we have context dict
            plasticity_context = None
            if context:
                plasticity_context = SignalContext(
                    regime=regime_type,
                    noise_ratio=context.get("noise_ratio", 0.3),
                    volatility=context.get("volatility", 0.5),
                    time_of_day=context.get("time_of_day", 0),
                    consecutive_failures=context.get("consecutive_failures", 0),
                    error_magnitude=context.get("error_magnitude", 0.0),
                    is_critical_zone=context.get("is_critical_zone", False),
                    timestamp=context.get("timestamp"),
                )
            
            # Call the actual dispatch function
            record_actual_dispatch(
                actual_value=actual_value,
                last_regime=regime,
                last_perceptions=engine_perceptions,
                last_signal_context=plasticity_context,
                enable_advanced_plasticity=self._enable_advanced,
                plasticity_coordinator=self._plasticity_coordinator,
                plasticity_tracker=self._plasticity_tracker,
                error_history=self._error_history,
                storage=self._storage,
                series_id=document_id,
                series_context=None,
            )
            
        except ImportError as e:
            logger.debug(f"plasticity_dispatch_unavailable: {e}")
        except Exception as e:
            logger.warning(f"plasticity_dispatch_failed: {e}")
