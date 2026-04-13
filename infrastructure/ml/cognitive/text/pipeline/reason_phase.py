"""Text reason phase: inhibit unreliable engines, fuse, classify severity."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

from iot_machine_learning.infrastructure.ml.cognitive.inhibition import InhibitionGate
from iot_machine_learning.infrastructure.ml.cognitive.fusion import WeightedFusion
from iot_machine_learning.infrastructure.ml.cognitive.text.perception_collector import DEFAULT_TEXT_WEIGHTS
from iot_machine_learning.infrastructure.ml.cognitive.text.severity_mapper import classify_text_severity

logger = logging.getLogger(__name__)

_PLASTICITY_AVAILABLE = True
try:
    from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker import (
        BayesianWeightTracker as PlasticityTracker,
    )
except (ImportError, ModuleNotFoundError):
    PlasticityTracker = None  # type: ignore[assignment,misc]


class TextReasonPhase:
    """Phase 4: Inhibit unreliable engines, fuse, classify severity."""

    def __init__(self, enable_plasticity: bool = True) -> None:
        self._inhibition = InhibitionGate()
        self._fusion = WeightedFusion()
        self._plasticity = (
            PlasticityTracker() if enable_plasticity and PlasticityTracker is not None
            else None
        )

    def execute(
        self,
        perceptions: List[Any],
        domain: str,
        document_id: str,
        urgency_score: float,
        urgency_severity: str,
        sentiment_label: str,
        full_text: str,
        impact_result: Any,
        timing: Dict[str, float],
    ) -> Tuple[float, float, str, Dict[str, float], str, str, Any, Dict[str, Any]]:
        """Execute reason phase.

        Args:
            perceptions: List of EnginePerception objects
            domain: Classified domain
            document_id: Document identifier
            urgency_score: Urgency score
            urgency_severity: Urgency severity
            sentiment_label: Sentiment label
            full_text: Complete document text
            impact_result: Impact detection result
            timing: Pipeline timing dict

        Returns:
            Tuple of (fused_val, fused_conf, fused_trend, final_weights, selected, reason, severity, phase_summaries)
        """
        phase_summaries: List[Dict[str, Any]] = []

        # Adapt: get base weights from plasticity or defaults
        t0 = time.monotonic()
        engine_names = [p.engine_name for p in perceptions]
        base_weights = self._get_base_weights(domain, engine_names)
        adapted = self._plasticity is not None and self._plasticity.has_history(domain)
        adapt_ms = (time.monotonic() - t0) * 1000
        timing["adapt"] = adapt_ms
        phase_summaries.append({
            "kind": "adapt",
            "summary": {"regime": domain, "adapted": adapted},
            "duration_ms": adapt_ms,
        })

        # Pre-fusion signal audit: Log all input signals for contradiction detection
        logger.info(
            "pre_fusion_signal_audit",
            extra={
                "document_id": document_id,
                "urgency_score": urgency_score,
                "urgency_severity": urgency_severity,
                "sentiment_label": sentiment_label,
                "domain": domain,
                "n_perceptions": len(perceptions),
                "perception_engines": [p.engine_name for p in perceptions],
                "perception_stabilities": [getattr(p, 'stability', None) for p in perceptions],
                "perception_confidences": [getattr(p, 'confidence', None) for p in perceptions],
            }
        )

        # Inhibit: suppress unreliable sub-analyzers
        t0 = time.monotonic()
        inh_states = self._inhibition.compute(
            perceptions, base_weights, series_id=document_id,
        )
        inhibit_ms = (time.monotonic() - t0) * 1000
        timing["inhibit"] = inhibit_ms
        
        # Log inhibition decisions with details
        n_inhibited = sum(1 for s in inh_states if s.suppression_factor > 0.01)
        inhibited_engines = [
            {"engine": s.engine_name, "reason": s.inhibition_reason, "factor": round(s.suppression_factor, 3)}
            for s in inh_states if s.suppression_factor > 0.01
        ]
        
        if n_inhibited > 0:
            logger.warning(
                "engines_inhibited",
                extra={
                    "document_id": document_id,
                    "n_inhibited": n_inhibited,
                    "inhibited_engines": inhibited_engines,
                }
            )
        
        phase_summaries.append({
            "kind": "inhibit",
            "summary": {
                "n_inhibited": n_inhibited,
                "inhibited_engines": inhibited_engines,
            },
            "duration_ms": inhibit_ms,
        })

        # Fuse: combine sub-analyzer scores
        t0 = time.monotonic()
        (
            fused_val, fused_conf, fused_trend,
            final_weights, selected, reason,
        ) = self._fusion.fuse(perceptions, inh_states)

        fusion_method = (
            "weighted_average" if len(perceptions) > 1 else "single_engine"
        )

        # Contradiction detection: High urgency + Stable pattern/Normal context
        contradiction_detected = self._detect_contradiction(
            perceptions=perceptions,
            urgency_score=urgency_score,
            final_weights=final_weights,
        )
        
        if contradiction_detected:
            # Apply suppression to urgency weight in final_weights
            if "text_urgency" in final_weights:
                original_urgency_weight = final_weights["text_urgency"]
                final_weights["text_urgency"] = original_urgency_weight * 0.5  # 50% suppression
                
                # Re-normalize weights
                total = sum(final_weights.values())
                if total > 0:
                    final_weights = {k: v / total for k, v in final_weights.items()}
                
                logger.warning(
                    "contradiction_detected_urgency_suppressed",
                    extra={
                        "document_id": document_id,
                        "urgency_score": urgency_score,
                        "original_weight": round(original_urgency_weight, 3),
                        "suppressed_weight": round(final_weights["text_urgency"], 3),
                        "contradiction_type": "high_urgency_vs_stable_context",
                    }
                )

        # Severity classification (3-axis: urgency + sentiment + impact)
        # Note: If contradiction detected, consider lowering urgency_score for classification
        adjusted_urgency_score = urgency_score
        if contradiction_detected:
            adjusted_urgency_score = urgency_score * 0.7  # Apply 30% reduction
            logger.info(
                "severity_classification_urgency_adjusted",
                extra={
                    "document_id": document_id,
                    "original_urgency": urgency_score,
                    "adjusted_urgency": adjusted_urgency_score,
                    "adjustment_reason": "contradiction_detected",
                }
            )
        
        severity = classify_text_severity(
            urgency_score=adjusted_urgency_score,
            urgency_severity=urgency_severity,
            sentiment_label=sentiment_label,
            has_critical_keywords=urgency_severity == "critical",
            domain=domain,
            full_text=full_text,
            impact_result=impact_result,
        )

        fuse_ms = (time.monotonic() - t0) * 1000
        timing["fuse"] = fuse_ms
        phase_summaries.append({
            "kind": "fuse",
            "summary": {
                "selected_engine": selected,
                "fused_confidence": round(fused_conf, 4),
                "severity": severity.severity,
            },
            "duration_ms": fuse_ms,
        })

        return fused_val, fused_conf, fused_trend, final_weights, selected, reason, severity, phase_summaries

    def _get_base_weights(
        self, domain: str, engine_names: List[str],
    ) -> Dict[str, float]:
        """Get base weights from plasticity or defaults."""
        if self._plasticity is not None:
            try:
                pw = self._plasticity.get_weights(domain, engine_names)
                if pw:
                    return pw
            except Exception:
                pass
        return {name: DEFAULT_TEXT_WEIGHTS.get(name, 0.2) for name in engine_names}

    def _detect_contradiction(
        self,
        perceptions: List[Any],
        urgency_score: float,
        final_weights: Dict[str, float],
    ) -> bool:
        """Detect contradiction between high urgency and stable/normal context.
        
        Condition: urgency_score > 0.8 AND (pattern == stable OR context == normal)
        
        Args:
            perceptions: List of EnginePerception objects
            urgency_score: Urgency score from text analysis
            final_weights: Final fused weights per engine
            
        Returns:
            True if contradiction detected, False otherwise
        """
        # Check urgency threshold
        if urgency_score <= 0.8:
            return False
        
        # Look for pattern perception
        pattern_stable = False
        context_normal = False
        
        for p in perceptions:
            # Check pattern engine for stability
            if p.engine_name == "text_pattern" and hasattr(p, 'metadata'):
                pattern_metadata = p.metadata or {}
                pattern_regime = pattern_metadata.get('regime', '')
                if pattern_regime == 'stable' or p.predicted_value < 0.3:
                    pattern_stable = True
            
            # Check domain/context engine for normal status
            if p.engine_name == "text_domain":
                domain = p.metadata.get('domain', 'general') if p.metadata else 'general'
                if domain in ['general', 'infrastructure', 'business']:
                    context_normal = True
        
        # Contradiction detected if high urgency but stable pattern or normal context
        contradiction = pattern_stable or context_normal
        
        if contradiction:
            logger.info(
                "contradiction_detection_analysis",
                extra={
                    "urgency_score": urgency_score,
                    "pattern_stable": pattern_stable,
                    "context_normal": context_normal,
                    "contradiction_detected": True,
                }
            )
        
        return contradiction
