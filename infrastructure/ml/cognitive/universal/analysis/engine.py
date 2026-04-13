"""UniversalAnalysisEngine — input-agnostic deep cognitive analysis."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from iot_machine_learning.domain.services.severity_rules import (
    SeverityResult,
    classify_severity_agnostic,
)
from iot_machine_learning.domain.entities.explainability.explanation import Explanation
from iot_machine_learning.infrastructure.ml.cognitive.explanation import ExplanationBuilder
from iot_machine_learning.infrastructure.ml.cognitive.pattern_interpreter.interpreter import PatternInterpreter

from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.types import UniversalInput, UniversalResult, UniversalContext, InputType
from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.pipeline import (
    PerceivePhase,
    AnalyzePhase,
    RememberPhase,
    ReasonPhase,
    ExplainPhase,
)
from .monte_carlo import MonteCarloSimulator
from .pattern_plasticity import PatternPlasticityTracker
from ..validation.coherence_validator import CoherenceValidator


logger = logging.getLogger(__name__)


class UniversalAnalysisEngine:
    """Universal cognitive engine — handles any input type.

    Pipeline:
        1. PERCEIVE — Detect type, classify domain, build signal profile
        2. ANALYZE  — Collect perceptions from type-specific sub-analyzers
        3. REMEMBER — Recall similar past analyses (optional, via CognitiveMemoryPort)
        4. REASON   — Inhibit + Adapt + Fuse (reuses existing cognitive components)
        5. EXPLAIN  — Assemble Explanation domain object

    Plasticity learns by DOMAIN (infrastructure, security, trading)
    not by data type (text vs numeric).

    Args:
        enable_plasticity: Enable regime-contextual weight learning
        enable_monte_carlo: Enable Monte Carlo uncertainty simulation
        budget_ms: Pipeline time budget in milliseconds
    """

    def __init__(
        self,
        *,
        enable_plasticity: bool = True,
        enable_monte_carlo: bool = True,
        budget_ms: float = 2000.0,
        deterministic_mode: bool = False,
        analysis_seed: int = 42,
    ) -> None:
        """Initialize universal engine with cognitive components."""
        self._perceive = PerceivePhase()
        self._analyze = AnalyzePhase()
        self._remember = RememberPhase()
        self._reason = ReasonPhase(enable_plasticity=enable_plasticity)
        self._explain = ExplainPhase()
        self._monte_carlo_simulator = MonteCarloSimulator(n_simulations=1000)
        self._pattern_plasticity = PatternPlasticityTracker()
        self._coherence_validator = CoherenceValidator()
        self._enable_monte_carlo = enable_monte_carlo
        self._budget_ms = budget_ms
        self._deterministic_mode = deterministic_mode
        self._analysis_seed = analysis_seed
        
        # Set seed if deterministic mode
        if self._deterministic_mode:
            import random
            import numpy as np
            random.seed(self._analysis_seed)
            np.random.seed(self._analysis_seed)

    def analyze(
        self,
        raw_data: Any,
        ctx: UniversalContext,
        pre_computed_scores: Optional[Dict[str, Any]] = None,
    ) -> UniversalResult:
        """Run full cognitive pipeline on any input.

        Args:
            raw_data: Any input (str, List[float], Dict, etc.)
            ctx: Pipeline configuration and environment
            pre_computed_scores: Optional pre-computed analysis scores

        Returns:
            UniversalResult with Explanation + severity + analysis dict
        """
        timing: Dict[str, float] = {}
        
        try:
            input_type, metadata, domain, signal, builder = self._perceive.execute(
                raw_data, ctx, timing
            )
            
            perceptions = self._analyze.execute(
                raw_data, input_type, metadata, pre_computed_scores, timing
            )
            
            if not perceptions:
                return self._build_fallback_result(
                    input_type, domain, signal, builder, timing, "no_perceptions"
                )
            
            # Pattern interpretation after perception collection
            interpreted_patterns = []
            try:
                interpreter = PatternInterpreter()
                interpreted_patterns = interpreter.interpret(
                    raw_patterns=pre_computed_scores.get("patterns", {}),
                    input_type=input_type.value,
                    domain=domain,
                    urgency_score=pre_computed_scores.get("urgency_score", 0.0),
                    sentiment_label=pre_computed_scores.get("sentiment_label", ""),
                )
            except Exception as e:
                logger.warning(f"pattern_interpretation_failed: {e}")
                # Graceful-fail - continue without patterns
            
            recall_ctx = self._remember.execute(raw_data, domain, ctx, timing)
            
            fused_val, fused_conf, fused_trend, final_weights, selected, reason, method = self._reason.execute(
                perceptions, domain, ctx.series_id, timing
            )
            
            # Run Monte Carlo simulation for uncertainty quantification
            # Skip if deterministic mode is enabled
            monte_carlo_result = None
            if self._enable_monte_carlo and not self._deterministic_mode:
                monte_carlo_result = self._run_monte_carlo(
                    perceptions, input_type, domain, metadata, timing
                )
            
            severity = self._classify_severity(
                input_type, domain, perceptions, metadata, fused_val
            )
            
            confidence = self._compute_confidence(
                input_type, metadata, recall_ctx is not None, perceptions, final_weights
            )
            
            analysis = self._build_analysis_dict(
                input_type, metadata, perceptions, final_weights, signal, pre_computed_scores, monte_carlo_result
            )
            
            # Build explanation BEFORE creating result (frozen dataclass)
            explanation = self._explain.execute(
                builder, fused_val, fused_conf, fused_trend,
                final_weights, selected, reason, method, timing
            )
            
            # Create final result with all fields
            final_result = UniversalResult(
                explanation=explanation,
                severity=severity,
                analysis=analysis,
                confidence=confidence,
                domain=domain,
                input_type=input_type,
                pipeline_timing=timing,
                recall_context=recall_ctx,
                patterns=interpreted_patterns,
                monte_carlo=monte_carlo_result,
            )
            
            # COHERENCE VALIDATION (after result creation)
            coherence_report = self._coherence_validator.validate(final_result)
            
            # After analysis completes, update pattern plasticity
            if self._pattern_plasticity and interpreted_patterns:
                for pattern in interpreted_patterns:
                    self._pattern_plasticity.record_pattern_outcome(
                        domain=domain,
                        pattern_name=pattern.pattern_type,
                        was_predictive=final_result.severity.severity in ["warning", "critical"]
                    )
            
            return final_result
        
        except Exception as e:
            logger.error(f"universal_analysis_failed: {e}", exc_info=True)
            return self._build_error_result(str(e))

    def _classify_severity(
        self,
        input_type: InputType,
        domain: str,
        perceptions: List,
        metadata: Dict[str, Any],
        fused_value: float,
    ) -> SeverityResult:
        """Classify severity based on input type and domain."""
        score = fused_value
        
        if input_type == InputType.TEXT:
            # For text inputs, use the TextCognitiveEngine's severity classification
            # which considers urgency, sentiment, and impact signals
            urgency_perc = next(
                (p for p in perceptions if p.engine_name == "text_urgency"), None
            )
            sentiment_perc = next(
                (p for p in perceptions if p.engine_name == "text_sentiment"), None
            )
            
            if urgency_perc and sentiment_perc:
                # Extract text-specific scores from perceptions
                urgency_score = urgency_perc.predicted_value
                sentiment_label = sentiment_perc.metadata.get("label", "neutral")
                has_critical_keywords = urgency_perc.metadata.get("severity", "info") == "critical"
                
                # Use the text-specific severity classifier
                from ...text.severity_mapper import classify_text_severity
                
                severity_result = classify_text_severity(
                    urgency_score=urgency_score,
                    urgency_severity=urgency_perc.metadata.get("severity", "info"),
                    sentiment_label=sentiment_label,
                    has_critical_keywords=has_critical_keywords,
                    domain=domain,
                    full_text="",
                    impact_result=None,
                )
                
                return severity_result
            else:
                # Fallback to agnostic classification for non-text or missing perceptions
                pass
        
        return classify_severity_agnostic(
            value=score,
            anomaly=score > 0.6,  # Consider high scores as anomalies
            threshold=None,
        )

    def _compute_confidence(
        self,
        input_type: InputType,
        metadata: Dict[str, Any],
        has_recall: bool,
        perceptions: List,
        final_weights: Dict[str, float],
    ) -> float:
        """Compute overall confidence score based on engine agreement (entropy).
        
        Real uncertainty quantification using normalized entropy of perception weights.
        Higher disagreement between engines → lower confidence.
        
        Args:
            input_type: Type of input data
            metadata: Input metadata (word_count, n_points, etc.)
            has_recall: Whether cognitive memory recall was successful
            perceptions: List of EnginePerception objects
            final_weights: Final fused weights dict {engine_name: weight}
            
        Returns:
            Confidence score [0.0, 1.0] based on engine consensus
        """
        import math
        
        # Base confidence from data quality (minimal, not dominant)
        base_confidence = 0.50
        
        # Data quality bonus (small contribution, max +0.15)
        if input_type == InputType.TEXT:
            word_count = metadata.get("word_count", 0)
            if word_count > 500:
                base_confidence += 0.15
            elif word_count > 100:
                base_confidence += 0.10
            elif word_count > 20:
                base_confidence += 0.05
        elif input_type == InputType.NUMERIC:
            n_points = metadata.get("n_points", 0)
            if n_points >= 50:
                base_confidence += 0.15
            elif n_points >= 20:
                base_confidence += 0.10
            elif n_points >= 10:
                base_confidence += 0.05
        
        # PRIMARY: Entropy-based uncertainty from engine disagreement
        if not perceptions or len(perceptions) < 2:
            # Single engine or no perceptions = uncertain
            consensus_factor = 0.5
        else:
            # Normalize weights to probabilities
            total_weight = sum(final_weights.values())
            if total_weight < 1e-12:
                consensus_factor = 0.5
            else:
                probs = [w / total_weight for w in final_weights.values()]
                
                # Compute normalized entropy: H / H_max
                # H = -Σ p_i * log(p_i)
                # H_max = log(n_engines)
                n = len(probs)
                entropy = 0.0
                for p in probs:
                    if p > 1e-12:
                        entropy -= p * math.log(p)
                
                h_max = math.log(n) if n > 1 else 1.0
                normalized_entropy = entropy / h_max if h_max > 0 else 0.0
                
                # Consensus factor = 1 - normalized_entropy
                # High entropy (disagreement) → low confidence
                # Low entropy (consensus) → high confidence
                consensus_factor = 1.0 - normalized_entropy
        
        # Inhibition penalty: if engines were suppressed, we're less confident
        n_inhibited = sum(1 for p in perceptions if getattr(p, 'inhibited', False))
        inhibition_ratio = n_inhibited / len(perceptions) if perceptions else 0.0
        inhibition_penalty = inhibition_ratio * 0.15  # Max -0.15 for all inhibited
        
        # Combine: base + consensus bonus - inhibition penalty
        # Consensus contributes up to +0.40 (strong consensus)
        # Range: [0.30, 0.95]
        confidence = base_confidence + (consensus_factor * 0.40) - inhibition_penalty
        
        # Recall bonus (small)
        if has_recall:
            confidence += 0.05
        
        # Clamp to valid range
        return max(0.10, min(0.95, confidence))
    
    def _run_monte_carlo(
        self,
        perceptions: List,
        input_type: InputType,
        domain: str,
        metadata: Dict[str, Any],
        timing: Dict[str, float],
    ) -> Optional[object]:
        """Run Monte Carlo simulation for uncertainty quantification.
        
        Args:
            perceptions: List of EnginePerception objects
            input_type: Detected input type
            domain: Classified domain
            metadata: Input metadata
            timing: Pipeline timing dict
            
        Returns:
            MonteCarloResult or None if simulation fails
        """
        t0 = time.monotonic()
        
        try:
            # Extract scores from perceptions
            analysis_scores = {}
            
            for perception in perceptions:
                if hasattr(perception, 'predicted_value'):
                    analysis_scores[perception.engine_name] = perception.predicted_value
                elif hasattr(perception, 'score'):
                    analysis_scores[perception.engine_name] = perception.score
            
            # Add metadata scores
            if input_type == InputType.TEXT:
                # Use urgency as primary score for text
                urgency_perc = next(
                    (p for p in perceptions if p.engine_name == "text_urgency"), None
                )
                if urgency_perc and hasattr(urgency_perc, 'predicted_value'):
                    analysis_scores["urgency"] = urgency_perc.predicted_value
            
            if not analysis_scores:
                logger.warning("no_scores_for_monte_carlo")
                return None
            
            # Run simulation
            result = self._monte_carlo_simulator.simulate(
                analysis_scores=analysis_scores,
                input_type=input_type,
                domain=domain,
            )
            
            elapsed = (time.monotonic() - t0) * 1000
            timing["monte_carlo"] = elapsed
            
            return result
        
        except Exception as e:
            logger.warning(f"monte_carlo_failed: {e}", exc_info=True)
            return None

    def _build_analysis_dict(
        self,
        input_type: InputType,
        metadata: Dict[str, Any],
        perceptions: List,
        final_weights: Dict[str, float],
        signal,
        pre_computed_scores: Optional[Dict[str, Any]] = None,
        monte_carlo_result: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Build backward-compatible analysis dict."""
        analysis = {
            "input_type": input_type.value,
            "metadata": metadata,
            "cognitive": {
                "engine_weights": {
                    k: round(v, 4) for k, v in final_weights.items()
                },
                "engine_perceptions": [
                    p.to_dict() for p in perceptions
                ],
                "signal_profile": signal.to_dict(),
            },
        }
        
        # Include Monte Carlo result if available
        if monte_carlo_result and hasattr(monte_carlo_result, 'to_dict'):
            analysis["monte_carlo"] = monte_carlo_result.to_dict()
        
        # Include pre_computed_scores for entity_extractor and conclusion_formatter
        if pre_computed_scores:
            analysis.update({
                "word_count": pre_computed_scores.get("word_count", 0),
                "urgency_score": pre_computed_scores.get("urgency_score", 0.0),
                "urgency_severity": pre_computed_scores.get("urgency_severity", "info"),
                "sentiment_score": pre_computed_scores.get("sentiment_score", 0.0),
                "sentiment_label": pre_computed_scores.get("sentiment_label", "neutral"),
                "paragraph_count": pre_computed_scores.get("paragraph_count", 0),
                "entities": pre_computed_scores.get("entities", []),
                "patterns": pre_computed_scores.get("patterns", {}),
            })
        
        return analysis

    def _build_fallback_result(
        self,
        input_type: InputType,
        domain: str,
        signal,
        builder,
        timing: Dict[str, float],
        reason: str,
    ) -> UniversalResult:
        """Build fallback result when pipeline fails."""
        builder.set_fallback(0.0, reason)
        explanation = builder.build()
        
        severity = classify_severity_agnostic(value=0.0, anomaly=False, threshold=None)
        
        return UniversalResult(
            explanation=explanation,
            severity=severity,
            analysis={"fallback": True, "reason": reason},
            confidence=0.5,
            domain=domain,
            input_type=input_type,
            pipeline_timing=timing,
        )

    def _build_error_result(self, error_msg: str) -> UniversalResult:
        """Build error result."""
        severity = classify_severity_agnostic(value=0.0, anomaly=False, threshold=None)
        
        return UniversalResult(
            explanation=Explanation.minimal("error"),
            severity=severity,
            analysis={"error": error_msg},
            confidence=0.0,
            domain="general",
            input_type=InputType.UNKNOWN,
        )
