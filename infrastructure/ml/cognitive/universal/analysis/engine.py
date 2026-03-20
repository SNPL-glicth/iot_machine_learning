"""UniversalAnalysisEngine — input-agnostic deep cognitive analysis."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ...inhibition import InhibitionGate, InhibitionConfig
from ...fusion import WeightedFusion
from ...explanation import ExplanationBuilder
from iot_machine_learning.domain.services.severity_rules import (
    SeverityResult,
    classify_severity_agnostic,
)

from .types import UniversalInput, UniversalResult, UniversalContext, InputType
from .input_detector import detect_input_type
from .domain_classifier import classify_domain
from .signal_profiler import UniversalSignalProfiler
from .perception_collector import UniversalPerceptionCollector
from .monte_carlo import MonteCarloSimulator


logger = logging.getLogger(__name__)

_PLASTICITY_AVAILABLE = True
try:
    from ...plasticity import PlasticityTracker
except (ImportError, ModuleNotFoundError):
    _PLASTICITY_AVAILABLE = False


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
        budget_ms: Pipeline time budget in milliseconds
    """

    def __init__(
        self,
        *,
        enable_plasticity: bool = True,
        budget_ms: float = 2000.0,
    ) -> None:
        """Initialize universal engine with cognitive components."""
        self._profiler = UniversalSignalProfiler()
        self._collector = UniversalPerceptionCollector()
        self._inhibition = InhibitionGate(InhibitionConfig())
        self._fusion = WeightedFusion()
        self._monte_carlo_simulator = MonteCarloSimulator(n_simulations=1000)
        
        self._plasticity = None
        if enable_plasticity and _PLASTICITY_AVAILABLE:
            self._plasticity = PlasticityTracker()
        
        self._budget_ms = budget_ms

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
            input_type, metadata, domain, signal, builder = self._perceive(
                raw_data, ctx, timing
            )
            
            perceptions = self._analyze(
                raw_data, input_type, metadata, pre_computed_scores, timing
            )
            
            if not perceptions:
                return self._build_fallback_result(
                    input_type, domain, signal, builder, timing, "no_perceptions"
                )
            
            recall_ctx = self._remember(raw_data, domain, ctx, timing)
            
            fused_val, fused_conf, fused_trend, final_weights, selected, reason, method = self._reason(
                perceptions, domain, ctx.series_id, timing
            )
            
            explanation = self._explain(
                builder, fused_val, fused_conf, fused_trend,
                final_weights, selected, reason, method, timing
            )
            
            severity = self._classify_severity(
                input_type, domain, perceptions, metadata, fused_val
            )
            
            confidence = self._compute_confidence(
                input_type, metadata, recall_ctx is not None
            )
            
            analysis = self._build_analysis_dict(
                input_type, metadata, perceptions, final_weights, signal
            )
            
            return UniversalResult(
                explanation=explanation,
                severity=severity,
                analysis=analysis,
                confidence=confidence,
                domain=domain,
                input_type=input_type,
                pipeline_timing=timing,
                recall_context=recall_ctx,
            )
        
        except Exception as e:
            logger.error(f"universal_analysis_failed: {e}", exc_info=True)
            return self._build_error_result(str(e))

    def _perceive(
        self,
        raw_data: Any,
        ctx: UniversalContext,
        timing: Dict[str, float],
    ) -> tuple:
        """Phase 1: Detect type, classify domain, build signal."""
        t0 = time.monotonic()
        
        input_type, metadata = detect_input_type(raw_data)
        
        domain = classify_domain(
            raw_data, input_type, metadata, ctx.domain_hint
        )
        
        signal = self._profiler.profile(
            raw_data, input_type, metadata, domain
        )
        
        builder = ExplanationBuilder(ctx.series_id)
        builder.set_signal(signal)
        
        timing["perceive"] = (time.monotonic() - t0) * 1000
        
        return input_type, metadata, domain, signal, builder

    def _analyze(
        self,
        raw_data: Any,
        input_type: InputType,
        metadata: Dict[str, Any],
        pre_computed_scores: Optional[Dict[str, Any]],
        timing: Dict[str, float],
    ) -> List:
        """Phase 2: Collect perceptions from sub-analyzers."""
        t0 = time.monotonic()
        
        perceptions = self._collector.collect(
            raw_data, input_type, metadata, pre_computed_scores
        )
        
        # Apply pattern plasticity weights if enabled
        if self._plasticity:
            perceptions = self._apply_pattern_weights(
                perceptions, metadata["domain"], input_type
            )
        
        timing["analyze"] = (time.monotonic() - t0) * 1000
        
        return perceptions

    def _remember(
        self,
        raw_data: Any,
        domain: str,
        ctx: UniversalContext,
        timing: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Phase 3: Recall similar past analyses."""
        t0 = time.monotonic()
        
        recall_ctx = None
        
        if ctx.cognitive_memory:
            try:
                query = str(raw_data)[:500] if isinstance(raw_data, str) else ""
                
                if hasattr(ctx.cognitive_memory, 'recall_similar_explanations'):
                    results = ctx.cognitive_memory.recall_similar_explanations(
                        query=query,
                        series_id=ctx.series_id if ctx.series_id != "unknown" else None,
                        limit=3,
                        min_certainty=0.7,
                    )
                    
                    if results:
                        recall_ctx = {
                            "n_matches": len(results),
                            "top_score": round(results[0].score, 3) if results else 0.0,
                            "has_context": True,
                        }
            except Exception as e:
                logger.debug(f"memory_recall_failed: {e}")
        
        timing["remember"] = (time.monotonic() - t0) * 1000
        
        return recall_ctx

    def _reason(
        self,
        perceptions: List,
        domain: str,
        series_id: str,
        timing: Dict[str, float],
    ) -> tuple:
        """Phase 4: Inhibit + Adapt + Fuse."""
        t0 = time.monotonic()
        
        engine_names = [p.engine_name for p in perceptions]
        
        base_weights = {}
        if self._plasticity and self._plasticity.has_history(domain):
            base_weights = self._plasticity.get_weights(domain, engine_names)
        else:
            n = len(engine_names)
            base_weights = {name: 1.0 / n for name in engine_names}
        
        timing["adapt"] = (time.monotonic() - t0) * 1000
        
        t0 = time.monotonic()
        inh_states = self._inhibition.compute(
            perceptions, base_weights, series_id=series_id
        )
        timing["inhibit"] = (time.monotonic() - t0) * 1000
        
        t0 = time.monotonic()
        (fused_val, fused_conf, fused_trend,
         final_weights, selected, reason) = self._fusion.fuse(
            perceptions, inh_states
        )
        
        method = "weighted_average" if len(perceptions) > 1 else "single_engine"
        
        timing["fuse"] = (time.monotonic() - t0) * 1000
        
        return fused_val, fused_conf, fused_trend, final_weights, selected, reason, method

    def _explain(
        self,
        builder,
        fused_value: float,
        fused_confidence: float,
        fused_trend: str,
        final_weights: Dict[str, float],
        selected: str,
        reason: str,
        method: str,
        timing: Dict[str, float],
    ):
        """Phase 5: Assemble Explanation domain object."""
        t0 = time.monotonic()
        
        builder.set_fusion(
            fused_value, fused_confidence, fused_trend,
            final_weights, selected, reason, method
        )
        
        explanation = builder.build()
        
        timing["explain"] = (time.monotonic() - t0) * 1000
        
        return explanation

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
            urgency_perc = next(
                (p for p in perceptions if p.engine_name == "text_urgency"), None
            )
            if urgency_perc:
                score = urgency_perc.predicted_value
        
        return classify_severity_agnostic(
            value=score,
            threshold_warning=0.4,
            threshold_critical=0.7,
        )

    def _compute_confidence(
        self,
        input_type: InputType,
        metadata: Dict[str, Any],
        has_recall: bool,
    ) -> float:
        """Compute overall confidence score."""
        confidence = 0.75
        
        if input_type == InputType.TEXT:
            word_count = metadata.get("word_count", 0)
            if word_count > 100:
                confidence = 0.80
            if word_count > 500:
                confidence = 0.85
        
        if input_type == InputType.NUMERIC:
            n_points = metadata.get("n_points", 0)
            if n_points >= 20:
                confidence = 0.85
            elif n_points >= 10:
                confidence = 0.80
        
        if has_recall:
            confidence = min(0.95, confidence + 0.05)
        
        return confidence
    
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
    ) -> Dict[str, Any]:
        """Build backward-compatible analysis dict."""
        return {
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
        
        severity = classify_severity_agnostic(0.0, 0.4, 0.7)
        
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
        from iot_machine_learning.domain.entities.explainability.explanation import Explanation
        
        severity = classify_severity_agnostic(0.0, 0.4, 0.7)
        
        return UniversalResult(
            explanation=Explanation.minimal("error"),
            severity=severity,
            analysis={"error": error_msg},
            confidence=0.0,
            domain="general",
            input_type=InputType.UNKNOWN,
        )
        return UniversalResult(
            explanation=Explanation.minimal("error"),
            severity=severity,
            analysis={"error": error_msg},
            confidence=0.0,
            domain="general",
            input_type=InputType.UNKNOWN,
        )
        
        severity = classify_severity_agnostic(0.0, 0.4, 0.7)
        
        return UniversalResult(
            explanation=Explanation.minimal("error"),
            severity=severity,
            analysis={"error": error_msg},
            confidence=0.0,
            domain="general",
            input_type=InputType.UNKNOWN,
        )
