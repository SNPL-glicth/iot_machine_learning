"""Document Analysis Service — Thin orchestrator.

Delegates to modular analysis pipeline components.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from .analysis import (
    analyze_with_universal,
    analyze_with_legacy,
    build_output_dict,
    extract_raw_data,
    analyze_with_neural,
    arbitrate_results,
    extract_analysis_scores,
)

logger = logging.getLogger(__name__)

# Graceful import - fall back to legacy analyzers if universal engines unavailable
_UNIVERSAL_AVAILABLE = False
try:
    from iot_machine_learning.infrastructure.ml.cognitive.universal import (
        UniversalAnalysisEngine,
        UniversalComparativeEngine,
    )
    _UNIVERSAL_AVAILABLE = True
    logger.info("universal_engines_available")
except Exception as e:
    logger.warning(f"universal_engines_unavailable_using_legacy_fallback: {e}")

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

    def __init__(self, cognitive_memory: Optional[object] = None):
        """Initialize with optional cognitive memory for comparative analysis.
        
        Args:
            cognitive_memory: Optional CognitiveMemoryPort for semantic recall
        """
        self._cognitive_memory = cognitive_memory
        self._analysis_engine = UniversalAnalysisEngine() if _UNIVERSAL_AVAILABLE else None
        self._comparative_engine = UniversalComparativeEngine() if _UNIVERSAL_AVAILABLE else None

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

        try:
            if _UNIVERSAL_AVAILABLE and self._analysis_engine:
                # Step 1: Run universal analysis
                universal_result, comparison_result = analyze_with_universal(
                    document_id=document_id,
                    content_type=content_type,
                    payload=normalized_payload,
                    tenant_id=tenant_id,
                    analysis_engine=self._analysis_engine,
                    comparative_engine=self._comparative_engine,
                    cognitive_memory=self._cognitive_memory,
                )
                
                # Step 2: Run neural analysis (if available)
                neural_result = None
                if _NEURAL_AVAILABLE and self._neural_engine:
                    # Extract scores from universal result
                    analysis_scores = extract_analysis_scores(universal_result)
                    domain = getattr(universal_result, 'domain', 'unknown')
                    
                    neural_result = analyze_with_neural(
                        analysis_scores=analysis_scores,
                        input_type=content_type,
                        domain=domain,
                        neural_engine=self._neural_engine,
                    )
                
                # Step 3: Arbitrate between neural and universal
                winner_result = universal_result
                winner_engine = "universal"
                arbiter_reason = "neural_unavailable"
                
                if neural_result is not None and self._neural_arbiter:
                    winner_result, winner_engine, arbiter_reason = arbitrate_results(
                        neural_result=neural_result,
                        universal_result=universal_result,
                        domain=getattr(universal_result, 'domain', 'unknown'),
                        arbiter=self._neural_arbiter,
                    )
                
                # Step 4: Build output from winner
                raw_data = extract_raw_data(normalized_payload, content_type)
                result = build_output_dict(winner_result, comparison_result, raw_data)
                
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
