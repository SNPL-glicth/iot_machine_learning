"""Bridge between DocumentAnalyzer and HybridNeuralEngine.

Handles neural engine invocation and result formatting.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Graceful import - fall back if neural engine unavailable
_NEURAL_AVAILABLE = False
try:
    from iot_machine_learning.infrastructure.ml.cognitive.neural import (
        HybridNeuralEngine,
        NeuralResult,
    )
    from iot_machine_learning.infrastructure.ml.cognitive.neural.competition import (
        NeuralArbiter,
    )
    from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.types import (
        InputType,
    )
    _NEURAL_AVAILABLE = True
    logger.info("neural_engine_available")
except Exception as e:
    logger.warning(f"neural_engine_unavailable: {e}")


def analyze_with_neural(
    analysis_scores: Dict[str, float],
    input_type: str,
    domain: str,
    neural_engine: Optional[object],
) -> Optional[Any]:
    """Analyze using HybridNeuralEngine.
    
    Args:
        analysis_scores: Dict of {analyzer_name: score [0, 1]}
        input_type: Input type string
        domain: Domain identifier
        neural_engine: HybridNeuralEngine instance
        
    Returns:
        NeuralResult or None if unavailable
    """
    if not _NEURAL_AVAILABLE or neural_engine is None:
        return None
    
    try:
        # Map input type string to enum
        input_type_enum = _map_input_type(input_type)
        
        # Run neural analysis
        result = neural_engine.analyze(
            analysis_scores=analysis_scores,
            input_type=input_type_enum,
            domain=domain,
        )
        
        return result
    
    except Exception as e:
        logger.error(f"neural_analysis_failed: {e}", exc_info=True)
        return None


def arbitrate_results(
    neural_result: Optional[Any],
    universal_result: Any,
    domain: str,
    arbiter: Optional[object],
) -> tuple[Any, str, str]:
    """Arbitrate between neural and universal results.
    
    Args:
        neural_result: NeuralResult or None
        universal_result: UniversalResult
        domain: Domain identifier
        arbiter: NeuralArbiter instance
        
    Returns:
        Tuple of (winner_result, winner_engine, reason)
    """
    if not _NEURAL_AVAILABLE or neural_result is None or arbiter is None:
        return universal_result, "universal", "neural_unavailable"
    
    try:
        # Run arbitration
        winner_engine, confidence, reason = arbiter.arbitrate(
            neural_result=neural_result,
            universal_result=universal_result,
            domain=domain,
        )
        
        # Return winning result
        if winner_engine == "neural":
            return neural_result, "neural", reason
        else:
            return universal_result, "universal", reason
    
    except Exception as e:
        logger.error(f"arbitration_failed: {e}", exc_info=True)
        return universal_result, "universal", f"arbitration_error: {e}"


def extract_analysis_scores(
    universal_result: Any,
) -> Dict[str, float]:
    """Extract analysis scores from UniversalResult for neural input.
    
    Args:
        universal_result: UniversalResult with perceptions
        
    Returns:
        Dict of {analyzer_name: score}
    """
    scores = {}
    
    # Try to extract from explanation
    if hasattr(universal_result, 'explanation'):
        explanation = universal_result.explanation
        
        # Extract from engine contributions
        if hasattr(explanation, 'engine_contributions'):
            for contrib in explanation.engine_contributions:
                if hasattr(contrib, 'engine_name') and hasattr(contrib, 'predicted_value'):
                    scores[contrib.engine_name] = contrib.predicted_value
    
    # Fallback: extract from analysis dict
    if not scores and hasattr(universal_result, 'analysis'):
        analysis = universal_result.analysis
        
        # Look for scores in various keys
        for key in ['text_scores', 'numeric_scores', 'scores']:
            if key in analysis and isinstance(analysis[key], dict):
                scores.update(analysis[key])
    
    return scores


def _map_input_type(input_type_str: str) -> 'InputType':
    """Map input type string to InputType enum.
    
    Args:
        input_type_str: Input type as string
        
    Returns:
        InputType enum value
    """
    from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.types import InputType
    
    mapping = {
        "text": InputType.TEXT,
        "numeric": InputType.NUMERIC,
        "tabular": InputType.TABULAR,
        "mixed": InputType.MIXED,
        "json": InputType.JSON,
        "special_chars": InputType.SPECIAL_CHARS,
    }
    
    return mapping.get(input_type_str.lower(), InputType.UNKNOWN)
