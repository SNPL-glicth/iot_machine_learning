"""Engine arbitrator for selecting between neural and universal engines.

Extracted from document_analyzer.py as part of refactoring Paso 2.
Responsible for deciding which engine wins based on domain and available results.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class EngineArbitrator:
    """Arbitrates between neural and universal engine results.
    
    Analyzes both results and selects the winner based on:
    - Domain appropriateness
    - Confidence scores
    - Energy efficiency (for neural)
    - Consistency of results
    
    Args:
        neural_arbiter: Optional NeuralArbiter instance for advanced arbitration.
    """
    
    def __init__(self, neural_arbiter: Optional[Any] = None) -> None:
        """Initialize arbitrator with optional neural arbiter."""
        self._neural_arbiter = neural_arbiter
    
    def arbitrate(
        self,
        neural_result: Optional[Any],
        universal_result: Any,
        domain: str,
    ) -> Tuple[Any, str, str]:
        """Arbitrate between neural and universal results.
        
        Args:
            neural_result: Result from neural engine (may be None)
            universal_result: Result from universal engine
            domain: Detected domain for context
            
        Returns:
            Tuple of (winner_result, winner_engine_name, arbitration_reason)
        """
        # If neural not available, use universal
        if neural_result is None:
            return universal_result, "universal", "neural_unavailable_or_disabled"
        
        # If no neural arbiter, simple comparison
        if self._neural_arbiter is None:
            return universal_result, "universal", "neural_arbiter_not_configured"
        
        try:
            # Use neural arbiter for sophisticated selection
            winner_result, winner_engine, arbiter_reason = self._neural_arbiter.arbitrate(
                neural_result=neural_result,
                universal_result=universal_result,
                domain=domain,
            )
            return winner_result, winner_engine, arbiter_reason
        except Exception as e:
            logger.warning(f"neural_arbitration_failed: {e}, falling back to universal")
            return universal_result, "universal", f"arbitration_error: {e}"
    
    def extract_analysis_scores(self, universal_result: Any) -> Dict[str, Any]:
        """Extract scores from universal result for neural analysis.
        
        Args:
            universal_result: Universal engine result
            
        Returns:
            Dict with analysis scores for neural processing
        """
        scores = {}
        
        # Extract from analysis if available
        if hasattr(universal_result, 'analysis') and universal_result.analysis:
            analysis = universal_result.analysis
            if isinstance(analysis, dict):
                # Copy relevant scores
                for key in ['urgency', 'sentiment', 'confidence', 'patterns']:
                    if key in analysis:
                        scores[key] = analysis[key]
        
        # Extract domain
        if hasattr(universal_result, 'domain'):
            scores['domain'] = universal_result.domain
        
        # Extract severity
        if hasattr(universal_result, 'severity'):
            scores['severity'] = universal_result.severity
        
        return scores
