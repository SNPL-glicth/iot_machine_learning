"""Confidence score comparison for engine arbitration."""

from __future__ import annotations

from typing import Optional, Tuple


class ConfidenceComparator:
    """Compares confidence scores between engines.
    
    Args:
        neural_margin: Margin neural needs to beat universal (default 0.1)
    """
    
    def __init__(self, neural_margin: float = 0.1) -> None:
        self.neural_margin = neural_margin
    
    def compare(
        self,
        neural_confidence: float,
        universal_confidence: float,
    ) -> Tuple[str, float, str]:
        """Compare confidence scores.
        
        Args:
            neural_confidence: Neural engine confidence [0, 1]
            universal_confidence: Universal engine confidence [0, 1]
            
        Returns:
            Tuple of (winner, winning_confidence, reason)
        """
        # Neural must beat universal by margin
        if neural_confidence > universal_confidence + self.neural_margin:
            return "neural", neural_confidence, f"neural_confidence={neural_confidence:.3f} > universal={universal_confidence:.3f} + margin={self.neural_margin}"
        
        # Otherwise universal wins
        return "universal", universal_confidence, f"universal_confidence={universal_confidence:.3f} >= neural={neural_confidence:.3f} + margin={self.neural_margin}"
    
    def compare_with_monte_carlo(
        self,
        neural_confidence: float,
        universal_confidence: float,
        neural_monte_carlo: Optional[object],
        universal_monte_carlo: Optional[object],
    ) -> Tuple[str, float, str]:
        """Compare confidence with Monte Carlo consistency check.
        
        If both engines have Monte Carlo results, check if their
        uncertainty estimates are consistent. Prefer engine with
        lower uncertainty.
        
        Args:
            neural_confidence: Neural engine confidence
            universal_confidence: Universal engine confidence
            neural_monte_carlo: Neural Monte Carlo result (optional)
            universal_monte_carlo: Universal Monte Carlo result (optional)
            
        Returns:
            Tuple of (winner, winning_confidence, reason)
        """
        # If no Monte Carlo, fall back to simple comparison
        if neural_monte_carlo is None and universal_monte_carlo is None:
            return self.compare(neural_confidence, universal_confidence)
        
        # Extract uncertainty metrics
        neural_uncertainty = self._extract_uncertainty(neural_monte_carlo)
        universal_uncertainty = self._extract_uncertainty(universal_monte_carlo)
        
        # If both have uncertainty, use it as tiebreaker
        if neural_uncertainty is not None and universal_uncertainty is not None:
            # Lower uncertainty is better
            if neural_confidence > universal_confidence + self.neural_margin:
                # Neural wins on confidence
                if neural_uncertainty < universal_uncertainty:
                    return "neural", neural_confidence, f"neural_confidence={neural_confidence:.3f} + lower_uncertainty={neural_uncertainty:.3f}"
                else:
                    return "neural", neural_confidence, f"neural_confidence={neural_confidence:.3f} (but higher_uncertainty={neural_uncertainty:.3f})"
            
            # Universal wins or tie - check uncertainty
            if universal_uncertainty < neural_uncertainty:
                return "universal", universal_confidence, f"universal_confidence={universal_confidence:.3f} + lower_uncertainty={universal_uncertainty:.3f}"
            else:
                # Neural has lower uncertainty but not enough confidence margin
                # Still give to universal (confidence is primary)
                return "universal", universal_confidence, f"universal_confidence={universal_confidence:.3f} (neural has lower_uncertainty but insufficient margin)"
        
        # Fallback to simple comparison
        return self.compare(neural_confidence, universal_confidence)
    
    def _extract_uncertainty(self, monte_carlo: Optional[object]) -> Optional[float]:
        """Extract uncertainty metric from Monte Carlo result.
        
        Args:
            monte_carlo: Monte Carlo result object
            
        Returns:
            Uncertainty score or None
        """
        if monte_carlo is None:
            return None
        
        # Try to get uncertainty classification
        if hasattr(monte_carlo, 'uncertainty_class'):
            # Map to numeric: very_low=0.1, low=0.3, moderate=0.5, high=0.7, very_high=0.9
            mapping = {
                "very_low": 0.1,
                "low": 0.3,
                "moderate": 0.5,
                "high": 0.7,
                "very_high": 0.9,
            }
            return mapping.get(monte_carlo.uncertainty_class, 0.5)
        
        # Try confidence interval width
        if hasattr(monte_carlo, 'ci_width'):
            return monte_carlo.ci_width
        
        # Try std deviation
        if hasattr(monte_carlo, 'std'):
            return monte_carlo.std
        
        return None
