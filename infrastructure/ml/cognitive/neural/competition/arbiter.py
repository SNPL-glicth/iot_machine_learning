"""Neural arbiter — decides winner between neural and universal engines.

Uses three factors:
1. Confidence score comparison (primary)
2. Monte Carlo consistency (secondary)
3. Domain win history (tertiary)
"""

from __future__ import annotations

from typing import Tuple

from .confidence_comparator import ConfidenceComparator
from .outcome_tracker import OutcomeTracker
from ..types import NeuralResult
from ...universal.analysis.types import UniversalResult


class NeuralArbiter:
    """Arbitrates between neural and universal analysis engines.
    
    Decision factors (in order):
    1. Confidence score comparison (primary)
    2. Monte Carlo result consistency if available (secondary)
    3. Domain win history from OutcomeTracker (tertiary)
    
    Args:
        confidence_comparator: Confidence comparison logic
        outcome_tracker: Win history tracker
        history_weight: Weight given to historical win rate (0-1)
    """
    
    def __init__(
        self,
        confidence_comparator: ConfidenceComparator = None,
        outcome_tracker: OutcomeTracker = None,
        history_weight: float = 0.1,
    ) -> None:
        self.comparator = confidence_comparator or ConfidenceComparator()
        self.tracker = outcome_tracker or OutcomeTracker()
        self.history_weight = history_weight
    
    def arbitrate(
        self,
        neural_result: NeuralResult,
        universal_result: UniversalResult,
        domain: str,
    ) -> Tuple[str, float, str]:
        """Decide winner between neural and universal engines.
        
        Args:
            neural_result: Neural engine analysis result
            universal_result: Universal engine analysis result
            domain: Domain identifier
            
        Returns:
            Tuple of (winner_engine, winning_confidence, reason)
            winner_engine: "neural" or "universal"
        """
        # Factor 1: Confidence comparison with Monte Carlo
        winner, confidence, reason_parts = self.comparator.compare_with_monte_carlo(
            neural_confidence=neural_result.confidence,
            universal_confidence=universal_result.confidence,
            neural_monte_carlo=neural_result.monte_carlo,
            universal_monte_carlo=universal_result.monte_carlo,
        )
        
        # Factor 2: Check domain history (tertiary)
        if self.tracker.has_history(domain):
            neural_rate = self.tracker.get_neural_win_rate(domain)
            universal_rate = self.tracker.get_universal_win_rate(domain)
            total_decisions = self.tracker.get_total_wins(domain)
            
            # If we have strong history, apply bias
            if total_decisions >= 10:
                # Compute history bias
                if neural_rate > 0.6:
                    history_bias = self.history_weight * (neural_rate - 0.5)
                    reason_parts += f" | history_bias_neural={history_bias:.3f} ({total_decisions} decisions)"
                elif universal_rate > 0.6:
                    history_bias = self.history_weight * (universal_rate - 0.5)
                    reason_parts += f" | history_bias_universal={history_bias:.3f} ({total_decisions} decisions)"
                else:
                    reason_parts += f" | history_neutral ({total_decisions} decisions)"
        
        # Record this decision
        self.tracker.record_win(domain, winner)
        
        return winner, confidence, reason_parts
    
    def get_win_statistics(self, domain: str) -> dict:
        """Get win statistics for a domain.
        
        Args:
            domain: Domain identifier
            
        Returns:
            Dict with win counts and rates
        """
        return {
            "neural_wins": self.tracker.domain_wins.get(domain, {}).get("neural", 0),
            "universal_wins": self.tracker.domain_wins.get(domain, {}).get("universal", 0),
            "total_decisions": self.tracker.get_total_wins(domain),
            "neural_rate": self.tracker.get_neural_win_rate(domain),
            "universal_rate": self.tracker.get_universal_win_rate(domain),
            "preferred_engine": self.tracker.get_preferred_engine(domain),
        }
