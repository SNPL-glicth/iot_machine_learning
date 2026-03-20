"""Outcome tracker — records per-domain win history."""

from __future__ import annotations

from typing import Dict, Optional


class OutcomeTracker:
    """Tracks which engine wins per domain over time.
    
    Stores win counts and computes win rates for each domain.
    Used by PlasticityTracker to adjust base weights.
    """
    
    def __init__(self) -> None:
        # Per-domain win counts: {domain: {"neural": N, "universal": M}}
        self.domain_wins: Dict[str, Dict[str, int]] = {}
    
    def record_win(
        self,
        domain: str,
        winner: str,
    ) -> None:
        """Record a win for an engine in a domain.
        
        Args:
            domain: Domain identifier
            winner: "neural" or "universal"
        """
        if domain not in self.domain_wins:
            self.domain_wins[domain] = {"neural": 0, "universal": 0}
        
        if winner in self.domain_wins[domain]:
            self.domain_wins[domain][winner] += 1
    
    def get_win_rate(
        self,
        domain: str,
        engine: str,
    ) -> float:
        """Get win rate for an engine in a domain.
        
        Args:
            domain: Domain identifier
            engine: "neural" or "universal"
            
        Returns:
            Win rate [0, 1] or 0.5 if no history
        """
        if domain not in self.domain_wins:
            return 0.5  # No history - assume equal
        
        wins = self.domain_wins[domain]
        total = wins.get("neural", 0) + wins.get("universal", 0)
        
        if total == 0:
            return 0.5
        
        return wins.get(engine, 0) / total
    
    def get_neural_win_rate(self, domain: str) -> float:
        """Get neural win rate for domain.
        
        Args:
            domain: Domain identifier
            
        Returns:
            Neural win rate [0, 1]
        """
        return self.get_win_rate(domain, "neural")
    
    def get_universal_win_rate(self, domain: str) -> float:
        """Get universal win rate for domain.
        
        Args:
            domain: Domain identifier
            
        Returns:
            Universal win rate [0, 1]
        """
        return self.get_win_rate(domain, "universal")
    
    def get_total_wins(self, domain: str) -> int:
        """Get total number of decisions for domain.
        
        Args:
            domain: Domain identifier
            
        Returns:
            Total decisions
        """
        if domain not in self.domain_wins:
            return 0
        
        wins = self.domain_wins[domain]
        return wins.get("neural", 0) + wins.get("universal", 0)
    
    def has_history(self, domain: str) -> bool:
        """Check if domain has decision history.
        
        Args:
            domain: Domain identifier
            
        Returns:
            True if domain has history
        """
        return domain in self.domain_wins and self.get_total_wins(domain) > 0
    
    def get_preferred_engine(
        self,
        domain: str,
        min_decisions: int = 10,
    ) -> Optional[str]:
        """Get preferred engine for domain based on win history.
        
        Args:
            domain: Domain identifier
            min_decisions: Minimum decisions before declaring preference
            
        Returns:
            "neural", "universal", or None if insufficient data
        """
        if not self.has_history(domain):
            return None
        
        total = self.get_total_wins(domain)
        if total < min_decisions:
            return None
        
        neural_rate = self.get_neural_win_rate(domain)
        universal_rate = self.get_universal_win_rate(domain)
        
        # Require significant difference (>60%)
        if neural_rate > 0.6:
            return "neural"
        elif universal_rate > 0.6:
            return "universal"
        
        return None  # Too close to call
    
    def to_dict(self) -> dict:
        """Serialize to dictionary.
        
        Returns:
            Dict representation
        """
        return {
            domain: {
                "neural_wins": wins.get("neural", 0),
                "universal_wins": wins.get("universal", 0),
                "total": wins.get("neural", 0) + wins.get("universal", 0),
                "neural_rate": self.get_neural_win_rate(domain),
                "universal_rate": self.get_universal_win_rate(domain),
            }
            for domain, wins in self.domain_wins.items()
        }
