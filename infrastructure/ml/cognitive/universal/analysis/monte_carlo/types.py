"""Monte Carlo result types.

Data structures for uncertainty quantification results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class MonteCarloResult:
    """Result of Monte Carlo uncertainty simulation.
    
    Attributes:
        n_simulations: Number of simulations executed (typically 1000)
        severity_distribution: Probability distribution over severity levels
            Example: {"critical": 0.73, "warning": 0.21, "info": 0.06}
        confidence_interval: (lower, upper) bounds at 95% confidence
            Tuple of severity scores [0, 1]
        expected_severity: Most probable severity level
        confidence_score: Probability of expected_severity
        scenario_outcomes: Future scenario projections
            {"best_case": {...}, "worst_case": {...}, "most_likely": {...}}
        uncertainty_level: "low" | "moderate" | "high"
    """
    n_simulations: int
    severity_distribution: Dict[str, float]
    confidence_interval: Tuple[float, float]
    expected_severity: str
    confidence_score: float
    scenario_outcomes: Dict[str, Any]
    uncertainty_level: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        return {
            "n_simulations": self.n_simulations,
            "severity_distribution": self.severity_distribution,
            "confidence_interval": {
                "lower": round(self.confidence_interval[0], 4),
                "upper": round(self.confidence_interval[1], 4),
            },
            "expected_severity": self.expected_severity,
            "confidence_score": round(self.confidence_score, 4),
            "scenario_outcomes": self.scenario_outcomes,
            "uncertainty_level": self.uncertainty_level,
        }
