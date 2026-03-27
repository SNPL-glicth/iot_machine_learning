"""SimulatedOutcome — scenario outcome for decision evidence.

Domain-pure dataclass representing one scenario outcome from
Monte Carlo simulation or decision modeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class SimulatedOutcome:
    """One scenario outcome from Monte Carlo or decision simulation.

    Represents the expected result of taking a particular action
    under a specific scenario.

    Attributes:
        scenario_name: Identifier for the scenario (e.g., "do_nothing", "act_conservative")
        probability: Likelihood of this scenario [0, 1]
        expected_risk: Quantified risk level if this scenario occurs
        confidence_interval: (lower, upper) bounds for risk estimate
        description: Human-readable explanation of this outcome
    """

    scenario_name: str
    probability: float = 0.0
    expected_risk: float = 0.0
    confidence_interval: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "probability": round(self.probability, 4),
            "expected_risk": round(self.expected_risk, 4),
            "confidence_interval": [
                round(self.confidence_interval[0], 4),
                round(self.confidence_interval[1], 4),
            ],
            "description": self.description,
        }
