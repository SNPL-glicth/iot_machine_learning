"""Monte Carlo simulation engine for uncertainty quantification.

Modular components for probabilistic severity analysis.
"""

from .types import MonteCarloResult
from .simulator import MonteCarloSimulator

__all__ = [
    "MonteCarloResult",
    "MonteCarloSimulator",
]
