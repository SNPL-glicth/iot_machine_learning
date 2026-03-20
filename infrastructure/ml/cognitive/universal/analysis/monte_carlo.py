"""Monte Carlo simulation engine — Backward compatibility facade.

DEPRECATED: Import from .monte_carlo subpackage instead.

This module re-exports MonteCarloSimulator and MonteCarloResult from the
modular monte_carlo/ subpackage for backward compatibility.

New code should import from:
    from .monte_carlo import MonteCarloSimulator, MonteCarloResult
"""

from __future__ import annotations

# Re-export from modular subpackage
from .monte_carlo import MonteCarloSimulator, MonteCarloResult

__all__ = ["MonteCarloSimulator", "MonteCarloResult"]
