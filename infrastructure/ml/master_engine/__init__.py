"""Master Engine module (ZENIN v2.2 specification).

Single decision entrypoint and Master Equation orchestration layer.
"""

from __future__ import annotations

from .master_equation import (
    MasterEquationComponents,
    compute_certeza,
    compute_magnitud_objetivo,
    compute_master_equation,
    compute_momentum_veto,
)
from .orchestrator import MasterEquationOrchestrator
from .port import MasterDecisionPort

__all__ = [
    "MasterDecisionPort",
    "MasterEquationOrchestrator",
    "MasterEquationComponents",
    "compute_certeza",
    "compute_magnitud_objetivo",
    "compute_momentum_veto",
    "compute_master_equation",
]
