"""Backward-compatible facade for the Taylor math package.

This module re-exports all public symbols from the ``taylor/`` package
so that existing callers (``from .taylor_math import ...``) continue
to work without modification.

New code should import directly from ``taylor/`` sub-modules:
    from .taylor import estimate_derivatives, project, compute_diagnostic
    from .taylor.types import TaylorCoefficients, TaylorDiagnostic

Mathematical documentation is in each sub-module's docstring:
    taylor/types.py       — data structures
    taylor/derivatives.py — backward, central, least-squares methods
    taylor/polynomial.py  — projection + local fit error
    taylor/diagnostics.py — stability analysis
    taylor/time_step.py   — robust Δt estimation
"""

from __future__ import annotations

from typing import List

# Re-export canonical API from taylor/ package
from .taylor import (  # noqa: F401
    DerivativeMethod,
    TaylorCoefficients,
    TaylorDiagnostic,
    compute_diagnostic,
    compute_dt,
    compute_local_fit_error,
    estimate_derivatives,
    project,
)
from .taylor.diagnostics import (  # noqa: F401
    compute_accel_variance,
    compute_stability_indicator,
)


# ---------------------------------------------------------------------------
# Backward-compatible aliases (legacy callers)
# ---------------------------------------------------------------------------

def compute_finite_differences(
    values: List[float], dt: float, order: int,
) -> dict:
    """Legacy wrapper — returns plain dict.  Use ``estimate_derivatives``."""
    return estimate_derivatives(values, dt, order).to_dict()


def taylor_expand(derivs: dict, h: float, order: int) -> float:
    """Legacy wrapper — accepts plain dict.  Use ``project``."""
    coeffs = TaylorCoefficients(
        f_t=derivs["f_t"],
        f_prime=derivs["f_prime"],
        f_double_prime=derivs["f_double_prime"],
        f_triple_prime=derivs["f_triple_prime"],
    )
    return project(coeffs, h, order)
