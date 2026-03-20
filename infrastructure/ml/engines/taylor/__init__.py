"""Taylor prediction engine — modular mathematical foundation.

Package structure:
    engine.py       — TaylorPredictionEngine orchestrator
    adapter.py      — DEPRECATED TaylorPredictionAdapter + KalmanFilterAdapter
    math.py         — Backward-compat facade re-exporting all math functions
    types.py        — TaylorCoefficients, TaylorDiagnostic, DerivativeMethod
    derivatives.py  — backward, central, least-squares derivative estimators
    polynomial.py   — Taylor polynomial projection + local fit error
    diagnostics.py  — stability analysis, acceleration variance
    time_step.py    — robust Δt estimation from timestamps
    least_squares.py — Least-squares derivative estimation
"""

from .engine import TaylorPredictionEngine
from .adapter import TaylorPredictionAdapter, KalmanFilterAdapter
from .derivatives import estimate_derivatives
from .diagnostics import compute_diagnostic
from .polynomial import compute_local_fit_error, project
from .time_step import compute_dt
from .types import DerivativeMethod, TaylorCoefficients, TaylorDiagnostic

__all__ = [
    "TaylorPredictionEngine",
    "TaylorPredictionAdapter",
    "KalmanFilterAdapter",
    "DerivativeMethod",
    "TaylorCoefficients",
    "TaylorDiagnostic",
    "compute_diagnostic",
    "compute_dt",
    "compute_local_fit_error",
    "estimate_derivatives",
    "project",
]
