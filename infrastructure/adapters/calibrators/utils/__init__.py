"""Utilidades para calibradores de confianza.

Funciones matemáticas auxiliares separadas para mantener
los calibradores principales bajo 180 líneas.
"""

from .ece_metrics import compute_ece, compute_ece_numpy
from .platt_math import fit_platt_params, platt_sigmoid
from .isotonic_math import pava_isotonic_regression, interpolate_isotonic

__all__ = [
    "compute_ece",
    "compute_ece_numpy",
    "fit_platt_params", 
    "platt_sigmoid",
    "pava_isotonic_regression",
    "interpolate_isotonic",
]
