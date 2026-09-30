"""Perfil de calibración y función pura de mapeo de scores a_i(x).

Transforma evidencia continua no acotada s_i en un score calibrado a_i en [0, 1].
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_EPSILON: float = 1e-9


@dataclass(frozen=True)
class DetectorCalibrationProfile:
    """Perfil de calibración empírico para un subdetector específico."""

    tau_nominal: float
    tau_anomaly: float
    is_inverted: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.tau_nominal) or not math.isfinite(self.tau_anomaly):
            raise ValueError("Los umbrales de calibración deben ser números finitos")


def compute_calibrated_score(
    raw_score: float,
    profile: DetectorCalibrationProfile,
) -> float:
    """Aplica la Ecuación de Calibración de Anomalía en O(1) tiempo de ejecución.

    Args:
        raw_score: Evidencia continua no acotada emitida por el subdetector.
        profile: Perfil empírico con referencias nominal y anómala.

    Returns:
        Score de anomalía calibrado a_i en [0.0, 1.0].
    """
    if not math.isfinite(raw_score):
        return 0.0

    if profile.is_inverted:
        # Menor valor = mayor anomalía (e.g. IsolationForest, LOF)
        # tau_nominal > tau_anomaly
        span = profile.tau_nominal - profile.tau_anomaly
        if span <= _EPSILON:
            return 1.0 if raw_score < profile.tau_anomaly else 0.0
        normalized = (profile.tau_nominal - raw_score) / span
    else:
        # Mayor valor = mayor anomalía (e.g. Z-Score, IQR, Rolling Z)
        # tau_anomaly > tau_nominal
        span = profile.tau_anomaly - profile.tau_nominal
        if span <= _EPSILON:
            return 1.0 if raw_score > profile.tau_anomaly else 0.0
        normalized = (raw_score - profile.tau_nominal) / span

    return float(max(0.0, min(1.0, normalized)))
