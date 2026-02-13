"""Funciones matemáticas puras para Filtro de Kalman 1D.

Responsabilidad ÚNICA: estado de Kalman, calibración y update.
Sin I/O, sin threading, sin logging. Agnóstico al dominio.

Ecuaciones de Kalman 1D (proceso estacionario):
    Predicción:  x_pred = x_hat,  P_pred = P + Q
    Gain:        K = P_pred / (P_pred + R)
    Corrección:  x_hat = x_pred + K·(z - x_pred),  P = (1-K)·P_pred
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# Mínimo valor de R para evitar división por cero en Kalman gain
MIN_R: float = 1e-6

# Mínimo valor de P para evitar que el filtro se "congele"
MIN_P: float = 1e-10


# Tamaño por defecto de la ventana de innovación para Q adaptativo
DEFAULT_INNOVATION_WINDOW: int = 20

# Factor de escala para convertir varianza de innovación a Q
ADAPTIVE_Q_SCALE: float = 0.1

# Límites de Q adaptativo para evitar inestabilidad
MIN_Q: float = 1e-8
MAX_Q: float = 1.0


@dataclass
class KalmanState:
    """Estado interno del filtro de Kalman para una serie temporal.

    Attributes:
        x_hat: Estimación actual del valor filtrado.
        P: Covarianza del error de estimación.
        Q: Varianza del proceso.
        R: Varianza de medición (auto-calibrada).
        initialized: ``True`` cuando el warmup completó.
        _innovations: Ventana circular de innovaciones recientes
            (para Q adaptativo).  Vacía si Q es fijo.
    """

    x_hat: float = 0.0
    P: float = 1.0
    Q: float = 1e-5
    R: float = 1.0
    initialized: bool = False
    _innovations: List[float] = field(default_factory=list)
    _innovation_window_size: int = 0


@dataclass
class WarmupBuffer:
    """Buffer temporal para acumular observaciones durante warmup.

    Attributes:
        values: Observaciones acumuladas.
        target_size: Número de lecturas necesarias para completar warmup.
    """

    values: List[float] = field(default_factory=list)
    target_size: int = 10

    @property
    def is_ready(self) -> bool:
        return len(self.values) >= self.target_size


def initialize_state(warmup_values: List[float], Q: float) -> KalmanState:
    """Crea estado inicial calibrado a partir del buffer de warmup.

    Calibración:
    - x_hat = mean(warmup_values)
    - P = var(warmup_values)
    - R = max(var(warmup_values), MIN_R)

    Args:
        warmup_values: Lecturas acumuladas durante warmup.
        Q: Varianza del proceso configurada.

    Returns:
        ``KalmanState`` inicializado y calibrado.
    """
    n = len(warmup_values)
    mean_val = sum(warmup_values) / n

    if n > 1:
        variance = sum((v - mean_val) ** 2 for v in warmup_values) / (n - 1)
    else:
        variance = 0.0

    R_calibrated = max(variance, MIN_R)

    return KalmanState(
        x_hat=mean_val,
        P=max(variance, MIN_P),
        Q=Q,
        R=R_calibrated,
        initialized=True,
    )


def kalman_update(state: KalmanState, measurement: float) -> float:
    """Aplica un paso de Kalman update (Q fijo).

    Modifica ``state`` in-place.

    Args:
        state: Estado actual de la serie.
        measurement: Nueva observación.

    Returns:
        Valor filtrado (x_hat actualizado).
    """
    x_pred = state.x_hat
    P_pred = state.P + state.Q

    K = P_pred / (P_pred + state.R)

    state.x_hat = x_pred + K * (measurement - x_pred)
    state.P = max((1.0 - K) * P_pred, MIN_P)

    return state.x_hat


def adaptive_kalman_update(state: KalmanState, measurement: float) -> float:
    """Aplica un paso de Kalman update con Q adaptativo.

    Q se ajusta basándose en la varianza de las innovaciones recientes
    (diferencia entre predicción y medición).  Cuando la serie cambia
    de régimen, las innovaciones crecen → Q sube → el filtro sigue
    el cambio más rápido.

    Modifica ``state`` in-place (incluyendo Q y _innovations).

    Args:
        state: Estado actual de la serie (debe tener
            ``_innovation_window_size > 0``).
        measurement: Nueva observación.

    Returns:
        Valor filtrado (x_hat actualizado).
    """
    x_pred = state.x_hat
    innovation = measurement - x_pred

    # --- Actualizar ventana de innovaciones ---
    state._innovations.append(innovation)
    win = state._innovation_window_size or DEFAULT_INNOVATION_WINDOW
    if len(state._innovations) > win:
        state._innovations = state._innovations[-win:]

    # --- Adaptar Q si hay suficientes innovaciones ---
    if len(state._innovations) >= 3:
        state.Q = _compute_adaptive_Q(state._innovations)

    # --- Kalman update estándar con Q adaptado ---
    P_pred = state.P + state.Q
    K = P_pred / (P_pred + state.R)

    state.x_hat = x_pred + K * innovation
    state.P = max((1.0 - K) * P_pred, MIN_P)

    return state.x_hat


def _compute_adaptive_Q(innovations: List[float]) -> float:
    """Calcula Q adaptativo a partir de la varianza de innovaciones.

    Q = clamp(scale * var(innovations), MIN_Q, MAX_Q)

    Args:
        innovations: Ventana de innovaciones recientes.

    Returns:
        Q adaptado.
    """
    n = len(innovations)
    mean_innov = sum(innovations) / n
    var_innov = sum((v - mean_innov) ** 2 for v in innovations) / max(n - 1, 1)

    q = ADAPTIVE_Q_SCALE * var_innov
    return max(MIN_Q, min(MAX_Q, q))
