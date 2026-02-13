"""Filtro de Media Móvil Exponencial (EMA) con α fijo o adaptativo.

Responsabilidad ÚNICA: suavizado exponencial de señal.
Sin warmup complejo — el primer valor inicializa el estado.
Agnóstico al dominio.

Ecuación:  x_hat(t) = α · z(t) + (1 - α) · x_hat(t-1)

Variante adaptativa: α se ajusta según la innovación reciente.
Innovación alta → α sube (seguir cambios).
Innovación baja → α baja (suavizar más).
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from iot_machine_learning.infrastructure.ml.interfaces import SignalFilter

logger = logging.getLogger(__name__)


@dataclass
class EMAState:
    """Estado interno del filtro EMA para una serie.

    Attributes:
        x_hat: Estimación actual (valor suavizado).
        alpha: Factor de suavizado actual (puede variar si adaptativo).
        n_seen: Número de observaciones procesadas.
    """

    x_hat: float = 0.0
    alpha: float = 0.3
    n_seen: int = 0


class EMASignalFilter(SignalFilter):
    """Filtro EMA con α fijo.

    Ventajas sobre Kalman:
    - Sin warmup: responde desde la primera observación.
    - Sin parámetros de modelo (Q, R): solo α.
    - Más rápido para seguir cambios de régimen.

    Args:
        alpha: Factor de suavizado (0 < α ≤ 1).
            α cercano a 1 → sigue la señal (poco suavizado).
            α cercano a 0 → suaviza mucho (más inercia).
    """

    def __init__(self, alpha: float = 0.3) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha debe estar en (0, 1], recibido {alpha}")

        self._alpha: float = alpha
        self._states: Dict[str, EMAState] = {}
        self._lock: threading.Lock = threading.Lock()

    def filter_value(self, series_id: str, value: float) -> float:
        with self._lock:
            state = self._states.get(series_id)

            if state is None:
                state = EMAState(x_hat=value, alpha=self._alpha, n_seen=1)
                self._states[series_id] = state
                return value

            state.x_hat = self._alpha * value + (1.0 - self._alpha) * state.x_hat
            state.n_seen += 1
            return state.x_hat

    def filter(
        self,
        values: List[float],
        timestamps: List[float],
    ) -> List[float]:
        if not values:
            return []

        result: List[float] = []
        x_hat = values[0]
        result.append(x_hat)

        for v in values[1:]:
            x_hat = self._alpha * v + (1.0 - self._alpha) * x_hat
            result.append(x_hat)

        return result

    def reset(self, series_id: Optional[str] = None) -> None:
        with self._lock:
            if series_id is None:
                self._states.clear()
                logger.info("ema_reset_all")
            else:
                self._states.pop(series_id, None)
                logger.info("ema_reset_series", extra={"series_id": series_id})

    def get_state(self, series_id: str) -> Optional[EMAState]:
        """Acceso al estado interno (para diagnóstico)."""
        with self._lock:
            return self._states.get(series_id)


class AdaptiveEMASignalFilter(SignalFilter):
    """Filtro EMA con α adaptativo basado en innovación.

    α se ajusta dinámicamente:
    - Innovación alta (cambio brusco) → α sube → sigue el cambio.
    - Innovación baja (señal estable) → α baja → suaviza más.

    Ecuación adaptativa:
        innovation = |z(t) - x_hat(t-1)|
        smooth_innovation = β · innovation + (1-β) · smooth_innovation_prev
        α(t) = clamp(smooth_innovation / scale, α_min, α_max)

    Args:
        alpha_min: Mínimo α (máximo suavizado).
        alpha_max: Máximo α (mínimo suavizado).
        beta: Factor de suavizado para la innovación (0 < β ≤ 1).
        scale: Escala de normalización de la innovación.
    """

    def __init__(
        self,
        alpha_min: float = 0.05,
        alpha_max: float = 0.5,
        beta: float = 0.2,
        scale: float = 1.0,
    ) -> None:
        if not (0.0 < alpha_min < alpha_max <= 1.0):
            raise ValueError(
                f"Se requiere 0 < alpha_min < alpha_max ≤ 1, "
                f"recibido alpha_min={alpha_min}, alpha_max={alpha_max}"
            )
        if not (0.0 < beta <= 1.0):
            raise ValueError(f"beta debe estar en (0, 1], recibido {beta}")
        if scale <= 0:
            raise ValueError(f"scale debe ser > 0, recibido {scale}")

        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._beta = beta
        self._scale = scale
        self._states: Dict[str, _AdaptiveEMAState] = {}
        self._lock: threading.Lock = threading.Lock()

    def filter_value(self, series_id: str, value: float) -> float:
        with self._lock:
            state = self._states.get(series_id)

            if state is None:
                alpha_init = (self._alpha_min + self._alpha_max) / 2.0
                state = _AdaptiveEMAState(
                    x_hat=value,
                    alpha=alpha_init,
                    smooth_innovation=0.0,
                    n_seen=1,
                )
                self._states[series_id] = state
                return value

            innovation = abs(value - state.x_hat)
            state.smooth_innovation = (
                self._beta * innovation
                + (1.0 - self._beta) * state.smooth_innovation
            )

            raw_alpha = state.smooth_innovation / self._scale
            state.alpha = max(self._alpha_min, min(self._alpha_max, raw_alpha))

            state.x_hat = state.alpha * value + (1.0 - state.alpha) * state.x_hat
            state.n_seen += 1
            return state.x_hat

    def filter(
        self,
        values: List[float],
        timestamps: List[float],
    ) -> List[float]:
        if not values:
            return []

        result: List[float] = []
        x_hat = values[0]
        result.append(x_hat)

        alpha = (self._alpha_min + self._alpha_max) / 2.0
        smooth_innov = 0.0

        for v in values[1:]:
            innovation = abs(v - x_hat)
            smooth_innov = self._beta * innovation + (1.0 - self._beta) * smooth_innov
            raw_alpha = smooth_innov / self._scale
            alpha = max(self._alpha_min, min(self._alpha_max, raw_alpha))

            x_hat = alpha * v + (1.0 - alpha) * x_hat
            result.append(x_hat)

        return result

    def reset(self, series_id: Optional[str] = None) -> None:
        with self._lock:
            if series_id is None:
                self._states.clear()
                logger.info("adaptive_ema_reset_all")
            else:
                self._states.pop(series_id, None)
                logger.info(
                    "adaptive_ema_reset_series",
                    extra={"series_id": series_id},
                )

    def get_state(self, series_id: str) -> Optional[_AdaptiveEMAState]:
        """Acceso al estado interno (para diagnóstico)."""
        with self._lock:
            return self._states.get(series_id)


@dataclass
class _AdaptiveEMAState:
    """Estado interno del filtro EMA adaptativo.

    Attributes:
        x_hat: Estimación actual.
        alpha: α actual (adaptado).
        smooth_innovation: Innovación suavizada (EMA de |z - x_hat|).
        n_seen: Observaciones procesadas.
    """

    x_hat: float = 0.0
    alpha: float = 0.3
    smooth_innovation: float = 0.0
    n_seen: int = 0
