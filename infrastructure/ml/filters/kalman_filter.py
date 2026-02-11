"""Filtro de Kalman 1D con auto-calibración de R y warmup.

Responsabilidad ÚNICA: orquestar warmup + filtering con thread-safety.
Delega math puro a kalman_math.py. Agnóstico al dominio.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from iot_machine_learning.infrastructure.ml.interfaces import SignalFilter

from .kalman_math import (
    KalmanState,
    WarmupBuffer,
    initialize_state,
    kalman_update,
)

logger = logging.getLogger(__name__)


class KalmanSignalFilter(SignalFilter):
    """Filtro de Kalman 1D con warmup y auto-calibración de R.

    Fases de operación por serie:
    1. **Warmup**: acumula valores, retorna valor crudo. Al completar, calibra R.
    2. **Filtering**: aplica Kalman update y retorna x_hat filtrado.
    """

    def __init__(self, Q: float = 1e-5, warmup_size: int = 10) -> None:
        if Q <= 0:
            raise ValueError(f"Q debe ser > 0, recibido {Q}")
        if warmup_size < 2:
            raise ValueError(f"warmup_size debe ser >= 2, recibido {warmup_size}")

        self._Q: float = Q
        self._warmup_size: int = warmup_size
        self._states: Dict[str, KalmanState] = {}
        self._warmup_buffers: Dict[str, WarmupBuffer] = {}
        self._lock: threading.Lock = threading.Lock()

    def filter_value(self, series_id: str, value: float) -> float:
        with self._lock:
            state = self._states.get(series_id)

            if state is None or not state.initialized:
                return self._handle_warmup(series_id, value)

            filtered = kalman_update(state, value)
            logger.debug(
                "kalman_update",
                extra={
                    "series_id": series_id,
                    "phase": "filtering",
                    "measurement": value,
                    "x_hat": state.x_hat,
                    "P": state.P,
                },
            )
            return filtered

    def filter(
        self,
        values: List[float],
        timestamps: List[float],
    ) -> List[float]:
        if not values:
            return []

        result: List[float] = []
        warmup_vals: List[float] = []
        state: Optional[KalmanState] = None

        for v in values:
            if state is None or not state.initialized:
                warmup_vals.append(v)
                result.append(v)

                if len(warmup_vals) >= self._warmup_size:
                    state = initialize_state(warmup_vals, self._Q)
            else:
                filtered = kalman_update(state, v)
                result.append(filtered)

        return result

    def reset(self, series_id: Optional[str] = None) -> None:
        with self._lock:
            if series_id is None:
                self._states.clear()
                self._warmup_buffers.clear()
                logger.info("kalman_reset_all")
            else:
                self._states.pop(series_id, None)
                self._warmup_buffers.pop(series_id, None)
                logger.info(
                    "kalman_reset_series",
                    extra={"series_id": series_id},
                )

    def get_state(self, series_id: str) -> Optional[KalmanState]:
        with self._lock:
            return self._states.get(series_id)

    def is_initialized(self, series_id: str) -> bool:
        with self._lock:
            state = self._states.get(series_id)
            return state is not None and state.initialized

    def _handle_warmup(self, series_id: str, value: float) -> float:
        if series_id not in self._warmup_buffers:
            self._warmup_buffers[series_id] = WarmupBuffer(
                values=[], target_size=self._warmup_size
            )

        buf = self._warmup_buffers[series_id]
        buf.values.append(value)

        logger.debug(
            "kalman_warmup",
            extra={
                "series_id": series_id,
                "phase": "warmup",
                "warmup_progress": f"{len(buf.values)}/{buf.target_size}",
            },
        )

        if buf.is_ready:
            state = initialize_state(buf.values, self._Q)
            self._states[series_id] = state
            del self._warmup_buffers[series_id]

            logger.info(
                "kalman_initialized",
                extra={
                    "series_id": series_id,
                    "x_hat": state.x_hat,
                    "P": state.P,
                    "R": state.R,
                    "Q": state.Q,
                },
            )

        return value
