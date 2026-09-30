"""Rolling Z-score sub-detector — detección de drift por ventana corta vs larga móvil.

Compara la media de una ventana corta reciente contra la media de una
ventana larga móvil (últimos N puntos). Detecta desplazamientos de régimen
(gradual drift) que los métodos globales fijos pierden.

Incorpora compuerta Freeze-on-Alert para prevenir la asimilación de fallas
prolongadas en la media móvil.

Sin sklearn, sin I/O.
"""

from __future__ import annotations

import logging
import math
from collections import deque

from core.parameters.numerical_constants import EPSILON

from ..core.protocol import SubDetector
from ..scoring.functions import compute_z_score, compute_z_vote

logger = logging.getLogger(__name__)


class RollingZScoreDetector(SubDetector):
    """Sub-detector basado en Z-score de ventana corta vs larga móvil con Freeze-on-Alert.

    Atributos:
        _short_window: Tamaño de la ventana corta (puntos recientes).
        _long_window: Tamaño de la ventana larga móvil.
        _lower: Z-score debajo del cual el voto es 0.
        _upper: Z-score encima del cual el voto es 1.
        _freeze_on_alert: Activa congelamiento de la ventana larga durante alertas.
        _long_history: Historial de la ventana larga móvil (línea base nominal).
        _short_history: Historial de la ventana corta reciente.
    """

    def __init__(
        self,
        short_window: int = 10,  # v1.0 production config
        long_window: int = 400,  # v1.0 production config
        lower: float = 3.5,  # v1.0 production config
        upper: float = 3.5,  # v1.0 production config
        hysteresis: int = 3,  # v1.0 production config
        freeze_on_alert: bool = True,
        cooldown_steps: int = 3,
    ) -> None:
        self._short_window = short_window
        self._long_window = long_window
        self._lower = lower
        self._upper = upper
        self._hysteresis = max(1, hysteresis)
        self._freeze_on_alert = freeze_on_alert
        self._cooldown_steps = max(1, cooldown_steps)

        # Historiales desacoplados para freeze-on-alert
        max_long = max(500, self._long_window + 100)
        self._long_history: deque[float] = deque(maxlen=max_long)
        self._short_history: deque[float] = deque(maxlen=self._short_window)
        # Alias para compatibilidad con introspección legacy
        self._value_history = self._long_history

        self._consecutive_count: int = 0
        self._is_frozen: bool = False
        self._nominal_count: int = 0
        self._audit_z_scores: list[float] = []

        logger.info(
            f"RollingZ init: long={long_window}, short={short_window}, hyst={hysteresis}, "
            f"z={upper}, freeze={freeze_on_alert}"
        )

    @property
    def method_name(self) -> str:
        return "rolling_z"

    @property
    def is_frozen(self) -> bool:
        """Indica si la ventana larga se encuentra actualmente congelada."""
        return self._is_frozen

    def train(self, values: list[float], **kwargs: object) -> None:
        n = len(values)
        if n < self._long_window:
            return

        self._long_history.clear()
        self._long_history.extend(values)

        self._short_history.clear()
        self._short_history.extend(values[-self._short_window:])

        self._is_frozen = False
        self._consecutive_count = 0
        self._nominal_count = 0

        # Calcular puntuaciones continuas secuenciales sobre el flujo de entrenamiento
        self._training_raw_scores: list[float] = []
        temp_history: deque[float] = deque(maxlen=self._long_window)
        for v in values:
            temp_history.append(v)
            if len(temp_history) >= self._long_window:
                long_vals = list(temp_history)
                long_mean = sum(long_vals) / len(long_vals)
                variance = sum((x - long_mean) ** 2 for x in long_vals) / max(len(long_vals) - 1, 1)
                long_std = math.sqrt(variance) if math.sqrt(variance) > EPSILON.DIVISION else EPSILON.DIVISION
                short_vals = long_vals[-self._short_window:]
                short_mean = sum(short_vals) / len(short_vals)
                sem = long_std / math.sqrt(self._short_window)
                self._training_raw_scores.append(float(compute_z_score(short_mean, long_mean, sem)))

    def get_training_raw_scores(
        self, values: list[float], **kwargs: object
    ) -> list[float]:
        """Retorna las puntuaciones continuas calculadas sobre el flujo de entrenamiento."""
        if hasattr(self, "_training_raw_scores") and self._training_raw_scores:
            return list(self._training_raw_scores)
        return []

    def _compute_stats(self, candidate_value: float | None = None) -> tuple[float, float, float, float]:
        """Calcula estadísticas contrastando ventana corta (con candidate si existe) vs larga."""
        long_values = list(self._long_history)[-self._long_window:]
        long_mean = sum(long_values) / len(long_values)
        variance = sum((v - long_mean) ** 2 for v in long_values) / max(len(long_values) - 1, 1)
        long_std = math.sqrt(variance) if math.sqrt(variance) > EPSILON.DIVISION else EPSILON.DIVISION

        if candidate_value is not None:
            short_vals = (list(self._short_history) + [candidate_value])[-self._short_window:]
        else:
            short_vals = list(self._short_history)[-self._short_window:] if self._short_history else long_values[-self._short_window:]

        short_mean = sum(short_vals) / len(short_vals)
        sem = long_std / math.sqrt(self._short_window)
        z = compute_z_score(short_mean, long_mean, sem)
        return z, short_mean, long_mean, sem

    def raw_score(self, value: float, **kwargs: object) -> float | None:
        if len(self._long_history) < self._long_window:
            return None
        z, _, _, _ = self._compute_stats(candidate_value=value)
        return float(z)

    def vote(self, value: float, **kwargs: object) -> float | None:
        if len(self._long_history) < self._long_window:
            return None

        z, short_mean, long_mean, sem = self._compute_stats(candidate_value=value)
        raw_vote = compute_z_vote(z, self._lower, self._upper)

        if raw_vote > 0:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        result = raw_vote if self._consecutive_count >= self._hysteresis else 0.0
        self._audit_z_scores.append(float(z))

        # Si no se maneja mediante observe() externo, aplicar transición aquí
        if not kwargs.get("_managed_observe", False):
            self._apply_observation(value, is_anomaly_candidate=(raw_vote > 0))

        logger.debug(
            "rolling_z_vote",
            extra={
                "value": value, "short_mean": round(short_mean, 4), "long_mean": round(long_mean, 4),
                "sem": round(sem, 4), "z": round(z, 4), "raw_vote": round(raw_vote, 4),
                "consecutive": self._consecutive_count, "hysteresis": self._hysteresis, "vote": round(result, 4),
                "frozen": self._is_frozen,
            },
        )
        return result

    def observe(self, value: float, is_anomaly_active: bool = False, **kwargs: object) -> None:
        """Actualiza el estado temporal del detector tras la decisión global del ensamble."""
        self._apply_observation(value, is_anomaly_candidate=is_anomaly_active)

    def _apply_observation(self, value: float, is_anomaly_candidate: bool) -> None:
        """Compuerta de actualización con Freeze-on-Alert."""
        self._short_history.append(value)

        if not self._freeze_on_alert:
            self._long_history.append(value)
            return

        if is_anomaly_candidate:
            self._is_frozen = True
            self._nominal_count = 0
            # Freeze: no se agrega value a _long_history para no asimilar la falla
        else:
            if self._is_frozen:
                self._nominal_count += 1
                if self._nominal_count >= self._cooldown_steps:
                    self._is_frozen = False
                    self._nominal_count = 0
                    self._long_history.append(value)
            else:
                self._long_history.append(value)

    @property
    def is_trained(self) -> bool:
        return len(self._long_history) >= self._long_window
