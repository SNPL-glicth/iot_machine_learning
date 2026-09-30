"""Cumulative Residual (CUSUM) sub-detector — detección de acumulación y deriva gradual.

Especialista secuencial en O(1) tiempo y O(1) memoria.
Acumula discrepancias persistentes respecto a la envolvente nominal para detectar
mesetas anómalas sostenidas donde las derivadas temporales se anulan (dx/dt ≈ 0).
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from ..core.protocol import SubDetector

logger = logging.getLogger(__name__)


class CumulativeResidualDetector(SubDetector):
    """Sub-detector basado en suma acumulada de residuales (CUSUM bilateral).

    Evalúa desviaciones acumuladas respecto al régimen nominal de entrenamiento:
        S_t^+ = max(0, S_{t-1}^+ + (x_t - μ_0) - k_pos)
        S_t^- = max(0, S_{t-1}^- - (x_t - μ_0) - k_neg)

    Atributos:
        _k_factor: Factor multiplicador para el slack k.
        _h_factor: Factor multiplicador para el umbral h de decisión.
        _anti_windup_factor: Cota máxima de acumulación relativa a h para evitar retraso en recuperación.
    """

    def __init__(
        self,
        k_factor: float = 1.0,
        h_factor: float = 2.0,
        anti_windup_factor: float = 3.0,
        leak_rate: float = 0.05,
    ) -> None:
        self._k_factor = k_factor
        self._h_factor = h_factor
        self._anti_windup_factor = anti_windup_factor
        self._leak_rate = leak_rate

        self._mu: Optional[float] = None
        self._sigma: Optional[float] = None
        self._k_pos: float = 0.0
        self._k_neg: float = 0.0
        self._h: float = 1.0

        self._s_pos: float = 0.0
        self._s_neg: float = 0.0
        self._training_raw_scores: List[float] = []

    @property
    def method_name(self) -> str:
        return "cumulative_residual"

    def train(self, values: List[float], **kwargs: object) -> None:
        """Entrena el monitor CUSUM estimando μ_0, σ_0 y las tolerancias k, h."""
        import numpy as np

        valid = [v for v in values if v is not None and math.isfinite(v)]
        if len(valid) < 10:
            return

        arr = np.array(valid, dtype=float)
        self._mu = float(np.mean(arr))
        std = float(np.std(arr))
        self._sigma = std if std > 1e-6 else 1.0

        # Tolerancia holgura (slack) basada en envolvente empírica
        max_val = float(np.max(arr))
        min_val = float(np.min(arr))

        # k_pos y k_neg cubren las excursiones nominales para evitar falsos positivos
        self._k_pos = max(self._k_factor * self._sigma, max_val - self._mu)
        self._k_neg = max(self._k_factor * self._sigma, self._mu - min_val)

        # Umbral h de acumulación requerida para saturación
        self._h = max(self._h_factor * self._sigma, 1e-3)

        self._s_pos = 0.0
        self._s_neg = 0.0

        # Calcular scores continuos sobre el flujo de entrenamiento para calibración
        self._training_raw_scores = []
        sim_pos = 0.0
        sim_neg = 0.0
        for x in valid:
            cand_pos = max(0.0, sim_pos + (x - self._mu) - self._k_pos)
            cand_neg = max(0.0, sim_neg - (x - self._mu) - self._k_neg)
            cand_s = max(cand_pos, cand_neg)
            self._training_raw_scores.append(float(cand_s / self._h))
            sim_pos = min(cand_pos, self._anti_windup_factor * self._h)
            sim_neg = min(cand_neg, self._anti_windup_factor * self._h)

        logger.info(
            "cusum_detector_trained",
            extra={
                "mu": self._mu,
                "sigma": self._sigma,
                "k_pos": self._k_pos,
                "k_neg": self._k_neg,
                "h": self._h,
            },
        )

    def raw_score(self, value: float, **kwargs: object) -> Optional[float]:
        """Calcula la evidencia continua acumulada candidata sin mutar el estado."""
        if not self.is_trained:
            return None
        if not math.isfinite(value):
            return 0.0

        e = value - self._mu  # type: ignore[operator]
        cand_pos = max(0.0, self._s_pos + e - self._k_pos)
        cand_neg = max(0.0, self._s_neg - e - self._k_neg)
        cand_s = max(cand_pos, cand_neg)
        return float(cand_s / self._h)

    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        """Emite voto en [0.0, 1.0] proporcional a la saturación acumulada."""
        raw_s = self.raw_score(value, **kwargs)
        if raw_s is None:
            return None

        # Si no se gestiona con observe() externo, actualizar acumuladores internos
        if not kwargs.get("_managed_observe", False):
            self.observe(value, is_anomaly_active=(raw_s >= 1.0))

        # Función suave: 0.0 por debajo de 0.5, escala continuamente a 1.0 en 1.0
        if raw_s <= 0.5:
            return 0.0
        if raw_s >= 1.0:
            return 1.0
        return float((raw_s - 0.5) / 0.5)

    def observe(self, value: float, is_anomaly_active: bool = False, **kwargs: object) -> None:
        """Actualiza causalmente los acumuladores CUSUM tras la observación."""
        if not self.is_trained or not math.isfinite(value):
            return

        e = value - self._mu  # type: ignore[operator]
        new_pos = max(0.0, self._s_pos + e - self._k_pos)
        new_neg = max(0.0, self._s_neg - e - self._k_neg)

        # Si el valor está dentro del rango nominal y no hay anomalía, permitir fuga gradual
        if not is_anomaly_active and new_pos == 0.0 and self._s_pos > 0.0:
            self._s_pos = max(0.0, self._s_pos * (1.0 - self._leak_rate))
        else:
            self._s_pos = min(new_pos, self._anti_windup_factor * self._h)

        if not is_anomaly_active and new_neg == 0.0 and self._s_neg > 0.0:
            self._s_neg = max(0.0, self._s_neg * (1.0 - self._leak_rate))
        else:
            self._s_neg = min(new_neg, self._anti_windup_factor * self._h)

    def get_training_raw_scores(
        self, values: List[float], **kwargs: object
    ) -> List[float]:
        if hasattr(self, "_training_raw_scores") and self._training_raw_scores:
            return list(self._training_raw_scores)
        return []

    @property
    def is_trained(self) -> bool:
        return self._mu is not None and self._sigma is not None
