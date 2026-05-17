"""MoEAlertService — alertas basadas en métricas en tiempo real.

Alertas configuradas:
- fallback rate > 10% en ventana de 5 min → ERROR
- discrepancy score > 3σ del baseline → WARNING
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, Dict, Optional

logger = logging.getLogger("moe.alerts")


class MoEAlertService:
    """Servicio de alertas con ventanas temporales deslizantes.

    Args:
        fallback_window_s: Ventana para calcular fallback rate (default 300s).
        fallback_threshold: Umbral de fallback rate para alerta ERROR (default 0.10).
        discrepancy_z_threshold: Umbral de σ para alerta WARNING (default 3.0).
    """

    def __init__(
        self,
        fallback_window_s: float = 300.0,
        fallback_threshold: float = 0.10,
        discrepancy_z_threshold: float = 3.0,
    ) -> None:
        self._fallback_window_s = fallback_window_s
        self._fallback_threshold = fallback_threshold
        self._discrepancy_z_threshold = discrepancy_z_threshold

        self._fallback_events: Deque[float] = deque()
        self._discrepancy_scores: Deque[float] = deque()
        self._discrepancy_baseline: Optional[tuple[float, float]] = None

    def record_fallback(self) -> None:
        """Registra un evento de fallback."""
        now = time.time()
        self._fallback_events.append(now)
        self._prune_fallback_window(now)

        total = len(self._fallback_events)
        # Asumimos ~1 predicción por segundo por ventana como proxy
        # El caller puede pasar total_predictions si lo conoce
        rate = self._calculate_fallback_rate(now, total_estimated=total + 10)
        if rate > self._fallback_threshold:
            logger.error(
                "moe_alert_fallback_rate_high",
                extra={
                    "rate": round(rate, 4),
                    "threshold": self._fallback_threshold,
                    "window_s": self._fallback_window_s,
                    "fallback_count": total,
                },
            )

    def record_discrepancy(self, score: float) -> None:
        """Registra un score de discrepancia y verifica si excede 3σ."""
        self._discrepancy_scores.append(score)
        if len(self._discrepancy_scores) > 1000:
            self._discrepancy_scores.popleft()

        mean, std = self._get_discrepancy_baseline()
        if std < 1e-9:
            return

        z_score = abs(score - mean) / std
        if z_score > self._discrepancy_z_threshold:
            logger.warning(
                "moe_alert_discrepancy_high",
                extra={
                    "score": round(score, 4),
                    "z_score": round(z_score, 2),
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "threshold_sigma": self._discrepancy_z_threshold,
                },
            )

    def _prune_fallback_window(self, now: float) -> None:
        cutoff = now - self._fallback_window_s
        while self._fallback_events and self._fallback_events[0] < cutoff:
            self._fallback_events.popleft()

    def _calculate_fallback_rate(self, now: float, total_estimated: int) -> float:
        self._prune_fallback_window(now)
        if total_estimated == 0:
            return 0.0
        return len(self._fallback_events) / total_estimated

    def _get_discrepancy_baseline(self) -> tuple[float, float]:
        if len(self._discrepancy_scores) < 20:
            return 0.0, 0.0
        scores = list(self._discrepancy_scores)
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        return mean, std
