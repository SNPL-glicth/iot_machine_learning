"""VarianceEstimator — online per-engine observation variance for Bayesian updates.

SOLID extraction: variance estimation is a separate responsibility from
Bayesian posterior computation.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class VarianceEstimator:
    """Estimates sigma2_obs per engine using a rolling window of recent errors.

    Keeps deques in memory only (no Redis/SQL) as required by the spec.
    """

    def __init__(
        self,
        window: int = 20,
        min_samples: int = 5,
        min_sigma2: float = 0.01,
        default_sigma2: float = 1.0,
    ) -> None:
        self._window = window
        self._min_samples = min_samples
        self._min_sigma2 = min_sigma2
        self._default_sigma2 = default_sigma2
        self._deques: Dict[str, deque] = {}

    def record_error(self, engine_name: str, error: float) -> None:
        """Append an absolute prediction error for the given engine."""
        if engine_name not in self._deques:
            self._deques[engine_name] = deque(maxlen=self._window)
        self._deques[engine_name].append(float(error))

    def get_sigma2_obs(self, engine_name: str) -> float:
        """Return estimated sigma2_obs for engine.

        Uses empirical variance if enough samples; otherwise falls back to
        default and logs the decision structurally.
        """
        dq = self._deques.get(engine_name)
        n = len(dq) if dq else 0

        if n < self._min_samples:
            logger.info(
                "sigma2_obs_fallback",
                extra={
                    "event": "sigma2_obs_estimated",
                    "engine": engine_name,
                    "sigma2_obs": self._default_sigma2,
                    "source": "fallback",
                    "n_samples": n,
                },
            )
            return self._default_sigma2

        var = float(np.var(list(dq), ddof=0))
        sigma2 = max(self._min_sigma2, var)
        logger.info(
            "sigma2_obs_empirical",
            extra={
                "event": "sigma2_obs_estimated",
                "engine": engine_name,
                "sigma2_obs": round(sigma2, 6),
                "source": "empirical",
                "n_samples": n,
            },
        )
        return sigma2
