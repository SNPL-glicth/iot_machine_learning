"""Bayesian update logic — GaussianPrior, BayesianUpdater, VarianceEstimator.

Consolidates modules previously spread across: update_mixin.py,
accuracy_mixin.py, variance_estimator.py, regularization.py,
weight_calculator.py, per_sensor_key.py.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import BayesianWeightConfig, _PERSIST_EVERY_N_UPDATES

logger = logging.getLogger(__name__)

_ERROR_STORE_PERCENTILE: float = 99.0
_ERROR_STORE_MIN_SAMPLES: int = 10
_PERCENTILE_SAMPLE_SIZE: int = 100
_MAD_K_SMALL: float = 9.0
_MAD_K_LARGE: float = 6.0
_MAD_EPSILON: float = 1e-12


# ── Per-sensor key helpers ──────────────────────────────────────────


def build_regime_key(domain: str, regime: str, series_id: Optional[str]) -> str:
    if series_id and series_id != "unknown":
        return f"{domain}:{regime}:{series_id}"
    return f"{domain}:{regime}"


def build_fallback_key(domain: str, regime: str) -> str:
    return f"{domain}:{regime}"


def should_use_per_sensor(
    series_id: Optional[str],
    accuracy: Dict[str, dict],
    domain: str,
    regime: str,
) -> bool:
    if not series_id or series_id == "unknown":
        return False
    key = build_regime_key(domain, regime, series_id)
    return key in accuracy


def compute_weights_from_accuracy(
    engine_names: List[str],
    regime_data: Dict[str, float],
    min_weight: float,
) -> Dict[str, float]:
    n = len(engine_names)
    if n == 0:
        return {}
    accs = []
    for name in engine_names:
        a = regime_data.get(name, min_weight)
        a = max(min_weight, min(1.0, a))
        accs.append(a)
    total = sum(accs)
    if total < 1e-12:
        return {name: 1.0 / n for name in engine_names}
    raw = {name: acc / total for name, acc in zip(engine_names, accs)}
    clamped = {name: max(min_weight, v) for name, v in raw.items()}
    c_total = sum(clamped.values())
    if c_total < 1e-12:
        return {name: 1.0 / n for name in engine_names}
    return {name: v / c_total for name, v in clamped.items()}


# ── Gaussian Prior ──────────────────────────────────────────────────


@dataclass
class GaussianPrior:
    mu_0: float = 0.0
    sigma2_0: float = 1.0

    def to_dict(self) -> dict:
        return {"mu_0": self.mu_0, "sigma2_0": self.sigma2_0}

    @classmethod
    def from_dict(cls, d: dict) -> GaussianPrior:
        return cls(mu_0=d.get("mu_0", 0.0), sigma2_0=d.get("sigma2_0", 1.0))


# ── Bayesian Updater ────────────────────────────────────────────────


class BayesianUpdater:
    """Gaussian Conjugate-Bayesian update."""

    def update(
        self,
        prior: GaussianPrior,
        observations: np.ndarray,
        sigma2_obs: float = 1.0,
    ) -> GaussianPrior:
        n = len(observations)
        if n == 0:
            return prior
        obs_mean = float(np.mean(observations))
        if sigma2_obs <= 0:
            sigma2_obs = 1.0
        post_var = 1.0 / (1.0 / prior.sigma2_0 + n / sigma2_obs)
        post_mean = post_var * (prior.mu_0 / prior.sigma2_0 + n * obs_mean / sigma2_obs)
        return GaussianPrior(mu_0=post_mean, sigma2_0=post_var)


# ── Accuracy computation ────────────────────────────────────────────


def _compute_robust_cap(history: List[float]) -> float:
    if not history:
        return float("inf")
    arr = np.asarray(history, dtype=np.float64)
    n = len(arr)
    median = float(np.median(arr))
    abs_dev = np.abs(arr - median)
    mad = float(np.median(abs_dev))
    k = _MAD_K_SMALL if n < 50 else _MAD_K_LARGE
    if mad < _MAD_EPSILON:
        mean_ad = float(np.mean(abs_dev))
        if mean_ad < _MAD_EPSILON:
            return median * 2.0 if median > 0 else float("inf")
        return median + k * mean_ad
    return median + k * mad


def compute_accuracy(
    prediction_error: float,
    error_store: Optional[object] = None,
    series_id: Optional[str] = None,
    engine_name: Optional[str] = None,
    error_history: Optional[Dict[str, List[float]]] = None,
) -> float:
    abs_error = float(abs(prediction_error))
    if not math.isfinite(abs_error):
        abs_error = float("inf")

    history: Optional[List[float]] = None
    if error_store is not None and series_id and engine_name:
        try:
            recent = error_store.get_recent(series_id, engine_name, _PERCENTILE_SAMPLE_SIZE)
            if len(recent) >= _ERROR_STORE_MIN_SAMPLES:
                history = recent
        except Exception:
            pass

    if history is None and error_history is not None:
        hist = error_history.get("__all__", [])
        if len(hist) >= _ERROR_STORE_MIN_SAMPLES:
            history = hist

    if history:
        cap = _compute_robust_cap(history)
        if 0.0 < cap < float("inf"):
            abs_error = min(abs_error, cap)

    accuracy = 1.0 / (1.0 + abs_error)
    return float(max(0.0, min(1.0, accuracy)))


# ── L2 Regularization ──────────────────────────────────────────────


def compute_regularization_strength(
    update_counter: int,
    base_strength: float = 0.01,
    drift_score: float = 0.0,
) -> float:
    strength = base_strength
    if drift_score > 0:
        strength += drift_score * 0.05
    return min(strength, 1.0)


def apply_l2_regularization(
    accuracies: Dict[str, float],
    engine_names: List[str],
    regularization_strength: float = 0.01,
) -> Dict[str, float]:
    if not engine_names:
        return {}
    n = len(engine_names)
    uniform = 1.0 / n
    result = {}
    for name in engine_names:
        acc = accuracies.get(name, uniform)
        result[name] = acc - regularization_strength * (acc - uniform)
    return result


# ── Variance Estimator ──────────────────────────────────────────────


class VarianceEstimator:
    """Online per-engine observation variance estimation."""

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
        if engine_name not in self._deques:
            self._deques[engine_name] = deque(maxlen=self._window)
        self._deques[engine_name].append(float(error))

    def get_sigma2_obs(self, engine_name: str) -> float:
        dq = self._deques.get(engine_name)
        n = len(dq) if dq else 0
        if n < self._min_samples:
            return self._default_sigma2
        var = float(np.var(list(dq), ddof=0))
        return max(self._min_sigma2, var)
