"""NeuralExpert — tiny feedforward neural network as an MoE expert.

Numpy-only, 1 hidden layer (6 neurons), Xavier init, manual backprop.
Trains per-series on first predict() call, persists weights to disk.
No PyTorch/TensorFlow dependencies.

Designed to match the ExpertPort interface exactly, same as
BaselineExpert, StatisticalExpert, and TaylorExpert.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np

from iot_machine_learning.domain.ports.expert_port import ExpertPort, ExpertOutput, ExpertCapability
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow

_WEIGHTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "infrastructure", "ml", "weights", "neural_expert_v1")
)
_METADATA_PATH = os.path.join(_WEIGHTS_ROOT, "metadata.json")

_INPUT_SIZE = 10
_HIDDEN_SIZE = 6
_LR = 0.001  # lower LR to prevent divergence
_EPOCHS = 300  # more epochs since lower LR
_MIN_TRAIN_SAMPLES = 5
_L2_LAMBDA = 1e-4  # L2 regularization strength
_GRAD_CLIP = 1.0  # max gradient norm


def _xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float64)


def _clip_grads(grads: list[np.ndarray], max_norm: float) -> list[np.ndarray]:
    total_norm = sum(np.sum(g ** 2) for g in grads) ** 0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        return [g * scale for g in grads]
    return grads


def _check_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        arr[:] = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)


class TinyNN:
    """Minimal 2-layer feedforward network: input → hidden(ReLU) → output(linear)."""

    __slots__ = ("W1", "b1", "W2", "b2", "rng")

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.W1 = _xavier_init(input_size, hidden_size, self.rng)
        self.b1 = np.zeros((1, hidden_size), dtype=np.float64)
        self.W2 = _xavier_init(hidden_size, 1, self.rng)
        self.b2 = np.zeros((1, 1), dtype=np.float64)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        _check_finite(z1, "z1")
        a1 = np.maximum(0.0, z1)
        a1 = np.minimum(a1, 1e6)
        z2 = a1 @ self.W2 + self.b2
        _check_finite(z2, "z2")
        return z2, a1

    def predict(self, X: np.ndarray) -> np.ndarray:
        out, _ = self.forward(X)
        return out

    def train(self, X: np.ndarray, y: np.ndarray, lr: float = _LR, epochs: int = _EPOCHS) -> list[float]:
        losses = []
        best_loss = float("inf")
        best_params = None
        patience = 30
        stall = 0

        for epoch in range(epochs):
            pred, a1 = self.forward(X)
            error = pred - y.reshape(-1, 1)
            loss = float(np.mean(error ** 2))
            losses.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_params = {k: v.copy() for k, v in self.get_params().items()}
                stall = 0
            else:
                stall += 1

            if np.isnan(loss) or np.isinf(loss):
                if best_params is not None:
                    self.set_params(best_params)
                break
            if stall >= patience:
                break

            dW2 = a1.T @ error + _L2_LAMBDA * self.W2
            db2 = np.sum(error, axis=0, keepdims=True)
            d_a1 = error @ self.W2.T
            d_z1 = d_a1 * (a1 > 0).astype(np.float64)
            dW1 = X.T @ d_z1 + _L2_LAMBDA * self.W1
            db1 = np.sum(d_z1, axis=0, keepdims=True)

            grads = _clip_grads([dW1, db1, dW2, db2], _GRAD_CLIP)
            dW1, db1, dW2, db2 = grads

            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

        if best_params is not None:
            self.set_params(best_params)

        return losses

    def get_params(self) -> dict:
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def set_params(self, params: dict) -> None:
        self.W1 = params["W1"]
        self.b1 = params["b1"]
        self.W2 = params["W2"]
        self.b2 = params["b2"]


def _build_training_data(values: list[float], input_size: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
    if len(values) < input_size + 2:
        return None, None, 0.0, 1.0
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr)) or 1.0
    normalized = (arr - mean) / std
    n = len(normalized)
    X_list, y_list = [], []
    for i in range(n - input_size):
        X_list.append(normalized[i: i + input_size])
        y_list.append(normalized[i + input_size])
    return np.array(X_list), np.array(y_list), mean, std


class NeuralExpert(ExpertPort):
    """Neural network expert — tiny MLP trained per-series on the fly.

    Capabilities:
    - regimes: all (stable, trending, volatile, noisy) — adaptable
    - computational_cost: 3.0 (higher than statistical but lower than
      anything requiring GPU)
    - min_points: _INPUT_SIZE + 2 (needs enough data for sub-windows)
    - specialties: ("pattern_learning", "non_linear_approximation")
    """

    def __init__(
        self,
        input_size: int = _INPUT_SIZE,
        hidden_size: int = _HIDDEN_SIZE,
        lr: float = _LR,
        epochs: int = _EPOCHS,
        weights_dir: str = _WEIGHTS_ROOT,
    ) -> None:
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._lr = lr
        self._epochs = epochs
        self._weights_dir = weights_dir
        os.makedirs(self._weights_dir, exist_ok=True)

        self._net: Optional[TinyNN] = None
        self._norm_mean: float = 0.0
        self._norm_std: float = 1.0
        self._last_predictions: list[float] = []

        self._capabilities = ExpertCapability(
            regimes=("stable", "trending", "volatile", "noisy"),
            domains=("iot",),
            min_points=self._input_size + 2,
            max_points=0,
            specialties=("pattern_learning", "non_linear_approximation"),
            computational_cost=5.0,
        )

    @property
    def name(self) -> str:
        return "neural"

    @property
    def capabilities(self) -> ExpertCapability:
        return self._capabilities

    def _weights_path(self, series_id: str) -> str:
        return os.path.join(self._weights_dir, f"{series_id}_weights.npz")

    def _load_weights(self, series_id: str) -> bool:
        path = self._weights_path(series_id)
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path)
            self._net = TinyNN(self._input_size, self._hidden_size)
            self._net.set_params({
                "W1": data["W1"],
                "b1": data["b1"].reshape(1, -1),
                "W2": data["W2"],
                "b2": data["b2"].reshape(1, 1),
            })
            self._norm_mean = float(data["norm_mean"].item())
            self._norm_std = float(data["norm_std"].item())
            return True
        except Exception:
            return False

    def _save_weights(self, series_id: str) -> None:
        if self._net is None:
            return
        params = self._net.get_params()
        path = self._weights_path(series_id)
        np.savez_compressed(
            path,
            W1=params["W1"],
            b1=params["b1"],
            W2=params["W2"],
            b2=params["b2"],
            norm_mean=np.array([self._norm_mean]),
            norm_std=np.array([self._norm_std]),
        )

    def _update_metadata(self, series_id: str, n_points: int, final_loss: float) -> None:
        meta: dict = {}
        if os.path.exists(_METADATA_PATH):
            try:
                with open(_METADATA_PATH) as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        meta[series_id] = {
            "trained_at": time.time(),
            "n_points": n_points,
            "final_loss": round(final_loss, 6),
            "input_size": self._input_size,
            "hidden_size": self._hidden_size,
            "epochs": self._epochs,
            "lr": self._lr,
        }
        with open(_METADATA_PATH, "w") as f:
            json.dump(meta, f, indent=2)

    def can_handle(self, window: SensorWindow) -> bool:
        return len(window.readings) >= self._input_size + 2

    def estimate_latency_ms(self, n_points: int) -> float:
        return 5.0 + (n_points * 0.1)

    def warmup(self, series_id: str, values: list[float]) -> bool:
        """Pre-train model for a series_id using ALL available values.

        Call once before the sliding-window loop so that predict()
        only does inference (no retraining overhead).
        Returns True if training succeeded.
        """
        if self._load_weights(series_id):
            return True
        X, y, mean, std = _build_training_data(values, self._input_size)
        if X is None or len(X) < _MIN_TRAIN_SAMPLES:
            return False
        self._norm_mean = mean
        self._norm_std = std
        self._net = TinyNN(self._input_size, self._hidden_size)
        losses = self._net.train(X, y, lr=self._lr, epochs=self._epochs)
        self._save_weights(series_id)
        self._update_metadata(series_id, len(values), losses[-1] if losses else 0.0)
        return True

    def predict(self, window: SensorWindow) -> ExpertOutput:
        values = [r.value for r in window.readings]
        series_id = window.series_id or "default"

        if not self._load_weights(series_id):
            return self._fallback_predict(values, "untrained_series")

        if self._net is None:
            return self._fallback_predict(values, "no_model")

        if len(values) < self._input_size:
            return self._fallback_predict(values, "window_too_small")

        input_vec = (np.array(values[-self._input_size:], dtype=np.float64) - self._norm_mean) / self._norm_std
        input_vec = input_vec.reshape(1, -1)
        pred_normalized = float(self._net.predict(input_vec)[0, 0])
        prediction = pred_normalized * self._norm_std + self._norm_mean

        if not math.isfinite(prediction):
            return self._fallback_predict(values, "nan_prediction")

        self._last_predictions.append(prediction)
        if len(self._last_predictions) > 10:
            self._last_predictions.pop(0)

        confidence = self._compute_confidence(values)
        trend = self._compute_trend(prediction, values[-1] if values else prediction)

        return ExpertOutput(
            prediction=prediction,
            confidence=confidence,
            trend=trend,
            metadata={
                "expert": "neural",
                "input_size": self._input_size,
                "hidden_size": self._hidden_size,
                "series_id": series_id,
            },
        )

    def _compute_confidence(self, values: list[float]) -> float:
        if len(values) < self._input_size + 2:
            return 0.1
        if self._net is None:
            return 0.1
        arr = np.array(values, dtype=np.float64)
        normalized = (arr - self._norm_mean) / self._norm_std
        n = len(normalized)
        errors = []
        for i in range(n - self._input_size):
            x = normalized[i: i + self._input_size].reshape(1, -1)
            y_true = normalized[i + self._input_size]
            y_pred = float(self._net.predict(x)[0, 0])
            errors.append(abs(y_true - y_pred))
        if not errors:
            return 0.1
        recent_errors = errors[-3:]
        avg_error = float(np.mean(recent_errors))
        raw_conf = 1.0 / (1.0 + avg_error)
        return max(0.05, min(0.95, raw_conf))

    def _compute_trend(self, prediction: float, last_value: float) -> str:
        diff = prediction - last_value
        threshold = self._norm_std * 0.05
        if diff > threshold:
            return "up"
        elif diff < -threshold:
            return "down"
        return "stable"

    def _fallback_predict(self, values: list[float], reason: str) -> ExpertOutput:
        last = values[-1] if values else 0.0
        prediction = float(np.mean(values[-3:])) if len(values) >= 3 else last
        return ExpertOutput(
            prediction=prediction,
            confidence=0.1,
            trend="stable",
            metadata={"expert": "neural", "fallback": True, "reason": reason},
        )

    def reset_series(self, series_id: str) -> None:
        path = self._weights_path(series_id)
        if os.path.exists(path):
            os.remove(path)
        self._net = None


def create_neural_expert(
    input_size: int = _INPUT_SIZE,
    hidden_size: int = _HIDDEN_SIZE,
    lr: float = _LR,
    epochs: int = _EPOCHS,
) -> NeuralExpert:
    return NeuralExpert(
        input_size=input_size,
        hidden_size=hidden_size,
        lr=lr,
        epochs=epochs,
    )
