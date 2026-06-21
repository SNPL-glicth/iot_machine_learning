"""Narrative embedding network — 18-dim situation → 8-dim narrative embedding.

Uses existing FeedforwardLayer (numpy, no PyTorch).
Architecture per plan:
    Dense(18 → 32, ReLU) → Dense(32 → 16, ReLU) → Dense(16 → 8, ReLU)

Weights initialized Xavier/Glorot (deterministic seed for reproducibility).
No training yet — offline learning on historical logs is future work.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from iot_machine_learning.infrastructure.ml.cognitive.narrative.layers import FeedforwardLayer


class NarrativeEmbeddingNetwork:
    """Situation-aware narrative embedding generator.

    Input: 18-dim normalized situation vector.
    Output: 8-dim non-negative embedding (ReLU).
    """

    def __init__(self, seed: int = 42) -> None:
        self.layer1 = FeedforwardLayer(18, 32, activation="relu", seed=seed)
        self.layer2 = FeedforwardLayer(32, 16, activation="relu", seed=seed + 1)
        self.layer3 = FeedforwardLayer(16, 8, activation="relu", seed=seed + 2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            x: situation vector, shape (18,) or (batch, 18).
               Values expected in [0, 1] (pre-normalized).

        Returns:
            8-dim embedding, all values >= 0 (ReLU).
        """
        h1 = self.layer1.forward(x)
        h2 = self.layer2.forward(h1)
        out = self.layer3.forward(h2)
        return out

    def embed(self, situation_vector: list[float]) -> list[float]:
        """Convenience wrapper: list → numpy → forward → list."""
        arr = np.array(situation_vector, dtype=np.float64)
        if arr.shape == (18,):
            out = self.forward(arr)
        elif arr.ndim == 2 and arr.shape[1] == 18:
            out = self.forward(arr)
        else:
            raise ValueError(f"Expected shape (18,) or (batch, 18), got {arr.shape}")
        return [round(float(v), 6) for v in out]

    def get_weights(self) -> dict[str, tuple]:
        """Serialize current weights for checkpointing / offline training."""
        return {
            "layer1": self.layer1.get_weights(),
            "layer2": self.layer2.get_weights(),
            "layer3": self.layer3.get_weights(),
        }

    def set_weights(self, weights: dict[str, tuple]) -> None:
        """Load weights from checkpoint."""
        self.layer1.set_weights(*weights["layer1"])
        self.layer2.set_weights(*weights["layer2"])
        self.layer3.set_weights(*weights["layer3"])
