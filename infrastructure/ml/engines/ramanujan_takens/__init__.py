"""Ramanujan Takens Topological Inference Engine Package.

Exposes:
- TakensRingBuffer: Zero-allocation circular ring buffer.
- RamanujanTakensEngine: PredictionEngine reconstructing latent phase space.
"""

from __future__ import annotations

from .circular_buffer import TakensRingBuffer
from .engine import RamanujanTakensEngine

__all__ = [
    "TakensRingBuffer",
    "RamanujanTakensEngine",
]
