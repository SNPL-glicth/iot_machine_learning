"""FASE 10.5 — Train/Val/Test Split (Temporal, No Shuffle)."""

from __future__ import annotations

from typing import Tuple, List

from .context_calibrator import ContextKey


def train_val_test_split(
    data: List[Tuple[ContextKey, float, bool]],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Tuple[List, List, List]:
    """Divide datos en train/val/test temporalmente (sin shuffle)."""
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Ratios deben sumar 1.0")
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return data[:train_end], data[train_end:val_end], data[val_end:]