"""Pure mathematical service for Takens delay coordinate embedding in R^m.

Conforms to:
- ISO/IEC 25010: Reliability and Numerical Fault Tolerance.
- Ultra-Low Latency (HFT): Fully vectorized index broadcasting without Python loops.
- Pure Hexagonal Domain Service: Decoupled from infrastructure and storage.
"""

from __future__ import annotations

import numpy as np

from domain.entities.takens.takens_parameters import TakensParameters

_DEFAULT_PARAMS = TakensParameters()


def extract_embedded_vector(
    series_history: np.ndarray | list[float],
    params: TakensParameters | None = None,
) -> np.ndarray:
    """Extract embedded vector y_t = [x_t, x_{t-tau1}, ..., x_{t-taum}]^T in R^m.

    Uses vectorized integer array indexing. If history length is smaller than max
    lag required by tau_strides, pads with initial observation to maintain stability.

    Args:
        series_history: 1D numerical sequence of observations.
        params: Embedding configuration; defaults to standard Takens parameters.

    Returns:
        np.ndarray of shape (m,) containing delayed coordinates.
    """
    cfg = params or _DEFAULT_PARAMS
    arr = np.asarray(series_history, dtype=np.float64).flatten()
    n_samples = arr.size

    # In case of empty or invalid input, return safe zero-filled vector
    if n_samples == 0:
        return np.zeros(cfg.m, dtype=np.float64)

    # Sanitize NaNs/Infs immediately (ISO 25010)
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

    # Stride offsets: stride[0] = 0 (t), stride[1] = tau1, etc.
    strides = np.array([0] + list(cfg.tau_strides[: cfg.m - 1]), dtype=np.int64)
    # Ensure stride length matches dimension m
    if strides.size < cfg.m:
        fill_count = cfg.m - strides.size
        last_stride = strides[-1] if strides.size > 0 else 0
        extra_strides = last_stride + np.arange(1, fill_count + 1, dtype=np.int64)
        strides = np.concatenate([strides, extra_strides])
    elif strides.size > cfg.m:
        strides = strides[: cfg.m]

    # Target indices from the end of history
    target_indices = (n_samples - 1) - strides

    # Bounded index clamping (replaces padding copy for zero-allocation performance)
    valid_indices = np.clip(target_indices, 0, n_samples - 1)
    embedded_vector = arr[valid_indices]

    return embedded_vector


def extract_delay_matrix(
    series_history: np.ndarray | list[float],
    window_size: int,
    params: TakensParameters | None = None,
) -> np.ndarray:
    """Construct trajectory matrix Y in R^(W x m) using 2D broadcasted strides.

    Evaluates W consecutive embedded vectors simultaneously via vectorized slicing,
    completely bypassing Python loops for HFT execution.

    Args:
        series_history: 1D numerical sequence of length N.
        window_size: Number of consecutive embedded vectors W to produce.
        params: Embedding configuration; defaults to standard Takens parameters.

    Returns:
        np.ndarray of shape (window_size, m).
    """
    cfg = params or _DEFAULT_PARAMS
    arr = np.asarray(series_history, dtype=np.float64).flatten()
    n_samples = arr.size
    w = max(1, int(window_size))

    if n_samples == 0:
        return np.zeros((w, cfg.m), dtype=np.float64)

    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

    strides = np.array([0] + list(cfg.tau_strides[: cfg.m - 1]), dtype=np.int64)
    if strides.size < cfg.m:
        fill_count = cfg.m - strides.size
        last_stride = strides[-1] if strides.size > 0 else 0
        extra_strides = last_stride + np.arange(1, fill_count + 1, dtype=np.int64)
        strides = np.concatenate([strides, extra_strides])
    elif strides.size > cfg.m:
        strides = strides[: cfg.m]

    # Vectorized 2D grid generation: shape (W, m)
    end_idx = n_samples - 1
    start_idx = max(0, end_idx - w + 1)
    actual_rows = end_idx - start_idx + 1

    time_offsets = np.arange(start_idx, end_idx + 1, dtype=np.int64)[:, None]
    stride_offsets = strides[None, :]

    index_grid = np.clip(time_offsets - stride_offsets, 0, n_samples - 1)
    matrix = arr[index_grid]

    # If requested window exceeds available rows, prepend with the first row
    if actual_rows < w:
        pad_rows = w - actual_rows
        first_row = matrix[0:1, :]
        padding = np.repeat(first_row, pad_rows, axis=0)
        matrix = np.vstack([padding, matrix])

    return matrix
