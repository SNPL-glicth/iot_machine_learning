"""Transformaciones canónicas para la auditoría de pérdida de representación de ZENIN.

Implementa las representaciones base:
- R0: Raw (señal original sin transformar)
- R1: Residual (x_t - rolling_median(x_{t-w:t}))
- R2: Downsample 2x (decimación / block pooling con mapeo temporal)
- R3: Downsample 5x
- R4: Downsample 10x
- R5: Envelope (amplitud pico a pico y extremos por ventana)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TransformedSignal:
    """Contenedor de una señal transformada preservando trazabilidad temporal."""

    name: str
    values: np.ndarray
    timestamps_sec: np.ndarray
    timestamps_raw: list[str]
    orig_indices: np.ndarray  # Índice en la serie raw original
    compression_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.values)


def to_raw(
    values: np.ndarray,
    timestamps_sec: np.ndarray,
    timestamps_raw: list[str],
) -> TransformedSignal:
    """R0: Señal cruda original como referencia absoluta."""
    n = len(values)
    return TransformedSignal(
        name="R0_raw",
        values=values.copy(),
        timestamps_sec=timestamps_sec.copy(),
        timestamps_raw=list(timestamps_raw),
        orig_indices=np.arange(n),
        compression_ratio=1.0,
        metadata={"description": "Raw baseline signal, no compression or transformation"},
    )


def to_residual(
    values: np.ndarray,
    timestamps_sec: np.ndarray,
    timestamps_raw: list[str],
    window: int = 50,
) -> TransformedSignal:
    """R1: Residuo causal respecto a la mediana móvil local.

    r_t = x_t - median(x_{max(0, t-w+1):t+1})
    """
    n = len(values)
    residuals = np.zeros(n, dtype=np.float64)

    # Cálculo causal en streaming
    for i in range(n):
        start_idx = max(0, i - window + 1)
        local_window = values[start_idx : i + 1]
        med = float(np.median(local_window))
        residuals[i] = values[i] - med

    return TransformedSignal(
        name="R1_residual",
        values=residuals,
        timestamps_sec=timestamps_sec.copy(),
        timestamps_raw=list(timestamps_raw),
        orig_indices=np.arange(n),
        compression_ratio=1.0,
        metadata={
            "description": f"Causal rolling median residual (window={window})",
            "window": window,
        },
    )


def to_downsample(
    values: np.ndarray,
    timestamps_sec: np.ndarray,
    timestamps_raw: list[str],
    factor: int = 2,
    mode: str = "decimate",  # "decimate" o "mean"
) -> TransformedSignal:
    """R2/R3/R4: Reducción de frecuencia temporal por factor k.

    - decimate: toma cada k-ésimo punto (x[0], x[k], x[2k], ...)
    - mean: promedio por bloques de k puntos
    """
    n = len(values)
    name = f"R_downsample_{factor}x_{mode}"

    if mode == "decimate":
        indices = np.arange(0, n, factor)
        ds_values = values[indices]
        ds_ts_sec = timestamps_sec[indices]
        ds_ts_raw = [timestamps_raw[i] for i in indices]
    elif mode == "mean":
        n_blocks = n // factor
        indices = np.arange(0, n_blocks * factor, factor) + (factor - 1)  # timestamp al final del bloque
        reshaped = values[: n_blocks * factor].reshape(n_blocks, factor)
        ds_values = np.mean(reshaped, axis=1)
        ds_ts_sec = timestamps_sec[indices]
        ds_ts_raw = [timestamps_raw[i] for i in indices]
    else:
        raise ValueError(f"Modo desconocido: {mode}")

    return TransformedSignal(
        name=name,
        values=ds_values,
        timestamps_sec=ds_ts_sec,
        timestamps_raw=ds_ts_raw,
        orig_indices=indices,
        compression_ratio=float(n) / float(len(ds_values)),
        metadata={
            "description": f"Downsample {factor}x ({mode})",
            "factor": factor,
            "mode": mode,
        },
    )


def to_envelope(
    values: np.ndarray,
    timestamps_sec: np.ndarray,
    timestamps_raw: list[str],
    window: int = 10,
    metric: str = "spread",  # "spread" = max - min; "extreme_dev" = max(|x - mean|)
) -> TransformedSignal:
    """R5: Boceto de envolvente multiescala por bloques.

    E_w = (max(x) - min(x)) sobre bloques contiguos de tamaño window.
    Preserva el rango dinámico y la energía de choque local.
    """
    n = len(values)
    n_blocks = n // window
    indices = np.arange(0, n_blocks * window, window) + (window - 1)

    reshaped = values[: n_blocks * window].reshape(n_blocks, window)

    if metric == "spread":
        block_max = np.max(reshaped, axis=1)
        block_min = np.min(reshaped, axis=1)
        env_values = block_max - block_min
    elif metric == "extreme_dev":
        block_mean = np.mean(reshaped, axis=1, keepdims=True)
        block_dev = np.max(np.abs(reshaped - block_mean), axis=1)
        env_values = block_dev
    else:
        raise ValueError(f"Métrica desconocida: {metric}")

    return TransformedSignal(
        name=f"R5_envelope_{metric}_w{window}",
        values=env_values,
        timestamps_sec=timestamps_sec[indices],
        timestamps_raw=[timestamps_raw[i] for i in indices],
        orig_indices=indices,
        compression_ratio=float(n) / float(len(env_values)),
        metadata={
            "description": f"Envelope block sketch ({metric}, window={window})",
            "window": window,
            "metric": metric,
        },
    )
