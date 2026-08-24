"""Market Replay Engine Config (FASE 5 → FASE 6)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from ..prediction.reward import RewardConfig
from .clock import Clock
from .feed import HistoricalFeed
from .baselines import Predictor

__all__ = ["ReplayEngineConfig"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayEngineConfig:
    """Configuración de un run del Market Replay."""

    symbol: str
    feed: HistoricalFeed
    interval_seconds: int
    horizons_seconds: tuple[int, ...]
    feature_window_size: int = 60
    predictor_lookback: int = 20
    strategy: str = "baseline"
    predictor: Predictor | None = None
    latency_sample_every: int | None = None
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    initial_clock: Clock | None = None  # FASE 6: permite LiveClock para live shadow
    # FASE 6: ventanas degradadas [start, end] del proveedor. Solo las
    # usa el modo shadow: predicciones emitidas en contexto incompleto
    # se invalidan al emitir (reason=provider_gap), jamás producen reward.
    # Vacío por defecto → el replay clásico es byte-por-byte idéntico.
    degraded_windows: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol no puede ser vacío")
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds debe ser > 0")
        if not self.horizons_seconds:
            raise ValueError("horizons_seconds no puede estar vacío")
        for horizon in self.horizons_seconds:
            if horizon <= 0:
                raise ValueError(f"horizonte inválido: {horizon!r}")
        if self.feature_window_size < self.predictor_lookback + 1:
            raise ValueError(
                "feature_window_size debe cubrir predictor_lookback + 1 velas"
            )
        for start, end in self.degraded_windows:
            if start < 0 or end < start:
                raise ValueError(
                    f"ventana degradada inválida: ({start!r}, {end!r})"
                )