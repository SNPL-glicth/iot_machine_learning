"""Live Shadow runner (FASE 6) — ZENIN consumiendo un feed live.

Composición explícita de infraestructura + dominio para el modo shadow:

    LiveFeed ──> MarketReplayEngine ──> LiveShadowResult

El engine es el MISMO del replay (condición 1): el dominio no cambia;
solo cambia la fuente (LiveFeed) y el reloj (LiveClock). Este runner
añade la capa de honestidad live:

    * pre-scan del feed (pasada seca con otro LiveFeed) para descubrir
      los gaps ANTES de correr el engine;
    * ventanas degradadas [expected, received] pasadas al engine: las
      predicciones emitidas sobre contexto incompleto se invalidan al
      emitir con ``reason="provider_gap"`` — jamás producen reward
      (condiciones 3 y 4);
    * historial de estados de conexión expuesto (condición 4).

Sin persistencia (condición 5): el shadow imprime a consola/dashboard;
la persistencia (MySQL → Outcome → Reward) es la siguiente etapa.

El pre-scan requiere un feed subyacente re-iterable (frozen); con un
WebSocket real el descubrimiento de gaps ocurre en streaming (futuro).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from iot_machine_learning.domain.entities.market.prediction import Prediction
from iot_machine_learning.domain.entities.market.replay import (
    MarketReplayEngine,
    ReplayEngineConfig,
    ReplayRunResult,
)
from iot_machine_learning.infrastructure.adapters.market.live_feed import (
    GapDetected,
    LiveFeed,
    StateTransition,
)


@dataclass(frozen=True, slots=True)
class DegradedWindow:
    """Ventana de tiempo sin datos del proveedor (contexto incompleto)."""

    start: float
    end: float

    def contains(self, timestamp: float) -> bool:
        return self.start <= timestamp <= self.end


@dataclass(frozen=True, slots=True)
class LiveShadowResult:
    """Resultado inmutable de un run live shadow."""

    symbol: str
    resolution_seconds: int
    predictions: tuple[Prediction, ...]
    invalidated: tuple[Prediction, ...]
    gaps: tuple[GapDetected, ...]
    transitions: tuple[StateTransition, ...]
    degraded_windows: tuple[DegradedWindow, ...]
    invalidated_by_gap: tuple[Prediction, ...]

    @property
    def all_predictions(self) -> tuple[Prediction, ...]:
        """Todas las predicciones del run, en orden de emisión, sin duplicar.

        ``invalidated`` es un subconjunto de ``predictions`` (el engine
        materializa los estados terminales en la lista principal), así que
        se deduplica por ``prediction_id`` para que la cuenta sea la
        cantidad real de predicciones emitidas.
        """
        seen: set[str] = set()
        merged: list[Prediction] = []
        for pred in self.predictions + self.invalidated:
            if pred.prediction_id not in seen:
                seen.add(pred.prediction_id)
                merged.append(pred)
        return tuple(merged)


class LiveShadowRunner:
    """Corre ZENIN sobre un feed live y marca la honestidad de los gaps."""

    def __init__(self, feed: LiveFeed, config: ReplayEngineConfig) -> None:
        if not isinstance(feed, LiveFeed):
            raise TypeError(f"feed debe ser LiveFeed, obtenido {type(feed).__name__}")
        if not isinstance(config, ReplayEngineConfig):
            raise TypeError(
                f"config debe ser ReplayEngineConfig, obtenido {type(config).__name__}"
            )
        self.feed = feed
        self.config = config

    def run(self) -> LiveShadowResult:
        """Pre-scan de gaps + engine con ventanas degradadas (una pasada)."""
        windows = self._discover_degraded_windows()
        engine_config = dataclasses.replace(
            self.config,
            degraded_windows=tuple((w.start, w.end) for w in windows),
        )
        engine = MarketReplayEngine(engine_config)
        run_result: ReplayRunResult = engine.run()

        invalidated_by_gap = tuple(
            pred
            for pred in run_result.invalidated
            if pred.invalidation_reason == "provider_gap"
        )
        return LiveShadowResult(
            symbol=self.config.symbol,
            resolution_seconds=self.config.interval_seconds,
            predictions=run_result.predictions,
            invalidated=run_result.invalidated,
            gaps=self.feed.gaps,
            transitions=self.feed.transitions,
            degraded_windows=windows,
            invalidated_by_gap=invalidated_by_gap,
        )

    def _discover_degraded_windows(self) -> tuple[DegradedWindow, ...]:
        """Pasada seca: detecta gaps sin consumir el feed del engine.

        El contexto se considera incompleto hasta que la vela recibida
        cierra: la predicción afectada se emite al cierre de esa vela
        (``received + interval``), no al evento recibido. El intervalo
        [expected, received + interval] cubre ambas emisiones.
        """
        dry_feed = LiveFeed(
            symbol=self.feed.symbol,
            historical_feed=self.feed.source,
            expected_interval_seconds=self.config.interval_seconds,
            gap_threshold_seconds=self.feed.gap_threshold,
        )
        for _ in dry_feed.iter_events():
            pass
        return tuple(
            DegradedWindow(
                gap.expected_timestamp,
                gap.received_timestamp + self.config.interval_seconds,
            )
            for gap in dry_feed.gaps
        )
