"""LiveFeed wrapper (FASE 6).

Implementa el mismo contrato que HistoricalFeed pero para datos en vivo.
Detecta gaps en timestamps y mantiene estado de conexión.

Condición 1: Mismo objeto MarketObservation que HistoricalFeed.
Condición 3: GAP_DETECTED visible cuando faltan timestamps.
Condición 4: Estados CONNECTED/DEGRADED/DISCONNECTED/RECONNECTING/RECOVERED
             con historial de transiciones visible.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

from iot_machine_learning.domain.entities.market import (
    ConnectionState,
    MarketObservation,
)
from iot_machine_learning.domain.entities.market.replay import HistoricalFeed


@dataclass(frozen=True, slots=True)
class GapDetected:
    """Gap detectado en el feed live (condición 3)."""

    expected_timestamp: float
    received_timestamp: float
    gap_seconds: float


@dataclass(frozen=True, slots=True)
class StateTransition:
    """Transición de estado de conexión registrada (condición 4)."""

    state: ConnectionState
    at_timestamp: float | None


class LiveFeed:
    """Feed live con detección de gaps y estados de conexión (FASE 6).

    Por restricción "sin red en runtime", este LiveFeed usa HistoricalFeed
    como fuente de datos pero simula el comportamiento live: detecta gaps
    y mantiene estados de conexión.

    Condición 1: Mismo contrato que HistoricalFeed → MarketObservation.
    Condición 3: GAP_DETECTED visible cuando expected != received.
    Condición 4: Estados CONNECTED/DEGRADED/DISCONNECTED/RECONNECTING/RECOVERED.
    """

    def __init__(
        self,
        symbol: str,
        historical_feed: HistoricalFeed,
        expected_interval_seconds: int,
        gap_threshold_seconds: float = 1.5,
    ) -> None:
        """Inicializa LiveFeed sobre datos históricos simulando live.

        Args:
            symbol: Símbolo a observar
            historical_feed: Feed de datos congelados (fuente sin red)
            expected_interval_seconds: Intervalo esperado entre eventos (ej. 60s para 1m)
            gap_threshold_seconds: Umbral para detectar gap (1.5x intervalo esperado)
        """
        self.symbol = symbol
        self._historical = historical_feed
        self._expected_interval = expected_interval_seconds
        self._gap_threshold = gap_threshold_seconds
        self._state = ConnectionState.CONNECTED
        self._last_timestamp: float | None = None
        self._gaps: list[GapDetected] = []
        self._transitions: list[StateTransition] = []
        self._was_degraded = False

    @property
    def state(self) -> ConnectionState:
        """Estado actual de conexión."""
        return self._state

    @property
    def source(self) -> HistoricalFeed:
        """Feed subyacente (para pre-scan del modo shadow)."""
        return self._historical

    @property
    def gap_threshold(self) -> float:
        """Umbral de gap configurado (multiplicador del intervalo)."""
        return self._gap_threshold

    @property
    def gaps(self) -> tuple[GapDetected, ...]:
        """Gaps detectados (inmutable)."""
        return tuple(self._gaps)

    @property
    def transitions(self) -> tuple[StateTransition, ...]:
        """Historial de transiciones de estado (inmutable)."""
        return tuple(self._transitions)

    def iter_events(self) -> Generator[MarketObservation, None, None]:
        """Itera eventos con detección de gaps (mismo contrato que HistoricalFeed).

        Un gap degrada el feed (DEGRADED); el primer evento sano posterior
        marca RECOVERED y la sesión vuelve a CONNECTED. La regla de oro se
        conserva: nada se descarta, solo se registra y se expone.
        """
        for event in self._historical.iter_events():
            if not isinstance(event, MarketObservation):
                raise TypeError(
                    f"feed entregó {type(event).__name__}, "
                    "se espera MarketObservation"
                )
            if event.symbol != self.symbol:
                raise ValueError(
                    f"feed fuera de símbolo: esperaba {self.symbol}, "
                    f"obtuvo {event.symbol!r}"
                )

            if self._last_timestamp is not None:
                gap = event.timestamp - self._last_timestamp
                if gap > self._gap_threshold * self._expected_interval:
                    gap_event = GapDetected(
                        expected_timestamp=self._last_timestamp + self._expected_interval,
                        received_timestamp=event.timestamp,
                        gap_seconds=gap,
                    )
                    self._gaps.append(gap_event)
                    self._set_state(ConnectionState.DEGRADED, event.timestamp)
                elif self._was_degraded:
                    self._set_state(ConnectionState.RECOVERED, event.timestamp)
                    self._set_state(ConnectionState.CONNECTED, event.timestamp)

            self._was_degraded = self._state is ConnectionState.DEGRADED
            self._last_timestamp = event.timestamp
            yield event

    def update_state(self, new_state: ConnectionState, at_timestamp: float | None = None) -> None:
        """Actualiza estado de conexión manualmente (condición 4)."""
        if not isinstance(new_state, ConnectionState):
            raise TypeError(f"estado inválido: {new_state!r}")
        self._set_state(new_state, at_timestamp if at_timestamp is not None else self._last_timestamp)

    def _set_state(self, new_state: ConnectionState, at_timestamp: float | None) -> None:
        if new_state is self._state:
            return
        self._state = new_state
        self._transitions.append(
            StateTransition(state=new_state, at_timestamp=at_timestamp)
        )
