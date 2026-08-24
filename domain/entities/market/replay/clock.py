"""Reloj del Market Replay (FASE 5 → FASE 6).

Un único invariante gobierna el replay: el reloj nunca retrocede.
``ReplayClock`` es un valor inmutable cuyo avance produce una copia:
compartir relojes no habilita retroceder el pasado de otro.

La regla de oro — *ZENIN no puede mirar el futuro* — se apoya en este
tipo: nada que tenga ``timestamp > clock.now`` es observable; cualquier
intento de retroceder el reloj es un error, no un estado.

FASE 6: Protocol Clock — abstracción para que el engine no sepa si está
en replay (tiempo lógico) o live (tiempo real). ``LiveClock`` usa el mismo
contrato que ``ReplayClock`` pero delega a ``time.time()``.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


class ClockRollbackError(ValueError):
    """Se intentó retroceder el reloj del replay."""


class Clock(Protocol):
    """Protocol de reloj: abstracción para replay vs live (FASE 6)."""

    now: float

    def advance_to(self, timestamp: float) -> Clock:
        """Avanza el reloj a ``timestamp`` (monótono; copia inmutable)."""
        ...


@dataclass(frozen=True, slots=True)
class ReplayClock:
    """Reloj lógico del replay (epoch float, segundos)."""

    now: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.now):
            raise ValueError(f"now debe ser finito: {self.now!r}")

    def advance_to(self, timestamp: float) -> ReplayClock:
        """Avanza el reloj a ``timestamp`` (monótono; copia inmutable)."""
        if timestamp < self.now:
            raise ClockRollbackError(
                f"el reloj no puede retroceder: {self.now!r} -> {timestamp!r}"
            )
        return ReplayClock(now=timestamp)


@dataclass(frozen=True, slots=True)
class LiveClock:
    """Reloj en vivo (epoch float, segundos). Usa ``time.time()`` (FASE 6)."""

    now: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.now):
            raise ValueError(f"now debe ser finito: {self.now!r}")

    @classmethod
    def current(cls) -> LiveClock:
        """Crea un LiveClock en el instante actual."""
        return cls(now=time.time())

    def advance_to(self, timestamp: float) -> LiveClock:
        """Avanza el reloj a ``timestamp`` (monótono; copia inmutable)."""
        if timestamp < self.now:
            raise ClockRollbackError(
                f"el reloj no puede retroceder: {self.now!r} -> {timestamp!r}"
            )
        return LiveClock(now=timestamp)
