"""Feed helpers compartidos por los runners live (FASE 6/7).

``FragmentFeed`` y ``DropWindowsFeed`` son envoltorios del contrato
``HistoricalFeed`` que recortan la secuencia a un fragmento de sesión y
simulan pérdida de datos (caídas). El fragmento y las caídas se expresan
en hora de mercado (09:30 = apertura de la sesión) y se anclan al primer
evento del dataset, sin importar su zona horaria.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from iot_machine_learning.domain.entities.market.replay import HistoricalFeed

__all__ = [
    "RESOLUTIONS",
    "FragmentFeed",
    "DropWindowsFeed",
    "fragment_bounds",
    "parse_drop",
    "drop_windows",
    "fmt_ts",
]

RESOLUTIONS: dict[str, tuple[int, tuple[int, ...]]] = {
    "1m": (60, (60, 300, 900, 3600)),
    "5m": (300, (300, 900, 3600)),
    "1h": (3600, (3600,)),
    "1d": (86400, (86400,)),
}

_SESSION_OPEN_SECS = 9 * 3600 + 30 * 60  # 09:30, apertura de la sesión


class FragmentFeed:
    """Feed histórico recortado a [start, end) (contrato HistoricalFeed)."""

    def __init__(self, inner: HistoricalFeed, start: float, end: float) -> None:
        self.symbol = inner.symbol
        self.resolution_seconds = inner.resolution_seconds
        self._candles = tuple(
            c for c in inner.iter_events() if start <= c.timestamp < end
        )

    def iter_events(self) -> Iterable:
        yield from self._candles

    def __len__(self) -> int:
        return len(self._candles)


class DropWindowsFeed:
    """Feed que omite eventos dentro de ventanas de caída (simula pérdida)."""

    def __init__(self, inner: FragmentFeed, drops: list[tuple[float, float]]) -> None:
        self.symbol = inner.symbol
        self.resolution_seconds = inner.resolution_seconds
        self._candles = tuple(
            c
            for c in inner.iter_events()
            if not any(start <= c.timestamp <= end for start, end in drops)
        )
        self.drops = drops

    def iter_events(self) -> Iterable:
        yield from self._candles

    def __len__(self) -> int:
        return len(self._candles)


def fmt_ts(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(ts))


def fragment_bounds(first_ts: float, start_hm: str, end_hm: str) -> tuple[float, float]:
    """Ancla el fragmento a la apertura de la primera sesión del dataset.

    ``09:30 -> 10:30`` (default) significa "la primera hora de la sesión",
    sin importar la zona horaria del dataset (UTC en los CSVs): el inicio
    es el primer evento y la duración es la del intervalo pedido.
    """
    sh, sm = (int(part) for part in start_hm.split(":"))
    eh, em = (int(part) for part in end_hm.split(":"))
    duration = eh * 3600 + em * 60 - (sh * 3600 + sm * 60)
    if duration <= 0:
        raise ValueError(f"fragmento inválido: {start_hm} -> {end_hm}")
    return first_ts, first_ts + duration


def parse_drop(value: str) -> tuple[float, float]:
    """``HH:MM:SS-HH:MM:SS`` en hora de mercado -> (start, end) en segundos
    desde la apertura de la sesión."""
    start_raw, end_raw = value.split("-")
    sh, sm, ss = (int(part) for part in start_raw.split(":"))
    eh, em, es = (int(part) for part in end_raw.split(":"))
    return float(sh * 3600 + sm * 60 + ss), float(eh * 3600 + em * 60 + es)


def drop_windows(
    start_ts: float, drops_raw: list[str]
) -> list[tuple[float, float]]:
    """Convierte caídas en hora de mercado a timestamps absolutos del
    fragmento (09:30 = apertura de la sesión = ``start_ts``)."""
    drops: list[tuple[float, float]] = []
    for value in drops_raw:
        drop_start, drop_end = parse_drop(value)
        drops.append(
            (
                start_ts + max(drop_start - _SESSION_OPEN_SECS, 0.0),
                start_ts + max(drop_end - _SESSION_OPEN_SECS, 0.0),
            )
        )
    return drops
