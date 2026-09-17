"""HistoricalCsvFeed (FASE 5) — feed histórico congelado en disco.

Lee el CSV normalizado por ``scripts/download_market_data.py`` (schema
``ts_open,o,h,l,c,v,ts_close``, epoch float en segundos) y lo entrega
al replay como velas ordenadas. Es infraestructura: el engine solo ve
el contrato ``HistoricalFeed``.

El archivo se valida al cargar (cabecera, número de columnas, orden
no-decreciente); los datos descargados son un artefacto congelado: el
replay jamás toca la red.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from iot_machine_learning.domain.entities.market import (
    Candle,
    DataStatus,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


class HistoricalCsvFeed:
    """Feed de velas cerradas desde un CSV congelado."""

    def __init__(
        self,
        path: str | Path,
        *,
        symbol: str,
        interval_seconds: int,
        data_status: DataStatus = DataStatus.EOD,
    ) -> None:
        self.path = Path(path)
        # Rutas relativas se resuelven contra la raíz del repo
        # (data/market/NVDA_1m.csv, no contra el CWD).
        if not self.path.is_absolute():
            resolved = _REPO_ROOT / self.path
            if resolved.exists():
                self.path = resolved
        self.symbol = symbol
        self.resolution_seconds = interval_seconds
        self._interval_seconds = interval_seconds
        self._data_status = data_status
        self._candles = self._load()

    def _load(self) -> tuple[Candle, ...]:
        if not self.path.exists():
            raise FileNotFoundError(f"dataset histórico no existe: {self.path}")
        candles: list[Candle] = []
        with self.path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            expected = ["ts_open", "o", "h", "l", "c", "v", "ts_close"]
            if header != expected:
                raise ValueError(
                    f"cabecera CSV inválida: {header!r}, esperada {expected!r}"
                )
            for lineno, row in enumerate(reader, start=2):
                if len(row) != 7:
                    raise ValueError(
                        f"{self.path}:{lineno}: se esperaban 7 columnas, "
                        f"obtenidas {len(row)}"
                    )
                ts_open, o, high, low, c, v, ts_close = (float(x) for x in row)
                if ts_close != ts_open + self._interval_seconds:
                    raise ValueError(
                        f"{self.path}:{lineno}: ts_close {ts_close!r} no "
                        f"coincide con ts_open + intervalo "
                        f"({ts_open!r}+{self._interval_seconds})"
                    )
                candle = Candle(
                    symbol=self.symbol,
                    timestamp=ts_open,
                    data_status=self._data_status,
                    source_provider="csv",
                    open=o,
                    high=high,
                    low=low,
                    close=c,
                    volume=v,
                    interval_seconds=self._interval_seconds,
                )
                if candles and candle.timestamp <= candles[-1].timestamp:
                    raise ValueError(
                        f"{self.path}:{lineno}: velas fuera de orden "
                        f"(ts {candle.timestamp!r} repetido o retrocedido)"
                    )
                candles.append(candle)
        return tuple(candles)

    def iter_events(self) -> Iterator[Candle]:
        yield from self._candles

    def __len__(self) -> int:
        return len(self._candles)
