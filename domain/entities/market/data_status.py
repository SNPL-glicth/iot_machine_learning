"""Estado de calidad/delay de una observación de mercado.

Contract v1 — cada ``MarketObservation`` lleva su ``DataStatus`` para que
el evaluador jamás mezcle señales live con datos retardados o de replay.
"""

from __future__ import annotations

from enum import Enum


class DataStatus(Enum):
    """Estado de la observación respecto a su frescura y origen."""

    REALTIME = "realtime"
    DELAYED = "delayed"
    EOD = "eod"
    REPLAY = "replay"
    STALE = "stale"
    UNVERIFIED = "unverified"

    @property
    def is_live_signal(self) -> bool:
        """``True`` si la observación puede considerarse señal live."""
        return self is DataStatus.REALTIME
