"""Data models and connection states for Alpaca WebSocket Feed."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeedStats:
    """Estadísticas del feed para monitoreo."""
    events_received: int = 0
    events_emitted: int = 0
    trades_received: int = 0
    quotes_received: int = 0
    quotes_stale_dropped: int = 0
    bars_received: int = 0
    reconnects: int = 0
    last_event_time: float = 0.0
    last_emitted_time: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "events_received": self.events_received,
            "events_emitted": self.events_emitted,
            "trades_received": self.trades_received,
            "quotes_received": self.quotes_received,
            "quotes_stale_dropped": self.quotes_stale_dropped,
            "bars_received": self.bars_received,
            "reconnects": self.reconnects,
            "avg_latency_ms": self.avg_latency_ms,
            "max_latency_ms": self.max_latency_ms,
        }


class ConnectionState:
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
