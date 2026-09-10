"""Auditoría de ingesta — tipos puros (FASE 0).

Dominio puro: sin infraestructura, sin red, sin tiempo real.
Todo cálculo es determinista sobre eventos ya materializados.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True, kw_only=True)
class AuditedEvent:
    """Evento ya mapeado a dominio, listo para auditar.

    Attributes:
        timestamp: Exchange/event ts (epoch segundos, sub-segundo).
        arrival_ts: Cuándo lo recibió el adapter (None si replay/CSV).
        seq: Secuencia del proveedor si existe (update_id, trade_id
            numérico, etc.). None si el provider no la da.
        symbol: Símbolo ya normalizado del dominio.
        kind: candle | quote | trade | book.
        status: DataStatus en minúsculas (realtime, delayed, ...).
    """

    timestamp: float
    symbol: str
    kind: str = "candle"
    status: str = "realtime"
    arrival_ts: float | None = None
    seq: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class FeedAnomaly:
    """Anomalía individual detectada (visible, nunca descartada)."""

    kind: str  # duplicate | out_of_order | gap | stale | symbol_mix | seq_gap | ...
    expected: float | None = None
    received: float | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class FeedAuditReport:
    """Reporte inmutable de la auditoría de un tramo de feed."""

    symbol: str
    total: int
    unique: int
    duplicates: int
    out_of_order: int
    missing_intervals: int
    gap_events: tuple[FeedAnomaly, ...] = ()
    anomalies: tuple[FeedAnomaly, ...] = ()
    completeness_pct: float = 100.0
    stale_count: int = 0
    avg_lag_seconds: float | None = None
    max_lag_seconds: float | None = None
    symbols_seen: tuple[str, ...] = ()
    statuses_seen: tuple[str, ...] = ()
    first_ts: float | None = None
    last_ts: float | None = None
    seq_gaps: int = 0
    seq_duplicates: int = 0
    seq_rewinds: int = 0
    extra: dict = field(default_factory=dict)
