"""Auditor puro del feed (FASE 0).

Preguntas que responde por tramo, en orden de llegada:
correcto / completo / sincronizado / duplicado / retrasado /
perdido / mezclado (símbolo, status, secuencia).

Nada se descarta: todo se cuenta y se expone en el reporte.
"""

from __future__ import annotations

from typing import Any, Callable
from .types import AuditedEvent, FeedAnomaly, FeedAuditReport

__all__ = ["audit_feed"]


def _audit_sequence_step(
    seq: int,
    seen_seq: set[int],
    last_seq: int | None,
    record: Callable[[list[FeedAnomaly], FeedAnomaly], None],
    anomalies: list[FeedAnomaly],
) -> tuple[int, int, int]:
    """Audita un paso de secuencia. Retorna (delta_duplicates, delta_gaps, delta_rewinds)."""
    if seq in seen_seq:
        record(anomalies, FeedAnomaly(
            kind="seq_duplicate", received=float(seq),
            detail=f"seq {seq} duplicada",
        ))
        return 1, 0, 0

    seen_seq.add(seq)
    if last_seq is None or seq == last_seq + 1:
        return 0, 0, 0

    if seq > last_seq + 1:
        gaps = seq - last_seq - 1
        record(anomalies, FeedAnomaly(
            kind="seq_gap", expected=float(last_seq + 1),
            received=float(seq),
            detail=f"salto {last_seq}->{seq}",
        ))
        return 0, gaps, 0

    record(anomalies, FeedAnomaly(
        kind="seq_rewind", expected=float(last_seq),
        received=float(seq), detail="secuencia retrocede",
    ))
    return 0, 0, 1


def audit_feed(
    events: list[AuditedEvent] | tuple[AuditedEvent, ...],
    *,
    symbol: str,
    expected_interval_seconds: float | None = None,
    gap_threshold: float = 1.5,
    stale_lag_seconds: float | None = None,
    max_anomalies: int = 200,
) -> FeedAuditReport:
    """Audita un tramo de eventos en orden de llegada."""
    events = list(events)
    total = len(events)
    anomalies: list[FeedAnomaly] = []
    gap_events: list[FeedAnomaly] = []

    def _record(target: list[FeedAnomaly], anomaly: FeedAnomaly) -> None:
        if len(anomalies) + len(gap_events) < max_anomalies:
            target.append(anomaly)

    if total == 0:
        return FeedAuditReport(
            symbol=symbol, total=0, unique=0, duplicates=0,
            out_of_order=0, missing_intervals=0,
        )

    seen_ts: set[float] = set()
    duplicates = 0
    out_of_order = 0
    max_ts_so_far: float | None = None
    symbols: list[str] = []
    statuses: list[str] = []
    lags: list[float] = []
    stale_count = 0
    seq_gaps = 0
    seq_duplicates = 0
    seq_rewinds = 0
    last_seq: int | None = None
    seen_seq: set[int] = set()

    for ev in events:
        if ev.symbol not in symbols:
            symbols.append(ev.symbol)
            if ev.symbol != symbol:
                _record(anomalies, FeedAnomaly(
                    kind="symbol_mix", received=ev.timestamp,
                    detail=f"símbolo {ev.symbol!r} en tramo {symbol!r}",
                ))

        if ev.status not in statuses:
            statuses.append(ev.status)

        if ev.timestamp in seen_ts:
            duplicates += 1
            _record(anomalies, FeedAnomaly(
                kind="duplicate", received=ev.timestamp,
                detail=f"ts {ev.timestamp} repetido ({ev.kind})",
            ))
        else:
            seen_ts.add(ev.timestamp)

        if max_ts_so_far is not None and ev.timestamp < max_ts_so_far:
            out_of_order += 1
            _record(anomalies, FeedAnomaly(
                kind="out_of_order", expected=max_ts_so_far,
                received=ev.timestamp, detail="llegada tardía",
            ))

        if max_ts_so_far is None or ev.timestamp > max_ts_so_far:
            max_ts_so_far = ev.timestamp

        if ev.arrival_ts is not None:
            lag = ev.arrival_ts - ev.timestamp
            lags.append(lag)
            if stale_lag_seconds is not None and lag > stale_lag_seconds:
                stale_count += 1
                _record(anomalies, FeedAnomaly(
                    kind="stale", expected=ev.timestamp,
                    received=ev.arrival_ts,
                    detail=f"lag {lag:.3f}s > umbral {stale_lag_seconds}s",
                ))

        if ev.seq is not None:
            dup_d, gap_d, rew_d = _audit_sequence_step(
                ev.seq, seen_seq, last_seq, _record, anomalies
            )
            seq_duplicates += dup_d
            seq_gaps += gap_d
            seq_rewinds += rew_d
            if last_seq is None or ev.seq > last_seq:
                last_seq = ev.seq

    # Gaps sobre tiempos únicos ordenados (pérdida real, no orden de llegada).
    missing = 0
    ordered = sorted(seen_ts)
    if expected_interval_seconds and len(ordered) >= 2:
        interval = expected_interval_seconds
        for prev, curr in zip(ordered, ordered[1:]):
            delta = curr - prev
            if delta > gap_threshold * interval:
                lost = round(delta / interval) - 1
                if lost > 0:
                    missing += lost
                    _record(gap_events, FeedAnomaly(
                        kind="gap", expected=prev + interval,
                        received=curr,
                        detail=f"faltan ~{lost} intervalos "
                               f"({prev}->{curr}, Δ={delta:.1f}s)",
                    ))

    unique = len(seen_ts)
    denom = unique + missing
    completeness = (unique / denom * 100.0) if denom else 100.0
    avg_lag = (sum(lags) / len(lags)) if lags else None
    max_lag = max(lags) if lags else None

    replay_mixed = "replay" in statuses and (
        "realtime" in statuses or "delayed" in statuses
    )

    return FeedAuditReport(
        symbol=symbol,
        total=total,
        unique=unique,
        duplicates=duplicates,
        out_of_order=out_of_order,
        missing_intervals=missing,
        gap_events=tuple(gap_events),
        anomalies=tuple(anomalies),
        completeness_pct=round(completeness, 4),
        stale_count=stale_count,
        avg_lag_seconds=avg_lag,
        max_lag_seconds=max_lag,
        symbols_seen=tuple(symbols),
        statuses_seen=tuple(statuses),
        first_ts=ordered[0] if ordered else None,
        last_ts=ordered[-1] if ordered else None,
        seq_gaps=seq_gaps,
        seq_duplicates=seq_duplicates,
        seq_rewinds=seq_rewinds,
        extra={"replay_mixed_with_live": replay_mixed},
    )
