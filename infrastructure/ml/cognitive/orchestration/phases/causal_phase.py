"""Causal Phase — Redis‑backed temporal correlation for anomaly chains.

For each anomaly detected in the current pipeline run, the phase:
  1. Scans Redis at ``zenin:anomaly_history:{equipment_id}:*`` for
     recent (≤6 h) anomalies in sibling parameters.
  2. Builds ``causal_events`` linking a preceding anomaly to the
     current one when a temporal gap exists.
  3. Writes the current anomaly back to Redis (TTL 24 h) so future
     runs can detect chains.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from . import PipelineContext

logger = logging.getLogger(__name__)

ANOMALY_HISTORY_PREFIX = "zenin:anomaly_history"
ANOMALY_TTL_S = 86400           # 24 h
RECENT_WINDOW_S = 21600          # 6 h
ANOMALY_Z_THRESHOLD = 2.5       # z‑score threshold for anomalies
REDIS_SCAN_COUNT = 100


@dataclass
class CausalEvent:
    preceding_param: str
    preceding_timestamp: float
    current_param: str
    current_timestamp: float
    time_delta_minutes: float
    correlation_strength: float = 0.5


def _redis(store: Any) -> Optional[Any]:
    """Extract Redis client from *store* or return None."""
    return store._redis if store and hasattr(store, "_redis") else None


def _equipment_id(ctx: PipelineContext) -> Optional[str]:
    """Derive equipment id from the context."""
    fc = getattr(ctx, "feature_context", None)
    if fc is not None:
        sp = getattr(fc, "sensor_profile", None)
        if sp is not None:
            eq = getattr(sp, "equipment_class", None)
            if eq is not None:
                return eq.value if hasattr(eq, "value") else str(eq).lower()
    # Fallback: use the first part of series_id before '_'
    parts = ctx.series_id.split("_")
    if len(parts) > 1:
        return parts[0]
    return None


def _anomaly_score(ctx: PipelineContext) -> float:
    z = getattr(ctx.profile, "z_score", 0.0) if ctx.profile else 0.0
    return abs(z)


def _is_anomaly(ctx: PipelineContext) -> bool:
    return _anomaly_score(ctx) > ANOMALY_Z_THRESHOLD


def _discover_related_params(
    redis: Any, equipment_id: str, current_param: str,
) -> List[str]:
    """Scan Redis for sibling parameter keys under the same equipment."""
    pattern = f"{ANOMALY_HISTORY_PREFIX}:{equipment_id}:*"
    params: List[str] = []
    try:
        cursor = 0
        while True:
            cursor, keys = redis.scan(cursor, match=pattern, count=REDIS_SCAN_COUNT)
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                # Extract param name after the last ':'
                param = key_str.rsplit(":", 1)[-1]
                if param and param != current_param:
                    params.append(param)
            if cursor == 0:
                break
    except Exception as exc:
        logger.debug("redis_scan_failed", extra={"error": str(exc)})
    return params


def _check_recent_anomalies(
    redis: Any, equipment_id: str, params: List[str], now: float,
) -> List[Dict[str, Any]]:
    """Fetch anomaly records for *params* that fall within ``RECENT_WINDOW_S``."""
    results: List[Dict[str, Any]] = []
    cutoff = now - RECENT_WINDOW_S
    for param in params:
        key = f"{ANOMALY_HISTORY_PREFIX}:{equipment_id}:{param}"
        try:
            raw = redis.get(key)
            if raw is None:
                continue
            data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
            ts = float(data.get("timestamp", 0))
            if ts >= cutoff:
                results.append({
                    "param": param,
                    "timestamp": ts,
                    "z_score": float(data.get("z_score", 0)),
                })
        except Exception as exc:
            logger.debug("redis_read_failed", extra={"key": key, "error": str(exc)})
    return results


def _save_current_anomaly(
    redis: Any, equipment_id: str, param: str, z_score: float, now: float,
) -> None:
    """Store the current anomaly in Redis for future causal detection."""
    key = f"{ANOMALY_HISTORY_PREFIX}:{equipment_id}:{param}"
    payload = json.dumps({
        "param": param,
        "timestamp": now,
        "z_score": round(z_score, 4),
    })
    try:
        redis.setex(key, ANOMALY_TTL_S, payload)
        logger.debug(
            "causal_anomaly_stored",
            extra={"key": key, "ttl": ANOMALY_TTL_S},
        )
    except Exception as exc:
        logger.debug("redis_write_failed", extra={"key": key, "error": str(exc)})


class CausalPhase:
    """Phase: Redis‑backed temporal correlation for anomaly chains.

    Looks at sibling parameters under the same ``equipment_id`` and
    builds ``causal_events`` when a preceding anomaly is found within
    the last 6 hours.
    """

    @property
    def name(self) -> str:
        return "causal"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        orchestrator = ctx.orchestrator
        store = getattr(orchestrator, "_series_values_store", None)
        r = _redis(store)
        if r is None:
            logger.debug("causal_phase_no_redis")
            return ctx

        now = time.time()
        eq = _equipment_id(ctx)
        if eq is None:
            logger.debug("causal_phase_no_equipment_id")
            return ctx

        param = ctx.series_id
        z = _anomaly_score(ctx)

        # Build causal chain when current reading is anomalous
        causal_events: List[CausalEvent] = []
        if _is_anomaly(ctx):
            related = _discover_related_params(r, eq, param)
            if related:
                recent = _check_recent_anomalies(r, eq, related, now)
                for rec in recent:
                    delta_min = (now - rec["timestamp"]) / 60.0
                    strength = min(1.0, max(0.1, 1.0 - delta_min / 360.0))
                    causal_events.append(CausalEvent(
                        preceding_param=rec["param"],
                        preceding_timestamp=rec["timestamp"],
                        current_param=param,
                        current_timestamp=now,
                        time_delta_minutes=round(delta_min, 1),
                        correlation_strength=round(strength, 3),
                    ))

            # Always persist current anomaly for future chains
            _save_current_anomaly(r, eq, param, z, now)

        logger.debug(
            "causal_phase_completed",
            extra={
                "series_id": ctx.series_id,
                "equipment": eq,
                "n_causal_events": len(causal_events),
            },
        )

        ctx.metadata["causal_events"] = [asdict(e) for e in causal_events]
        return ctx.with_field(
            causal_events=[asdict(e) for e in causal_events],
        )
