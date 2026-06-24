"""Observability Phase — async metrics collection for cognitive monitoring.

Never blocks the pipeline.  Metrics are persisted to Redis for the
dashboard endpoint ``GET /metrics/{series_id}``.

Metrics recorded:
  * Pipeline latency per phase (already collected by executor)
  * Confidence distribution per regime (Redis sorted set)
  * Anomaly rate per sensor, last 24 h (Redis sorted set, TTL 24h)
  * Engine reliability scores (from inhibition tracker)
  * ReadinessGate verdict distribution (Redis counter)
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from . import PipelineContext

logger = logging.getLogger(__name__)

_REDIS_KEY_CONFIDENCE = "zenin:metrics:confidence"
_REDIS_KEY_ANOMALY_RATE = "zenin:metrics:anomaly_rate"
_REDIS_KEY_GATE_VERDICT = "zenin:metrics:gate_verdict"
_REDIS_KEY_ENGINE_RELIABILITY = "zenin:metrics:engine_reliability"
_REDIS_METRICS_TTL_S = 86400  # 24 h

_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="observability")
    return _EXECUTOR


def _redis(ctx: PipelineContext) -> Optional[Any]:
    store = getattr(getattr(ctx, "orchestrator", None), "_series_values_store", None)
    return store._redis if store and hasattr(store, "_redis") else None


def _record_confidence(redis: Any, series_id: str, confidence: float, regime: str) -> None:
    key = f"{_REDIS_KEY_CONFIDENCE}:{regime}"
    try:
        redis.zadd(key, {series_id: confidence})
        redis.expire(key, _REDIS_METRICS_TTL_S)
    except Exception:
        pass


def _record_anomaly_rate(redis: Any, series_id: str) -> None:
    now = time.time()
    key = _REDIS_KEY_ANOMALY_RATE
    try:
        redis.zadd(key, {series_id: now})
        # Remove entries older than 24h
        cutoff = now - _REDIS_METRICS_TTL_S
        redis.zremrangebyscore(key, 0, cutoff)
        redis.expire(key, _REDIS_METRICS_TTL_S)
    except Exception:
        pass


def _record_gate_verdict(redis: Any, verdict: str) -> None:
    key = f"{_REDIS_KEY_GATE_VERDICT}:{verdict}"
    try:
        redis.incr(key)
        redis.expire(key, _REDIS_METRICS_TTL_S)
    except Exception:
        pass


def _record_engine_reliability(redis: Any, ctx: PipelineContext) -> None:
    tracker = getattr(ctx.orchestrator, "_reliability_tracker", None)
    if tracker is None:
        return
    inhib = getattr(ctx, "inhibition_states", None) or []
    for state in inhib:
        eng = getattr(state, "engine_name", None)
        if eng and hasattr(tracker, "p_broken"):
            try:
                pb = tracker.p_broken(ctx.series_id, eng)
                key = f"{_REDIS_KEY_ENGINE_RELIABILITY}:{eng}"
                redis.setex(key, _REDIS_METRICS_TTL_S, json.dumps({
                    "series_id": ctx.series_id,
                    "p_broken": round(pb, 4),
                    "ts": time.time(),
                }))
            except Exception:
                pass


class ObservabilityPhase:
    """Phase: async observability metrics — never blocks the pipeline."""

    @property
    def name(self) -> str:
        return "observability"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        r = _redis(ctx)
        if r is None:
            return ctx

        executor = _get_executor()
        confidence = getattr(ctx, "fused_confidence", 0.0) or 0.0
        regime = getattr(ctx, "regime", "UNKNOWN") or "UNKNOWN"

        # Confidence distribution (async)
        executor.submit(_record_confidence, r, ctx.series_id, confidence, regime)

        # Anomaly rate (async)
        z = getattr(ctx.profile, "z_score", 0.0) if ctx.profile else 0.0
        if abs(z) > 2.5:
            executor.submit(_record_anomaly_rate, r, ctx.series_id)

        # Gate verdict (async)
        max_action = getattr(ctx, "max_action", "PREDICT")
        executor.submit(_record_gate_verdict, r, max_action)

        # Engine reliability (async)
        executor.submit(_record_engine_reliability, r, ctx)

        return ctx
