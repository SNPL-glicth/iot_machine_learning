"""Inhibit Phase — Beta-Bernoulli reliability inhibition only.

Legacy 3-threshold mode has been removed.  The sole inhibition authority
is the EngineReliabilityTracker Beta-Bernoulli posterior.

Redis persistence:
  zenin:inhibition:{series_id}:{engine} — suppression factor (TTL 3600s)

Thread-safe via threading.RLock on _prev_suppression state.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from . import PipelineContext

from ...analysis.types import InhibitionState

logger = logging.getLogger(__name__)

_INHIBITION_REDIS_PREFIX = "zenin:inhibition"
_INHIBITION_REDIS_TTL = 3600


def _get_redis(ctx: PipelineContext) -> Any:
    store = getattr(ctx.orchestrator, "_series_values_store", None)
    if store is not None:
        return getattr(store, "_redis", None)
    return None


def _redis_suppression_key(series_id: str, engine: str) -> str:
    return f"{_INHIBITION_REDIS_PREFIX}:{series_id}:{engine}"


def _load_suppression_from_redis(redis: Any, key: str) -> Optional[float]:
    if redis is None:
        return None
    try:
        raw = redis.get(key)
        if raw is not None:
            return float(raw.decode() if isinstance(raw, bytes) else raw)
    except Exception:
        pass
    return None


def _save_suppression_to_redis(redis: Any, key: str, value: float) -> None:
    if redis is None:
        return
    try:
        redis.setex(key, _INHIBITION_REDIS_TTL, str(round(value, 4)))
    except Exception:
        pass


class InhibitPhase:
    """Phase 4: Inhibition gate — Beta-Bernoulli reliability only.

    Thread-safe: uses threading.RLock for _prev_suppression access.
    Redis-backed: suppression state survives restarts.
    """

    def __init__(self) -> None:
        self._prev_suppression: Dict[Tuple[str, str], float] = {}
        self._last_update: Dict[Tuple[str, str], float] = {}
        self._lock = threading.RLock()
        self._suppression_half_life = 300.0
        self._suppression_smoothing = 0.3
        self._max_entries = 10000
        self._redis_warned: bool = False

    @property
    def name(self) -> str:
        return "inhibit"

    def _apply_decay(self, prev: float, elapsed: float) -> float:
        rate = math.log(2) / self._suppression_half_life
        return prev * math.exp(-rate * elapsed)

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        orchestrator = ctx.orchestrator
        redis = _get_redis(ctx)
        series_id = ctx.series_id

        # Get the reliability tracker
        reliability = getattr(orchestrator, "_reliability_tracker", None)
        if reliability is None:
            if not self._redis_warned:
                logger.warning(
                    "inhibit_no_reliability_tracker",
                    extra={"series_id": series_id},
                )
                self._redis_warned = True
            return ctx.with_field(
                inhibition_states=[],
                mediated_weights=ctx.plasticity_weights,
            )

        weights = ctx.plasticity_weights or {}
        perceptions = ctx.perceptions or []
        states: List[InhibitionState] = []
        now = time.monotonic()

        for p in perceptions:
            eng = p.engine_name
            bw = weights.get(eng, 0.0)
            key = (series_id, eng)

            # Check Beta-Bernoulli reliability
            reliable = reliability.is_reliable(series_id, eng)

            with self._lock:
                # Load previous suppression from Redis on first access
                if key not in self._prev_suppression:
                    rkey = _redis_suppression_key(series_id, eng)
                    rval = _load_suppression_from_redis(redis, rkey)
                    if rval is not None:
                        self._prev_suppression[key] = rval
                        self._last_update[key] = now

                # Apply decay to previous suppression
                if key in self._prev_suppression:
                    elapsed = now - self._last_update.get(key, now)
                    prev = self._apply_decay(self._prev_suppression[key], elapsed)
                else:
                    prev = 0.0

                # Compute instant suppression from reliability
                if reliable:
                    instant = 0.0
                    reason = "none"
                else:
                    p_broken = reliability.p_broken(series_id, eng)
                    instant = min(p_broken, 0.95)
                    reason = f"unreliable p_broken={p_broken:.3f}"

                # EMA smoothing
                smoothed = (
                    (1.0 - self._suppression_smoothing) * prev
                    + self._suppression_smoothing * instant
                )

                # Update in-memory state
                self._prev_suppression[key] = smoothed
                self._last_update[key] = now

                # LRU eviction
                if len(self._prev_suppression) >= self._max_entries:
                    oldest = min(self._last_update, key=self._last_update.get)
                    self._prev_suppression.pop(oldest, None)
                    self._last_update.pop(oldest, None)

            # Persist to Redis
            rkey = _redis_suppression_key(series_id, eng)
            _save_suppression_to_redis(redis, rkey, smoothed)

            inhibited = bw * (1.0 - smoothed)

            states.append(InhibitionState(
                engine_name=eng,
                base_weight=bw,
                inhibited_weight=inhibited,
                inhibition_reason=reason,
                suppression_factor=smoothed,
            ))

        if not redis and not self._redis_warned:
            logger.warning(
                "inhibit_no_redis_fallback_in_memory",
                extra={"series_id": series_id},
            )
            self._redis_warned = True

        return ctx.with_field(
            inhibition_states=states,
            mediated_weights=ctx.plasticity_weights,
        )
