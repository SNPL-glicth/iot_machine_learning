"""Action Guard Phase — real-time series_state from Redis.

Reads ``zenin:series_state:{series_id}`` from Redis to determine the
true operational state of the series, independent of any stale value
carried in the pipeline context.

States (inspired by STATS‑1 freshness model):
  * INITIALIZING → < 10 historical predictions
  * ACTIVE       → predictions received in the last 2 h
  * STALE        → no prediction in > 2 × expected interval
  * OFFLINE      → no prediction in > 24 h
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import PipelineContext

from iot_machine_learning.domain.services.actions.action_guard import ActionGuard

logger = logging.getLogger(__name__)

_SERIES_STATE_PREFIX = "zenin:series_state"
_SERIES_STATE_TTL_S = 86400            # 24 h

# Thresholds (seconds)
_ACTIVE_WINDOW_S = 7200                # 2 h
_STALE_WINDOW_S = 4 * 3600             # 4 h (2× expected 2h interval)
_OFFLINE_WINDOW_S = 86400              # 24 h
_INITIALIZING_MIN_PREDICTIONS = 10

_PREDICTION_COUNT_PREFIX = "zenin:prediction_count"


def _redis(ctx: PipelineContext) -> Optional[object]:
    store = getattr(getattr(ctx, "orchestrator", None), "_series_values_store", None)
    return store._redis if store and hasattr(store, "_redis") else None


def _resolve_series_state(redis: object, series_id: str) -> str:
    """Determine the real series state from Redis data."""
    try:
        # Check prediction count
        count_key = f"{_PREDICTION_COUNT_PREFIX}:{series_id}"
        raw_count = redis.get(count_key)
        count = int(raw_count) if raw_count is not None else 0

        if count < _INITIALIZING_MIN_PREDICTIONS:
            return "INITIALIZING"

        # Check last prediction timestamp
        state_key = f"{_SERIES_STATE_PREFIX}:{series_id}"
        raw_ts = redis.get(state_key)
        if raw_ts is None:
            return "UNKNOWN"

        last_ts = float(raw_ts)
        now = time.time()
        age = now - last_ts

        if age > _OFFLINE_WINDOW_S:
            return "OFFLINE"
        if age > _STALE_WINDOW_S:
            return "STALE"
        if age > _ACTIVE_WINDOW_S:
            return "STALE"
        return "ACTIVE"

    except Exception as exc:
        logger.debug("series_state_resolve_failed", extra={"series_id": series_id, "error": str(exc)})
        return "UNKNOWN"


def _update_state_on_redis(redis: object, series_id: str) -> None:
    """Update the last-prediction timestamp and increment prediction count."""
    now = time.time()
    try:
        state_key = f"{_SERIES_STATE_PREFIX}:{series_id}"
        redis.setex(state_key, _SERIES_STATE_TTL_S, now)
        count_key = f"{_PREDICTION_COUNT_PREFIX}:{series_id}"
        redis.incr(count_key)
        redis.expire(count_key, _SERIES_STATE_TTL_S * 7)
    except Exception as exc:
        logger.debug("series_state_update_failed", extra={"series_id": series_id, "error": str(exc)})


class ActionGuardPhase:
    """Phase: action guard with real-time series_state from Redis."""

    @property
    def name(self) -> str:
        return "action_guard"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            r = _redis(ctx)

            # ── Resolve real series_state from Redis ────────────
            series_state = _resolve_series_state(r, ctx.series_id) if r else "UNKNOWN"

            # ── Persist current prediction for next state calc ──
            if r:
                _update_state_on_redis(r, ctx.series_id)

            # ── Extract action info from explanation ────────────
            action_required = getattr(ctx, "max_action", "PREDICT") != "PREDICT"
            recommended_action = ctx.max_action if action_required else None
            severity = "NORMAL"
            if ctx.explanation:
                outcome = getattr(ctx.explanation, "outcome", None)
                if outcome:
                    severity = getattr(outcome, "severity", "NORMAL")

            guard = ActionGuard()
            guarded_action = guard.guard(
                action_required=action_required,
                recommended_action=recommended_action,
                severity=severity,
                series_state=series_state,
            )

            if not guarded_action.action_allowed:
                logger.warning(
                    "action_suppressed",
                    extra={
                        "series_id": ctx.series_id,
                        "series_state": series_state,
                        "original_action": recommended_action,
                        "reason": guarded_action.suppressed_reason,
                    },
                )

            return ctx.with_field(guarded_action=guarded_action)

        except Exception as e:
            logger.debug(f"action_guard_skipped: {e}")
            return ctx
