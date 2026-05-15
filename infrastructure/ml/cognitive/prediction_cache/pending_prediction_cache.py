"""Redis-backed cache for pending predictions using ZSET for O(log N) matching."""

from __future__ import annotations

import logging
from typing import List, Optional

from iot_machine_learning.infrastructure.security.redis_namespace import (
    RedisNamespace,
    get_namespace,
)

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS: int = 7200  # 2h max TTL for ZSET


class PendingPredictionCache:
    """Manages pending prediction entries in Redis ZSET for fast matching.

    Key format: {env}:{app}:{tenant}:predictions:{series_id}:pending
    ZSET score: target_timestamp_epoch (float)
    ZSET member: prediction_id
    """

    def __init__(
        self,
        redis_client=None,
        namespace: Optional[RedisNamespace] = None,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        tenant_id: str = "default",
    ) -> None:
        self._redis = redis_client
        self._namespace = namespace or get_namespace(tenant_id=tenant_id)
        self._ttl = ttl_seconds

    def _get_key(self, series_id: str) -> str:
        return self._namespace.key(
            resource_type="predictions",
            resource_id=series_id,
            suffix="pending",
        )

    def register(
        self,
        series_id: str,
        prediction_id: str,
        target_timestamp_epoch: float,
    ) -> None:
        """Register a pending prediction in the ZSET."""
        if self._redis is None:
            return
        key = self._get_key(series_id)
        try:
            self._redis.zadd(key, {prediction_id: target_timestamp_epoch})
            self._redis.expire(key, self._ttl)
            logger.debug(
                "pending_prediction_registered",
                extra={
                    "series_id": series_id,
                    "prediction_id": prediction_id,
                    "target_epoch": target_timestamp_epoch,
                },
            )
        except Exception as e:
            logger.warning(f"pending_prediction_register_failed: {e}")

    def has_pending(self, series_id: str, now_epoch: float) -> bool:
        """Return True if there is any pending prediction with target >= now."""
        if self._redis is None:
            return False
        key = self._get_key(series_id)
        try:
            count = self._redis.zcount(key, now_epoch, "+inf")
            return bool(count and count > 0)
        except Exception as e:
            logger.warning(f"pending_prediction_has_pending_failed: {e}")
            return False

    def find_match(
        self,
        series_id: str,
        reading_timestamp_epoch: float,
        tolerance_seconds: float,
    ) -> Optional[str]:
        """Find closest prediction_id within tolerance of reading timestamp."""
        if self._redis is None:
            return None
        key = self._get_key(series_id)
        min_score = reading_timestamp_epoch - tolerance_seconds
        max_score = reading_timestamp_epoch + tolerance_seconds
        try:
            candidates = self._redis.zrangebyscore(
                key, min_score, max_score, withscores=True,
            )
            if not candidates:
                return None
            # candidates: list of (prediction_id_bytes, score)
            # Pick the one closest to reading_timestamp_epoch
            closest = min(
                candidates,
                key=lambda item: abs(float(item[1]) - reading_timestamp_epoch),
            )
            prediction_id = closest[0]
            if isinstance(prediction_id, bytes):
                prediction_id = prediction_id.decode()
            logger.debug(
                "pending_prediction_matched",
                extra={
                    "series_id": series_id,
                    "prediction_id": prediction_id,
                    "reading_epoch": reading_timestamp_epoch,
                },
            )
            return prediction_id
        except Exception as e:
            logger.warning(f"pending_prediction_match_failed: {e}")
            return None

    def remove(self, series_id: str, prediction_id: str) -> None:
        """Remove a prediction_id from the ZSET."""
        if self._redis is None:
            return
        key = self._get_key(series_id)
        try:
            self._redis.zrem(key, prediction_id)
        except Exception as e:
            logger.warning(f"pending_prediction_remove_failed: {e}")
