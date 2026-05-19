"""Memory recall mixin for PredictSensorValueUseCase.

Handles optional cognitive-memory enrichment and feature-flag checks.
Includes a simple in-process TTL cache to reduce Weaviate round-trips.
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar, Optional

logger = logging.getLogger(__name__)


class _PredictionRecallMixin:
    """Mixin that provides memory recall enrichment."""

    _RECALL_CACHE_TTL_S: ClassVar[int] = 60
    _recall_cache: ClassVar[dict[str, tuple[Any, float]]] = {}

    def _try_recall(self: Any, prediction, series_id: str) -> Optional[dict]:
        if not self._should_recall():
            return None

        now = time.monotonic()
        cached = self.__class__._recall_cache.get(series_id)
        if cached is not None:
            result, ts = cached
            if (now - ts) < self.__class__._RECALL_CACHE_TTL_S:
                logger.debug(
                    "recall_cache_hit",
                    extra={"series_id": series_id, "age_s": round(now - ts, 1)},
                )
                return result

        try:
            recall_ctx = self._recall_enricher.enrich(prediction)
            if recall_ctx.has_context:
                result = recall_ctx.to_dict()
                self.__class__._recall_cache[series_id] = (result, now)
                return result
        except (ConnectionError, TimeoutError, TypeError, AttributeError) as exc:
            logger.warning(
                "memory_recall_failed",
                extra={
                    "series_id": series_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            from ...ml_service.metrics.observability import get_observability
            get_observability().silent_failures.record(
                "memory_recall", str(exc), {"series_id": series_id}
            )
        return None

    def _should_recall(self: Any) -> bool:
        """Check if memory recall is enabled and available."""
        if self._recall_enricher is None:
            return False
        if self._flags is None:
            return False
        return self._flags.ML_ENABLE_MEMORY_RECALL
