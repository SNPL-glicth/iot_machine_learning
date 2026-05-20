"""Application warm-start — pre-loads engines and pools on startup.

Reduces cold-start latency by eagerly initializing:
- Prediction engines (Taylor, Statistical, Cognitive)
- Redis connection pool
- DB connection pool
- Configuration cache
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


class WarmupResult:
    """Result of warm-start initialization."""

    def __init__(self) -> None:
        self.components: Dict[str, Dict[str, Any]] = {}
        self.total_ms: float = 0.0

    def record(self, name: str, success: bool, elapsed_ms: float, detail: str = "") -> None:
        self.components[name] = {
            "success": success,
            "elapsed_ms": round(elapsed_ms, 2),
            "detail": detail,
        }

    @property
    def all_healthy(self) -> bool:
        return all(c["success"] for c in self.components.values())

    def to_dict(self) -> dict:
        return {
            "all_healthy": self.all_healthy,
            "total_ms": round(self.total_ms, 2),
            "components": self.components,
        }


def warmup_engines(result: WarmupResult) -> None:
    """Pre-load ML prediction engines."""
    t0 = time.monotonic()
    try:
        from iot_machine_learning.infrastructure.ml.engines.taylor.engine import TaylorPredictionEngine
        engine = TaylorPredictionEngine()
        # Warm internal caches with a dummy prediction
        engine.predict([20.0, 20.1, 20.2, 20.3, 20.4])
        elapsed = (time.monotonic() - t0) * 1000.0
        result.record("taylor_engine", True, elapsed, "initialized + warm prediction")
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000.0
        result.record("taylor_engine", False, elapsed, str(exc))

    t0 = time.monotonic()
    try:
        from iot_machine_learning.infrastructure.ml.engines.statistical.engine import StatisticalPredictionEngine
        engine = StatisticalPredictionEngine()
        engine.predict([20.0, 20.1, 20.2, 20.3, 20.4])
        elapsed = (time.monotonic() - t0) * 1000.0
        result.record("statistical_engine", True, elapsed, "initialized + warm prediction")
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000.0
        result.record("statistical_engine", False, elapsed, str(exc))

    t0 = time.monotonic()
    try:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import MetaCognitiveOrchestrator
        from iot_machine_learning.infrastructure.ml.engines.taylor.engine import TaylorPredictionEngine
        from iot_machine_learning.infrastructure.ml.engines.statistical.engine import StatisticalPredictionEngine
        from iot_machine_learning.ml_service.config.feature_flags import get_feature_flags
        orch = MetaCognitiveOrchestrator(engines=[TaylorPredictionEngine(), StatisticalPredictionEngine()])
        orch.predict(
            series_id="warmup",
            values=[20.0, 20.1, 20.2, 20.3, 20.4],
            flags_snapshot=get_feature_flags(),
        )
        elapsed = (time.monotonic() - t0) * 1000.0
        result.record("cognitive_orchestrator", True, elapsed, "initialized + warm prediction")
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000.0
        result.record("cognitive_orchestrator", False, elapsed, str(exc))


def warmup_redis(result: WarmupResult) -> None:
    """Verify Redis connectivity and pre-warm connection pool."""
    t0 = time.monotonic()
    try:
        import os
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        from redis import Redis
        client = Redis.from_url(redis_url, socket_connect_timeout=3)
        pong = client.ping()
        elapsed = (time.monotonic() - t0) * 1000.0
        result.record("redis", pong, elapsed, f"ping={'PONG' if pong else 'FAIL'}")
        client.close()
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000.0
        result.record("redis", False, elapsed, str(exc))


def warmup_db(result: WarmupResult) -> None:
    """Verify database connectivity."""
    t0 = time.monotonic()
    try:
        from iot_ingest_services.common.db import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        elapsed = (time.monotonic() - t0) * 1000.0
        result.record("database", True, elapsed, "connection verified")
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000.0
        result.record("database", False, elapsed, str(exc))


def warmup_config(result: WarmupResult) -> None:
    """Pre-load configuration models."""
    t0 = time.monotonic()
    try:
        from iot_machine_learning.ml_service.config.feature_flags import get_feature_flags
        from iot_machine_learning.ml_service.config.batch_config import BatchConfig
        flags = get_feature_flags()
        batch_cfg = BatchConfig()
        elapsed = (time.monotonic() - t0) * 1000.0
        result.record("config", True, elapsed, f"flags loaded, batch_workers={batch_cfg.ML_BATCH_MAX_WORKERS}")
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000.0
        result.record("config", False, elapsed, str(exc))


def run_warmup() -> WarmupResult:
    """Execute full warm-start sequence.

    Called during application lifespan startup.
    Non-fatal: failures are logged but don't prevent startup.
    """
    t_total = time.monotonic()
    result = WarmupResult()

    logger.info("warmup_start")

    warmup_config(result)
    warmup_engines(result)
    warmup_redis(result)
    warmup_db(result)

    result.total_ms = (time.monotonic() - t_total) * 1000.0

    if result.all_healthy:
        logger.info("warmup_complete", extra=result.to_dict())
    else:
        logger.warning("warmup_partial", extra=result.to_dict())

    return result
