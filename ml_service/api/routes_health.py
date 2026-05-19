"""Health and readiness endpoints — extracted from routes.py for ≤180 lines."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Depends, Request

from .dependencies import DbConnDep, verify_api_key
from .schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    _: str = Depends(verify_api_key),
) -> HealthResponse:
    """Liveness probe. FIX P2-4: Full watchdog + SQL pool health.

    Returns 503 if any critical component is unhealthy.
    Returns 200 with degraded=true if system works but with reduced capacity.
    """
    broker_health = None
    try:
        from ..broker import get_broker_health
        broker_health = get_broker_health()
    except Exception:
        pass

    # Watchdog: ZeninQueuePoller
    poller_status: dict | None = None
    poller = getattr(request.app.state, "zenin_poller", None)
    if poller is not None:
        stats = poller.stats
        healthy = poller.is_healthy()
        poller_status = {"enabled": True, "healthy": healthy,
                         "total_processed": stats.get("total_processed", 0),
                         "total_errors": stats.get("total_errors", 0)}
        if not healthy:
            raise HTTPException(status_code=503, detail={
                "status": "unhealthy", "component": "zenin_poller",
                "broker": broker_health, "poller": poller_status})
    else:
        poller_status = {"enabled": False}

    # Watchdog: Redis broker consumer
    broker_obj = getattr(request.app.state, "broker", None)
    broker_watchdog_healthy = True
    if broker_obj is not None and hasattr(broker_obj, "is_healthy"):
        broker_watchdog_healthy = broker_obj.is_healthy()
    if not broker_watchdog_healthy:
        raise HTTPException(status_code=503, detail={
            "status": "unhealthy", "component": "redis_broker_watchdog",
            "broker": {"watchdog_healthy": False}, "poller": poller_status})

    # FIX P2-4: SQL pool usage check
    sql_pool_status: dict | None = None
    degraded = False
    try:
        from iot_ingest_services.common.db import get_engine as _get_db_engine
        pool = _get_db_engine().pool
        checked_out = pool.checkedout()
        pool_size = pool.size()
        overflow = pool.overflow()
        sql_pool_status = {"checked_out": checked_out, "pool_size": pool_size,
                           "overflow": overflow, "healthy": checked_out < pool_size}
        if checked_out >= pool_size:
            degraded = True
    except Exception:
        sql_pool_status = {"healthy": True, "note": "pool_stats_unavailable"}

    # FIX PROD-1: expose circuit breaker states
    cb_states: dict = {}
    try:
        from infrastructure.persistence.redis.circuit_factory import _redis_circuit_breakers
        for name in ("tsdb_adapter", "dist_window_adapter"):
            cb = _redis_circuit_breakers.get(name)
            if cb:
                cb_states[name] = cb.state.value
                if cb.state.value == "open":
                    degraded = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        degraded=degraded,
        broker=broker_health,
        poller=poller_status,
        sql_pool=sql_pool_status,
        tsdb_circuit=cb_states.get("tsdb_adapter", "unknown"),
        dist_window_circuit=cb_states.get("dist_window_adapter", "unknown"),
    )


@router.get("/ready")
async def ready(conn: DbConnDep, _: str = Depends(verify_api_key)) -> dict:
    """Readiness probe — checks DB connectivity."""
    try:
        from sqlalchemy import text
        conn.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as e:
        logger.warning("[ML-SERVICE] Readiness check failed: %s", e)
        raise HTTPException(status_code=503, detail="not ready")
