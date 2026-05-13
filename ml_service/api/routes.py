"""API routes for ML service.

Modular FastAPI router with all endpoints.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Depends, Request

from .dependencies import DbConnDep, verify_api_key
from .schemas import (
    MetacognitiveResponse,
    PredictRequest,
    PredictResponse,
    HealthResponse,
    StructuralAnalysisResponse,
)
from .services import PredictionService
from application.dto.prediction_dto import PredictionDTO
from .routes_observability import router as observability_router
from .routes_query import router as query_router

logger = logging.getLogger(__name__)

router = APIRouter()
router.include_router(observability_router)
router.include_router(query_router)


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    _: str = Depends(verify_api_key),
) -> HealthResponse:
    """Liveness probe — returns ok if process is running.

    Returns 503 if Zenin Queue Poller is enabled but the daemon
    thread has died, allowing Docker/K8s to restart the container.
    """
    broker_health = None
    try:
        from ..broker import get_broker_health
        broker_health = get_broker_health()
    except Exception:
        pass

    poller_status: dict | None = None
    poller = getattr(request.app.state, "zenin_poller", None)
    poller_thread = getattr(request.app.state, "zenin_poller_thread", None)
    if poller is not None and poller_thread is not None:
        stats = poller.stats
        alive = poller_thread.is_alive()
        poller_status = {
            "enabled": True,
            "alive": alive,
            "total_processed": stats.get("total_processed", 0),
            "total_errors": stats.get("total_errors", 0),
        }
        if not alive:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "degraded",
                    "broker": broker_health,
                    "poller": poller_status,
                },
            )
    else:
        poller_status = {"enabled": False}

    return HealthResponse(
        status="ok",
        broker=broker_health,
        poller=poller_status,
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


@router.post("/ml/predict", response_model=PredictResponse)
async def ml_predict(payload: PredictRequest, conn: DbConnDep, _: str = Depends(verify_api_key)) -> PredictResponse:
    """Generate a prediction for a sensor.
    
    Args:
        payload: Prediction request with sensor_id, horizon, window, dedupe
        conn: Database connection (injected)
        
    Returns:
        Prediction response with predicted value, confidence, etc.
    """
    logger.info(
        "[ML-SERVICE] /ml/predict sensor_id=%s horizon=%s window=%s dedupe=%s",
        payload.sensor_id,
        payload.horizon_minutes,
        payload.window,
        payload.dedupe_minutes,
    )
    
    try:
        service = PredictionService(conn)
        result = service.predict(
            sensor_id=payload.sensor_id,
            horizon_minutes=payload.horizon_minutes,
            window=payload.window,
            dedupe_minutes=payload.dedupe_minutes,
        )

        # Build enrichment sub-models (None-safe)
        sa_data = result.get("structural_analysis")
        structural_analysis = (
            StructuralAnalysisResponse(**sa_data) if sa_data else None
        )
        mc_data = result.get("metacognitive")
        metacognitive = (
            MetacognitiveResponse(**mc_data) if mc_data else None
        )

        # Extract decision output from metadata if available
        decision_output = None
        metadata = result.get("metadata", {})
        if metadata:
            from application.dto.decision_output import DecisionOutput
            decision_output = PredictionDTO(
                series_id=str(result["sensor_id"]),
                predicted_value=result["predicted_value"],
                confidence_score=result["confidence"],
                confidence_level=result.get("confidence_level", "unknown"),
                trend=result.get("trend", "stable"),
                engine_name=result.get("engine_used", "unknown"),
                explanation_text=result.get("explanation_summary", ""),
                audit_trace_id=result.get("audit_trace_id"),
                metadata=metadata,
            ).to_decision_output(metadata=metadata)

        return PredictResponse(
            # Base fields
            sensor_id=result["sensor_id"],
            model_id=result["model_id"],
            prediction_id=result["prediction_id"],
            predicted_value=result["predicted_value"],
            confidence=result["confidence"],
            target_timestamp=result["target_timestamp"],
            horizon_minutes=result["horizon_minutes"],
            window=result["window"],
            # Enrichment fields
            trend=result.get("trend"),
            engine_used=result.get("engine_used"),
            confidence_level=result.get("confidence_level"),
            structural_analysis=structural_analysis,
            metacognitive=metacognitive,
            audit_trace_id=result.get("audit_trace_id"),
            processing_time_ms=result.get("processing_time_ms"),
            explanation_summary=result.get("explanation_summary"),
            # Decision fields (interpretative analysis)
            decision=decision_output.decision if decision_output else None,
            verdict=decision_output.verdict if decision_output else None,
            severity=decision_output.severity if decision_output else None,
            action_required=decision_output.action_required if decision_output else None,
            action=decision_output.action if decision_output else None,
        )
    except ValueError as e:
        logger.warning("[ML-SERVICE] Prediction failed: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("[ML-SERVICE] Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/ml/broker/health")
async def broker_health(_: str = Depends(verify_api_key)) -> dict:
    """Get broker health status."""
    try:
        from ..broker import get_broker_health
        return get_broker_health()
    except Exception as e:
        return {
            "error": str(e),
            "connected": False,
        }


@router.get("/ml/metrics")
async def ml_metrics(_: str = Depends(verify_api_key)) -> dict:
    """Get ML service performance metrics."""
    try:
        from ..metrics import get_metrics
        metrics = get_metrics()
        return metrics.to_dict()


    except Exception as e:
        logger.exception("[ML-SERVICE] Metrics error: %s", str(e))
        return {"error": str(e)}


@router.get("/telemetry/ml-features/latest/{sensor_id}")
async def get_latest_telemetry_features(
    sensor_id: int,
    conn: DbConnDep,
    _: str = Depends(verify_api_key),
) -> dict:
    """Get latest telemetry metrics for a sensor.
    
    PHASE 4 FIX: Never returns 404 - returns empty response when no data.
    
    Args:
        sensor_id: Sensor ID
        conn: Database connection (injected)
        
    Returns:
        Latest telemetry metrics or empty response with status="no_data"
    """
    import time
    start_time = time.time()
    
    logger.info("[ML-SERVICE] /telemetry/ml-features/latest sensor_id=%s", sensor_id)
    
    try:
        from sqlalchemy import text
        
        # Query latest telemetry metrics for sensor
        query = text("""
            SELECT TOP 1
                sensor_id,
                range_key,
                computed_at,
                min_value,
                max_value,
                fluctuation,
                points_count,
                warning_min,
                warning_max,
                alert_min,
                alert_max
            FROM telemetry_sensor_metrics
            WHERE sensor_id = :sensor_id
            ORDER BY computed_at DESC
        """)
        
        result = conn.execute(query, {"sensor_id": sensor_id}).fetchone()
        
        latency_ms = (time.time() - start_time) * 1000
        
        if not result:
            logger.info(
                "[ML-SERVICE] No telemetry data for sensor_id=%s (latency=%sms)",
                sensor_id,
                f"{latency_ms:.2f}",
            )
            return {
                "sensor_id": sensor_id,
                "features": [],
                "status": "no_data",
                "latency_ms": f"{latency_ms:.2f}",
            }
        
        # Transform to response
        response = {
            "sensor_id": sensor_id,
            "features": {
                "range_key": result[1],
                "computed_at": result[2].isoformat() if result[2] else None,
                "min_value": float(result[3]) if result[3] is not None else None,
                "max_value": float(result[4]) if result[4] is not None else None,
                "fluctuation": float(result[5]) if result[5] is not None else None,
                "points_count": result[6],
                "warning_min": float(result[7]) if result[7] is not None else None,
                "warning_max": float(result[8]) if result[8] is not None else None,
                "alert_min": float(result[9]) if result[9] is not None else None,
                "alert_max": float(result[10]) if result[10] is not None else None,
            },
            "status": "ok",
            "latency_ms": f"{latency_ms:.2f}",
        }
        
        logger.info(
            "[ML-SERVICE] Telemetry features retrieved for sensor_id=%s (latency=%sms, points=%s)",
            sensor_id,
            f"{latency_ms:.2f}",
            result[6],
        )
        
        return response
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.exception(
            "[ML-SERVICE] Error fetching telemetry features for sensor_id=%s (latency=%sms): %s",
            sensor_id,
            f"{latency_ms:.2f}",
            str(e),
        )
        # PHASE 4 FIX: Never return 404, return error status instead
        return {
            "sensor_id": sensor_id,
            "features": [],
            "status": "error",
            "error": str(e),
            "latency_ms": f"{latency_ms:.2f}",
        }
