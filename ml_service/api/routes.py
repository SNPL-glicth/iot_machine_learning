"""API routes for ML service.

Modular FastAPI router with all endpoints.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Depends

from .dependencies import DbConnDep, verify_api_key
from .schemas import (
    MetacognitiveResponse,
    PredictRequest,
    PredictResponse,
    HealthResponse,
    StructuralAnalysisResponse,
)
from .services import PredictionService
from .routes_observability import router as observability_router
from .routes_query import router as query_router

logger = logging.getLogger(__name__)

router = APIRouter()
router.include_router(observability_router)
router.include_router(query_router)


@router.get("/health", response_model=HealthResponse)
async def health(_: str = Depends(verify_api_key)) -> HealthResponse:
    """Liveness probe — always returns ok if process is running."""
    broker_health = None
    try:
        from ..broker import get_broker_health
        broker_health = get_broker_health()
    except Exception:
        pass
    
    return HealthResponse(
        status="ok",
        broker=broker_health,
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
