"""API routes for ML service.

Modular FastAPI router with all endpoints.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from .dependencies import DbConnDep
from .schemas import PredictRequest, PredictResponse, HealthResponse
from .services import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    logger.info("[ML-SERVICE] Health check")
    
    # Get broker health if available
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


@router.post("/ml/predict", response_model=PredictResponse)
async def ml_predict(payload: PredictRequest, conn: DbConnDep) -> PredictResponse:
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
        
        return PredictResponse(
            sensor_id=result["sensor_id"],
            model_id=result["model_id"],
            prediction_id=result["prediction_id"],
            predicted_value=result["predicted_value"],
            confidence=result["confidence"],
            target_timestamp=result["target_timestamp"],
            horizon_minutes=result["horizon_minutes"],
            window=result["window"],
        )
    except ValueError as e:
        logger.warning("[ML-SERVICE] Prediction failed: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("[ML-SERVICE] Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/ml/broker/health")
async def broker_health() -> dict:
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
async def ml_metrics() -> dict:
    """Get ML service performance metrics."""
    try:
        from ..metrics import get_metrics
        metrics = get_metrics()
        return metrics.to_dict()
    except Exception as e:
        logger.exception("[ML-SERVICE] Metrics error: %s", str(e))
        return {"error": str(e)}
