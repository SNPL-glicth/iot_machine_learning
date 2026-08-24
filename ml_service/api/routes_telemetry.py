"""Telemetry endpoints — extracted from routes.py for ≤180 lines."""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

from .dependencies import DbConnDep, verify_api_key
from .schemas import TelemetryRequest, TelemetryResponse, ActionEnvelopeResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory device state cache for delta computation
# In production, replace with Redis or persistent store
_device_state_cache: Dict[str, Dict[str, Any]] = {}


@router.get("/telemetry/ml-features/latest/{sensor_id}")
async def get_latest_telemetry_features(
    sensor_id: int,
    conn: DbConnDep,
    _: str = Depends(verify_api_key),
) -> dict:
    """Get latest telemetry metrics for a sensor.
    
    PHASE 4 FIX: Never returns 404 - returns empty response when no data.
    """
    start_time = time.time()
    
    logger.info("[ML-SERVICE] /telemetry/ml-features/latest sensor_id=%s", sensor_id)
    
    try:
        from sqlalchemy import text
        
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
                sensor_id, f"{latency_ms:.2f}",
            )
            return {
                "sensor_id": sensor_id,
                "features": [],
                "status": "no_data",
                "latency_ms": f"{latency_ms:.2f}",
            }
        
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
            "[ML-SERVICE] Telemetry features retrieved for sensor_id=%s (latency=%sms)",
            sensor_id, f"{latency_ms:.2f}",
        )
        return response
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.exception(
            "[ML-SERVICE] Error fetching telemetry features for sensor_id=%s: %s",
            sensor_id, str(e),
        )
        return {
            "sensor_id": sensor_id,
            "features": [],
            "status": "error",
            "error": str(e),
            "latency_ms": f"{latency_ms:.2f}",
        }


@router.post("/telemetry/{device_id}", response_model=TelemetryResponse)
async def stream_telemetry(
    device_id: str,
    request: TelemetryRequest,
    req: Request,
    _: str = Depends(verify_api_key),
) -> TelemetryResponse:
    """
    Real-time telemetry streaming endpoint for Rosa Roja Engine.
    
    Accepts multidimensional telemetry vector, computes delta_state,
    runs through Rosa Roja pipeline, and returns ActionEnvelope.
    
    Args:
        device_id: Device identifier (e.g., chiller_01, ca_01)
        request: TelemetryRequest with timestamp and telemetry_vector
        
    Returns:
        TelemetryResponse with action, confidence, and ActionEnvelope
        
    Flow:
        1. Compute delta_state = current_state - previous_state (from cache)
        2. Compute delta_time = current_timestamp - previous_timestamp
        3. Call RosaRojaEngine.process_event(delta_state, delta_time)
        4. Dispatch ExecutionPlan to IoTActuatorHandler
        5. Update cache with current state
        6. Return response
    """
    start_time = time.perf_counter()
    
    logger.info(
        "[ML-SERVICE] /telemetry/%s received vector of %d dims",
        device_id, len(request.telemetry_vector)
    )
    
    # Get Rosa Roja engine and actuator from app state
    engine = getattr(req.app.state, "rosa_roja", None)
    actuator = getattr(req.app.state, "actuator", None)
    
    if engine is None:
        logger.warning("[ML-SERVICE] Rosa Roja engine not initialized")
        raise HTTPException(
            status_code=503,
            detail="Rosa Roja engine not available. Check service initialization."
        )
    
    if actuator is None:
        logger.warning("[ML-SERVICE] Actuator handler not initialized")
        raise HTTPException(
            status_code=503,
            detail="Actuator handler not available."
        )
    
    try:
        # Validate request
        if request.device_id != device_id:
            raise HTTPException(
                status_code=400,
                detail="device_id in path does not match request body"
            )
        
        # Get previous state from cache
        prev_state = _device_state_cache.get(device_id)
        current_vector = request.telemetry_vector
        current_time = request.timestamp
        
        if prev_state is None:
            # First reading - initialize cache, return HOLD
            _device_state_cache[device_id] = {
                "state": current_vector,
                "timestamp": current_time,
            }
            
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.info(f"[ML-SERVICE] First reading for {device_id}, initializing cache")
            
            return TelemetryResponse(
                device_id=device_id,
                action="HOLD",
                confidence=0.0,
                envelope=None,
                invalidation_step=None,
                regime_alert=False,
                processing_time_ms=processing_time,
                veto_details={"reason": "First reading - initializing state cache"},
            )
        
        # Compute delta_state and delta_time
        prev_vector = prev_state["state"]
        prev_time = prev_state["timestamp"]
        
        # Ensure same dimensionality
        if len(current_vector) != len(prev_vector):
            logger.warning(
                f"[ML-SERVICE] Dimension mismatch for {device_id}: "
                f"current={len(current_vector)}, prev={len(prev_vector)}"
            )
            # Reset cache
            _device_state_cache[device_id] = {"state": current_vector, "timestamp": current_time}
            
            processing_time = (time.perf_counter() - start_time) * 1000
            return TelemetryResponse(
                device_id=device_id,
                action="HOLD",
                confidence=0.0,
                envelope=None,
                invalidation_step=None,
                regime_alert=False,
                processing_time_ms=processing_time,
                veto_details={"reason": f"Dimension mismatch: {len(current_vector)} vs {len(prev_vector)}"},
            )
        
        import numpy as np
        delta_state = np.array(current_vector, dtype=np.float64) - np.array(prev_vector, dtype=np.float64)
        delta_time = max(current_time - prev_time, 1e-6)  # Avoid division by zero
        
        # Update cache BEFORE processing (so we have latest state even if processing fails)
        _device_state_cache[device_id] = {
            "state": current_vector,
            "timestamp": current_time,
        }
        
        # Process through Rosa Roja Engine
        execution_plan = engine.process_event(delta_state, delta_time)
        
        # Dispatch to actuator
        if execution_plan.action in ("EXECUTE", "EMERGENCY_FLUSH"):
            await actuator.dispatch_execution(execution_plan)
        
        # Build response
        processing_time = (time.perf_counter() - start_time) * 1000
        
        envelope_response = None
        if execution_plan.envelope:
            envelope_response = ActionEnvelopeResponse(
                magnitude=execution_plan.envelope.magnitude,
                bounds=execution_plan.envelope.bounds,
                max_steps=execution_plan.envelope.max_steps,
                metadata=execution_plan.envelope.metadata,
            )
        
        response = TelemetryResponse(
            device_id=device_id,
            action=execution_plan.action,
            confidence=execution_plan.global_confidence,
            envelope=envelope_response,
            invalidation_step=execution_plan.invalidation_step,
            regime_alert=execution_plan.regime_alert,
            processing_time_ms=processing_time,
            veto_details=execution_plan.veto_details or {},
        )
        
        logger.info(
            "[ML-SERVICE] /telemetry/%s completed in %.2fms: action=%s, confidence=%.3f",
            device_id, processing_time, execution_plan.action, execution_plan.global_confidence
        )
        
        return response
        
    except ValidationError as e:
        processing_time = (time.perf_counter() - start_time) * 1000
        logger.warning(f"[ML-SERVICE] Validation error for {device_id}: {e}")
        raise HTTPException(status_code=422, detail=str(e))
        
    except HTTPException:
        raise
        
    except Exception as e:
        processing_time = (time.perf_counter() - start_time) * 1000
        logger.exception(f"[ML-SERVICE] Error processing telemetry for {device_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error processing telemetry: {str(e)}"
        )


@router.get("/telemetry/{device_id}/state")
async def get_device_state(
    device_id: str,
    _: str = Depends(verify_api_key),
) -> dict:
    """Get cached device state for debugging."""
    state = _device_state_cache.get(device_id)
    if state is None:
        return {"device_id": device_id, "state": None, "message": "No cached state"}
    return {
        "device_id": device_id,
        "cached_state": state["state"],
        "cached_timestamp": state["timestamp"],
        "dimensions": len(state["state"]),
    }


@router.delete("/telemetry/{device_id}/state")
async def clear_device_state(
    device_id: str,
    _: str = Depends(verify_api_key),
) -> dict:
    """Clear cached device state (e.g., after maintenance)."""
    if device_id in _device_state_cache:
        del _device_state_cache[device_id]
        return {"device_id": device_id, "cleared": True}
    return {"device_id": device_id, "cleared": False, "message": "No cached state to clear"}
