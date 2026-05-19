"""Telemetry endpoints — extracted from routes.py for ≤180 lines."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends

from .dependencies import DbConnDep, verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


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
