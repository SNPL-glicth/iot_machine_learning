"""Batch prediction API endpoint.

Accepts multiple sensor predictions in a single request for higher throughput.
Uses concurrent execution with configurable parallelism.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .dependencies import DbConnDep, verify_api_key
from .schemas import PredictRequest, PredictResponse
from ..config.feature_flags import get_feature_flags

logger = logging.getLogger(__name__)

router = APIRouter(tags=["batch"])


# ─── Schemas ─────────────────────────────────────────────────────────────────

class BatchPredictRequest(BaseModel):
    """Batch prediction request — multiple sensors in one call."""
    predictions: List[PredictRequest] = Field(
        ..., min_length=1, max_length=100,
        description="List of prediction requests (max 100 per batch)",
    )
    max_concurrency: int = Field(
        10, gt=0, le=50,
        description="Max concurrent predictions within this batch",
    )


class BatchPredictItemResult(BaseModel):
    """Result for a single item in batch prediction."""
    sensor_id: int
    success: bool
    result: Optional[PredictResponse] = None
    error: Optional[str] = None
    elapsed_ms: Optional[float] = None


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    total: int = Field(description="Total requests in batch")
    succeeded: int = Field(description="Number of successful predictions")
    failed: int = Field(description="Number of failed predictions")
    results: List[BatchPredictItemResult] = Field(description="Individual results")
    total_elapsed_ms: float = Field(description="Total batch processing time in ms")


# ─── Endpoint ────────────────────────────────────────────────────────────────

@router.post("/ml/predict/batch", response_model=BatchPredictResponse)
async def ml_predict_batch(
    payload: BatchPredictRequest,
    conn: DbConnDep,
    request: Request,
    _: str = Depends(verify_api_key),
) -> BatchPredictResponse:
    """Generate predictions for multiple sensors in a single request.

    Features:
    - Concurrent execution with configurable parallelism
    - Individual error isolation (one failure doesn't affect others)
    - Circuit breaker aware (fails fast if downstream is unhealthy)
    """
    t_start = time.monotonic()
    flags = get_feature_flags()

    logger.info(
        "batch_predict_start",
        extra={
            "batch_size": len(payload.predictions),
            "max_concurrency": payload.max_concurrency,
        },
    )

    # Build cognitive orchestrator once for the whole batch
    cognitive_orchestrator = None
    if flags.ML_USE_COGNITIVE_ORCHESTRATOR:
        try:
            from iot_machine_learning.ml_service.runners.wiring.container import (
                BatchEnterpriseContainer,
            )
            from iot_ingest_services.common.db import get_engine

            container = BatchEnterpriseContainer(engine=get_engine(), flags=flags)
            cognitive_adapter = container.get_cognitive_adapter()
            cognitive_orchestrator = cognitive_adapter.orchestrator
        except Exception as exc:
            logger.warning(
                "batch_cognitive_orchestrator_init_failed",
                extra={"error": str(exc)},
            )

    from .services import PredictionService

    service = PredictionService(conn, cognitive_orchestrator=cognitive_orchestrator)

    # Execute predictions concurrently with semaphore
    semaphore = asyncio.Semaphore(payload.max_concurrency)

    async def _predict_one(req: PredictRequest) -> BatchPredictItemResult:
        async with semaphore:
            t0 = time.monotonic()
            try:
                result = await asyncio.to_thread(
                    service.predict,
                    sensor_id=req.sensor_id,
                    horizon_minutes=req.horizon_minutes,
                    window=req.window,
                    dedupe_minutes=req.dedupe_minutes,
                )
                elapsed = round((time.monotonic() - t0) * 1000.0, 2)

                from .schemas import StructuralAnalysisResponse, MetacognitiveResponse
                from datetime import datetime, timezone, timedelta

                sa_data = result.get("structural_analysis")
                structural_analysis = StructuralAnalysisResponse(**sa_data) if sa_data else None
                mc_data = result.get("metacognitive")
                metacognitive = MetacognitiveResponse(**mc_data) if mc_data else None

                response = PredictResponse(
                    sensor_id=result["sensor_id"],
                    model_id=result["model_id"],
                    prediction_id=result["prediction_id"],
                    predicted_value=result["predicted_value"],
                    confidence=result["confidence"],
                    target_timestamp=result["target_timestamp"],
                    horizon_minutes=result["horizon_minutes"],
                    window=result["window"],
                    trend=result.get("trend"),
                    engine_used=result.get("engine_used"),
                    confidence_level=result.get("confidence_level"),
                    structural_analysis=structural_analysis,
                    metacognitive=metacognitive,
                    audit_trace_id=result.get("audit_trace_id"),
                    processing_time_ms=result.get("processing_time_ms"),
                    explanation_summary=result.get("explanation_summary"),
                )
                return BatchPredictItemResult(
                    sensor_id=req.sensor_id,
                    success=True,
                    result=response,
                    elapsed_ms=elapsed,
                )
            except Exception as exc:
                elapsed = round((time.monotonic() - t0) * 1000.0, 2)
                logger.warning(
                    "batch_predict_item_failed",
                    extra={"sensor_id": req.sensor_id, "error": str(exc)},
                )
                return BatchPredictItemResult(
                    sensor_id=req.sensor_id,
                    success=False,
                    error=str(exc),
                    elapsed_ms=elapsed,
                )

    # Run all predictions concurrently
    results = await asyncio.gather(
        *[_predict_one(req) for req in payload.predictions]
    )

    total_elapsed = round((time.monotonic() - t_start) * 1000.0, 2)
    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded

    logger.info(
        "batch_predict_complete",
        extra={
            "total": len(results),
            "succeeded": succeeded,
            "failed": failed,
            "total_elapsed_ms": total_elapsed,
        },
    )

    return BatchPredictResponse(
        total=len(results),
        succeeded=succeeded,
        failed=failed,
        results=results,
        total_elapsed_ms=total_elapsed,
    )
