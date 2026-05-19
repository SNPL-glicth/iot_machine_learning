"""API routes for ML service — core ML endpoints.

Sub-routers:
- routes_health: /health, /ready
- routes_telemetry: /telemetry/...
- routes_observability: /ml/observability/...
- routes_query: query endpoints
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Depends, Header, Query, Request
from fastapi.responses import JSONResponse

from .dependencies import DbConnDep, verify_api_key
from .schemas import (
    MetacognitiveResponse,
    PredictRequest,
    PredictResponse,
    StructuralAnalysisResponse,
)
from .services import PredictionService
from application.dto.prediction_dto import PredictionDTO
from ..config.feature_flags import get_feature_flags
from .routes_health import router as health_router
from .routes_telemetry import router as telemetry_router
from .routes_observability import router as observability_router
from .routes_query import router as query_router
from .routes_predict_async import ML_ASYNC_ENABLED, _handle_async_prediction

logger = logging.getLogger(__name__)

router = APIRouter()
router.include_router(health_router)
router.include_router(telemetry_router)
router.include_router(observability_router)
router.include_router(query_router)

@router.post("/ml/predict", response_model=PredictResponse)
async def ml_predict(
    payload: PredictRequest,
    conn: DbConnDep,
    request: Request,
    x_async_prediction: Optional[str] = Header(None),
    async_param: bool = Query(False, alias="async"),
    _: str = Depends(verify_api_key),
) -> PredictResponse:
    """Generate a prediction for a sensor.

    FIX P3-3: Async mode via X-Async-Prediction: true header or ?async=true query.
    When async, returns 202 Accepted with prediction_id immediately.
    """
    is_async = (x_async_prediction == "true") or async_param
    if is_async and ML_ASYNC_ENABLED:
        return _handle_async_prediction(request, payload)

    logger.info(
        "[ML-SERVICE] /ml/predict sensor_id=%s horizon=%s window=%s dedupe=%s",
        payload.sensor_id, payload.horizon_minutes, payload.window, payload.dedupe_minutes,
    )
    try:
        # FASE-1A: Inject cognitive orchestrator when flag enabled
        cognitive_orchestrator = None
        flags = get_feature_flags()
        if flags.ML_USE_COGNITIVE_ORCHESTRATOR:
            try:
                from iot_machine_learning.ml_service.runners.wiring.container import (
                    BatchEnterpriseContainer,
                )
                from iot_ingest_services.common.db import get_engine

                container = BatchEnterpriseContainer(
                    engine=get_engine(), flags=flags,
                )
                cognitive_adapter = container.get_cognitive_adapter()
                cognitive_orchestrator = cognitive_adapter.orchestrator
                logger.info(
                    "cognitive_orchestrator_injected",
                    extra={"sensor_id": payload.sensor_id},
                )
            except Exception as exc:
                logger.warning(
                    "cognitive_orchestrator_init_failed",
                    extra={
                        "error": str(exc),
                        "fallback": "baseline+kalman",
                        "sensor_id": payload.sensor_id,
                    },
                )

        service = PredictionService(
            conn, cognitive_orchestrator=cognitive_orchestrator,
        )
        result = service.predict(
            sensor_id=payload.sensor_id,
            horizon_minutes=payload.horizon_minutes,
            window=payload.window,
            dedupe_minutes=payload.dedupe_minutes,
        )

        sa_data = result.get("structural_analysis")
        structural_analysis = StructuralAnalysisResponse(**sa_data) if sa_data else None
        mc_data = result.get("metacognitive")
        metacognitive = MetacognitiveResponse(**mc_data) if mc_data else None

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
        return {"error": str(e), "connected": False}

@router.get("/ml/metrics")
async def ml_metrics(
    request: Request,
    _: str = Depends(verify_api_key),
    format: str = "",
) -> dict:
    """Get ML service performance metrics.

    FIX P2-2: Retorna Prometheus exposition text si ?format=prometheus
    o si Accept header contiene 'text/plain'. Retrocompatible: sin param → JSON.
    """
    try:
        accept = request.headers.get("accept", "")
        want_prometheus = format == "prometheus" or "text/plain" in accept

        if want_prometheus:
            from ..metrics.performance_metrics import MetricsCollector
            from fastapi.responses import PlainTextResponse
            collector = MetricsCollector.get_instance()
            body = collector.to_prometheus_text()
            return PlainTextResponse(
                content=body,
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

        from ..metrics import get_metrics
        metrics = get_metrics()
        return metrics.to_dict()

    except Exception as e:
        logger.exception("[ML-SERVICE] Metrics error: %s", str(e))
        return {"error": str(e)}

@router.get("/ml/diagnostics")
async def ml_diagnostics(_: str = Depends(verify_api_key)) -> dict:
    """FIX PROD-2: Cognitive pipeline diagnostics."""
    from ..metrics.performance_metrics import MetricsCollector
    collector = MetricsCollector.get_instance()
    with collector._update_lock:
        times = list(collector._cognitive_phase_times)
        budget_exceeded = collector._cognitive_budget_exceeded
        skipped = collector._cognitive_phases_skipped
        fallbacks = collector._cognitive_fallbacks
    slowest = {}
    for phase, dur in times:
        slowest[phase] = max(slowest.get(phase, 0), dur)
    top_slow = sorted(slowest.items(), key=lambda x: x[1], reverse=True)[:5]
    return {
        "slowest_phases": [{"phase": p, "max_ms": round(d, 2)} for p, d in top_slow],
        "cognitive_stats": {
            "budget_exceeded": budget_exceeded,
            "phases_skipped": skipped,
            "fallbacks": fallbacks,
            "total_recorded": len(times),
        },
    }

@router.get("/ml/predict/{prediction_id}/status")
async def ml_predict_status(
    prediction_id: str,
    request: Request,
    _: str = Depends(verify_api_key),
) -> dict:
    """FIX P3-3: Check status of an async prediction."""
    result_store = getattr(request.app.state, "result_store", None)
    if result_store is None:
        raise HTTPException(status_code=503, detail="Result store not available")
    result = result_store.get(prediction_id)
    _MISSING = result_store.get_missing_sentinel()
    if result is _MISSING:
        raise HTTPException(status_code=404, detail="Prediction not found or expired")
    if result is None:
        return JSONResponse(status_code=202, content={"status": "pending"})
    return {"status": "completed", "result": result}
