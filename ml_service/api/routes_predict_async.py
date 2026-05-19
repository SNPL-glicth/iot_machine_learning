"""FIX P3-3: Async prediction helpers extracted from routes.py to keep it ≤180 lines."""
from __future__ import annotations

import logging
import os
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .schemas import PredictRequest

logger = logging.getLogger(__name__)

ML_ASYNC_ENABLED = os.environ.get("ML_ASYNC_PREDICTIONS_ENABLED", "true").lower() in ("1", "true", "yes")


def _handle_async_prediction(request: Request, payload: PredictRequest) -> JSONResponse:
    """Enqueue prediction and return 202 with prediction_id."""
    prediction_worker = getattr(request.app.state, "prediction_worker", None)
    result_store = getattr(request.app.state, "result_store", None)
    if prediction_worker is None or result_store is None:
        raise HTTPException(status_code=503, detail="Async prediction not available")
    prediction_id = str(uuid.uuid4())
    result_store.set(prediction_id, None)  # mark as pending
    from ..consumers.prediction_worker import PredictionTask
    enqueued = prediction_worker.enqueue(
        PredictionTask(
            sensor_id=payload.sensor_id,
            prediction_id=prediction_id,
            mode="http",
            horizon_minutes=payload.horizon_minutes,
            window=payload.window,
            dedupe_minutes=payload.dedupe_minutes,
        )
    )
    if not enqueued:
        raise HTTPException(status_code=503, detail="Prediction queue full")
    logger.info(
        "[ML-SERVICE] async_prediction_enqueued prediction_id=%s sensor_id=%s",
        prediction_id, payload.sensor_id,
    )
    return JSONResponse(
        status_code=202,
        content={"prediction_id": prediction_id, "status": "pending"},
    )
