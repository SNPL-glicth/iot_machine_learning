"""Ensemble prediction logic — extracted from ensemble_predictor.py."""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow

logger = logging.getLogger(__name__)


def run_ensemble_prediction(
    ensemble,
    window: SensorWindow,
) -> Prediction:
    """Combina predicciones de todos los engines."""
    predictions: List[Optional[Prediction]] = []
    individual_meta: List[Dict[str, object]] = []

    for engine in ensemble._engines:
        try:
            if not engine.can_handle(window.size):
                predictions.append(None)
                individual_meta.append({
                    "engine": engine.name,
                    "status": "skipped_insufficient_data",
                })
                continue

            result = engine.predict(window)
            predictions.append(result)
            individual_meta.append({
                "engine": engine.name,
                "status": "success",
                "prediction": result.predicted_value,
                "confidence": result.confidence_score,
                "trend": result.trend,
            })
        except Exception as exc:
            logger.warning(
                "ensemble_engine_failed",
                extra={
                    "engine": engine.name,
                    "series_id": str(window.sensor_id),
                    "error": str(exc),
                },
            )
            predictions.append(None)
            individual_meta.append({
                "engine": engine.name,
                "status": "failed",
                "error": str(exc),
            })

    valid_indices = [i for i, p in enumerate(predictions) if p is not None]

    if not valid_indices:
        warnings.warn(
            "EnsembleWeightedPredictor is deprecated and will "
            "be removed. Use MetaCognitiveOrchestrator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "ensemble_all_engines_failed",
            extra={"series_id": ensemble._series_id},
        )
        return Prediction(
            series_id=str(window.sensor_id),
            predicted_value=0.0,
            confidence_score=0.0,
            trend="stable",
            engine_name="ensemble_weighted",
            metadata={
                "is_ensemble_fallback": True,
                "reason": "all_engines_failed",
            },
        )

    valid_preds = [predictions[i] for i in valid_indices]
    valid_weights = [ensemble._weights[i] for i in valid_indices]

    total_w = sum(valid_weights)
    if total_w < 1e-12:
        valid_weights = [1.0 / len(valid_weights)] * len(valid_weights)
    else:
        valid_weights = [w / total_w for w in valid_weights]

    final_value = sum(
        p.predicted_value * w  # type: ignore[union-attr]
        for p, w in zip(valid_preds, valid_weights)
    )

    final_confidence = sum(
        p.confidence_score * w  # type: ignore[union-attr]
        for p, w in zip(valid_preds, valid_weights)
    )

    trend_counts: Dict[str, float] = {"up": 0.0, "down": 0.0, "stable": 0.0}
    for p, w in zip(valid_preds, valid_weights):
        trend_counts[p.trend] += w  # type: ignore[union-attr]
    final_trend = max(trend_counts, key=trend_counts.get)  # type: ignore[arg-type]

    weight_map = {
        ensemble._engines[i].name: round(valid_weights[j], 4)
        for j, i in enumerate(valid_indices)
    }

    logger.debug(
        "ensemble_prediction",
        extra={
            "series_id": str(window.sensor_id),
            "n_engines_used": len(valid_indices),
            "weights": weight_map,
            "final_value": round(final_value, 4),
            "final_confidence": round(final_confidence, 4),
        },
    )

    return Prediction(
        series_id=str(window.sensor_id),
        predicted_value=final_value,
        confidence_score=final_confidence,
        trend=final_trend,  # type: ignore[arg-type]
        engine_name="ensemble_weighted",
        metadata={
            "ensemble_weights": weight_map,
            "individual_predictions": individual_meta,
            "n_engines_total": len(ensemble._engines),
            "n_engines_used": len(valid_indices),
        },
    )
