"""Ensemble weight update logic — extracted from ensemble_predictor.py."""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

_WEIGHT_UPDATE_INTERVAL: int = 10
_MIN_WEIGHT: float = 0.05


def update_ensemble_weights(
    ensemble,
    actual_value: float,
    predictions: List[Optional[object]],
) -> None:
    """Actualiza pesos según error reciente (inverse error weighting)."""
    if not ensemble._adapt_weights:
        return

    for i, pred in enumerate(predictions):
        if pred is not None:
            error = abs(pred.predicted_value - actual_value)
            ensemble._engine_errors[ensemble._engines[i].name].append(error)

    ensemble._update_count += 1

    if ensemble._update_count % _WEIGHT_UPDATE_INTERVAL == 0:
        recalculate_ensemble_weights(ensemble)


def recalculate_ensemble_weights(ensemble) -> None:
    """Recalcula pesos usando inverse error weighting."""
    avg_errors: List[float] = []

    for engine in ensemble._engines:
        errors = ensemble._engine_errors[engine.name]
        if errors:
            avg_errors.append(sum(errors) / len(errors))
        else:
            avg_errors.append(1.0)

    inverse = [1.0 / (err + 1e-9) for err in avg_errors]
    total = sum(inverse)

    new_weights = [max(_MIN_WEIGHT, inv / total) for inv in inverse]
    total_new = sum(new_weights)
    ensemble._weights = [w / total_new for w in new_weights]

    logger.info(
        "ensemble_weights_updated",
        extra={
            "avg_errors": [round(e, 4) for e in avg_errors],
            "new_weights": [round(w, 4) for w in ensemble._weights],
        },
    )

    if ensemble._weights_repo and ensemble._series_id:
        try:
            weights_dict = {
                ensemble._engines[i].name: ensemble._weights[i]
                for i in range(len(ensemble._engines))
            }
            errors_dict = {
                ensemble._engines[i].name: avg_errors[i]
                for i in range(len(ensemble._engines))
            }
            ensemble._weights_repo.save_weights(
                series_id=ensemble._series_id,
                domain_type=ensemble._domain_type,
                weights=weights_dict,
                errors=errors_dict,
            )
            logger.debug(
                "ensemble_weights_persisted",
                extra={"series_id": ensemble._series_id},
            )
        except Exception as exc:
            logger.warning(
                "ensemble_weights_persist_failed",
                extra={
                    "series_id": ensemble._series_id,
                    "error": str(exc),
                },
            )
