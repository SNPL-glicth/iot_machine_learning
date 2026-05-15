"""SensorFeedbackService — orchestrates predict-and-verify loop.

Always VERIFY first, then PREDICT, then REGISTER.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from iot_machine_learning.infrastructure.ml.cognitive.prediction_cache.pending_prediction_cache import (
        PendingPredictionCache,
    )
    from iot_machine_learning.infrastructure.ml.interfaces import (
        PredictionEngine,
        PredictionResult,
    )
    from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.prediction_verification_repository import (
        PredictionVerificationRepository,
    )

logger = logging.getLogger(__name__)


class SensorFeedbackService:
    """Orchestrates the predict-and-verify loop for sensor readings."""

    def __init__(
        self,
        orchestrator: Optional["PredictionEngine"],
        verification_repo: "PredictionVerificationRepository",
        pending_cache: "PendingPredictionCache",
        horizon_seconds: int,
        tolerance_seconds: Optional[int] = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._repo = verification_repo
        self._cache = pending_cache
        self._horizon_seconds = horizon_seconds
        self._tolerance_seconds = max(
            tolerance_seconds or int(horizon_seconds * 0.1),
            30,
        )

    def verify(
        self,
        series_id: str,
        reading_value: float,
        reading_timestamp: float,
    ) -> bool:
        """Verify if this reading matches a pending prediction.

        Returns True if matched and verified, False otherwise.
        """
        reading_dt = datetime.fromtimestamp(reading_timestamp, tz=timezone.utc)

        # 1. Try Redis cache first for fast matching
        match_id = self._cache.find_match(
            series_id=series_id,
            reading_timestamp_epoch=reading_timestamp,
            tolerance_seconds=float(self._tolerance_seconds),
        )

        if match_id is None:
            # Fallback to SQL if Redis miss (cache may have expired)
            match = self._repo.find_match(
                series_id=series_id,
                reading_timestamp=reading_dt,
                tolerance_seconds=self._tolerance_seconds,
            )
            if match is None:
                logger.debug(
                    "no_match_for_actual",
                    extra={"series_id": series_id, "timestamp": reading_timestamp},
                )
                return False
            match_id = match["prediction_id"]
            predicted_value = match["predicted_value"]
        else:
            # We need predicted_value from SQL for error calculation
            match = self._repo.find_match(
                series_id=series_id,
                reading_timestamp=reading_dt,
                tolerance_seconds=self._tolerance_seconds,
            )
            if match is None:
                logger.warning(
                    "cache_match_sql_miss",
                    extra={"prediction_id": match_id, "series_id": series_id},
                )
                self._cache.remove(series_id, match_id)
                return False
            predicted_value = match["predicted_value"]

        # 2. Calculate error against the PREDICTED VALUE, not last prediction
        absolute_error = abs(predicted_value - reading_value)

        # 3. Mark as verified in SQL
        self._repo.mark_verified(match_id, reading_value, absolute_error)

        # 4. Remove from Redis cache
        self._cache.remove(series_id, match_id)

        # 5. Feed back to orchestrator if available (cognitive mode)
        if self._orchestrator is not None:
            self._orchestrator.record_actual(
                actual_value=reading_value,
                series_id=series_id,
            )

        logger.debug(
            "prediction_verified",
            extra={
                "prediction_id": match_id,
                "series_id": series_id,
                "predicted": predicted_value,
                "actual": reading_value,
                "error": absolute_error,
            },
        )
        return True

    def predict_and_register(
        self,
        series_id: str,
        values: list[float],
        timestamps: Optional[list[float]],
        reading_timestamp: float,
    ) -> Optional["PredictionResult"]:
        """Predict and register the prediction as pending.

        Returns None if cooldown is active (pending prediction exists).
        """
        now_epoch = reading_timestamp

        # 1. Cooldown check
        if self._cache.has_pending(series_id, now_epoch):
            logger.debug(
                "cooldown_active",
                extra={"series_id": series_id, "horizon": self._horizon_seconds},
            )
            return None

        # 2. Predict (requires orchestrator)
        if self._orchestrator is None:
            raise RuntimeError(
                "predict_and_register requires an orchestrator. "
                "Pass orchestrator=... to SensorFeedbackService."
            )
        result = self._orchestrator.predict(
            series_id=series_id,
            values=values,
            timestamps=timestamps,
        )

        # 3. Calculate target timestamp
        target_ts_epoch = reading_timestamp + self._horizon_seconds
        target_dt = datetime.fromtimestamp(target_ts_epoch, tz=timezone.utc)

        # 4. Extract engine name from metadata
        engine_name = result.metadata.get("selected_engine", "ensemble") if result.metadata else "ensemble"
        if engine_name == "ensemble":
            # Try individual engine name if ensemble not available
            engine_name = result.metadata.get("engine_name", "ensemble") if result.metadata else "ensemble"

        # 5. Save to SQL
        prediction_id = self._repo.save_pending(
            series_id=series_id,
            predicted_value=result.predicted_value,
            target_timestamp=target_dt,
            horizon_seconds=self._horizon_seconds,
            engine_name=str(engine_name),
            confidence=result.confidence,
        )

        # 6. Register in Redis cache
        self._cache.register(series_id, prediction_id, target_ts_epoch)

        logger.debug(
            "prediction_registered",
            extra={
                "prediction_id": prediction_id,
                "series_id": series_id,
                "target_timestamp": target_dt.isoformat(),
                "engine": engine_name,
            },
        )
        return result
