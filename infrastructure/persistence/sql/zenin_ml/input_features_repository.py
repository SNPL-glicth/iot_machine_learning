"""Repository for prediction input features (reproducibility)."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
    ZeninDbConnection,
)

logger = logging.getLogger(__name__)


class InputFeaturesRepository:
    """Persists input features for prediction reproducibility."""

    def __init__(self, engine: Optional[Engine] = None):
        self._engine = engine or ZeninDbConnection.get_engine()

    def save_input_features(
        self,
        prediction_id: UUID,
        input_values: List[float],
        input_timestamps: Optional[List[float]] = None,
        computed_features: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """Save input features for a prediction.

        Args:
            prediction_id: UUID of the prediction
            input_values: List of input values
            input_timestamps: Optional list of timestamps
            computed_features: Optional computed features dict

        Returns:
            UUID of saved input features record
        """
        try:
            # Compute hash for deduplication
            input_hash = self._compute_hash(input_values, input_timestamps)

            # Check if already exists
            existing_id = self._find_by_hash(input_hash)
            if existing_id:
                logger.debug(
                    "input_features_already_exist",
                    extra={"input_hash": input_hash, "existing_id": str(existing_id)},
                )
                return existing_id

            # Insert new record
            features_id = uuid4()

            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO zenin_ml.prediction_inputs (
                            Id, PredictionId,
                            InputValues, InputTimestamps, WindowSize,
                            InputHash, ComputedFeatures
                        ) VALUES (
                            :id, :prediction_id,
                            :input_values, :input_timestamps, :window_size,
                            :input_hash, :computed_features
                        )
                    """),
                    {
                        "id": str(features_id),
                        "prediction_id": str(prediction_id),
                        "input_values": json.dumps(input_values),
                        "input_timestamps": json.dumps(input_timestamps) if input_timestamps else None,
                        "window_size": len(input_values),
                        "input_hash": input_hash,
                        "computed_features": json.dumps(computed_features) if computed_features else None,
                    },
                )

            logger.debug(
                "input_features_saved",
                extra={
                    "features_id": str(features_id),
                    "prediction_id": str(prediction_id),
                    "window_size": len(input_values),
                    "input_hash": input_hash,
                },
            )

            return features_id

        except Exception as exc:
            logger.error(
                "input_features_save_failed",
                extra={
                    "prediction_id": str(prediction_id),
                    "error": str(exc),
                },
            )
            raise

    def load_input_features(
        self,
        prediction_id: UUID,
    ) -> Optional[Dict[str, Any]]:
        """Load input features for a prediction.

        Args:
            prediction_id: UUID of the prediction

        Returns:
            Dict with input_values, input_timestamps, computed_features or None
        """
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text("""
                        SELECT InputValues, InputTimestamps, ComputedFeatures
                        FROM zenin_ml.prediction_inputs
                        WHERE PredictionId = :prediction_id
                    """),
                    {"prediction_id": str(prediction_id)},
                ).fetchone()

            if result is None:
                return None

            return {
                "input_values": json.loads(result[0]) if result[0] else [],
                "input_timestamps": json.loads(result[1]) if result[1] else None,
                "computed_features": json.loads(result[2]) if result[2] else None,
            }

        except Exception as exc:
            logger.error(
                "input_features_load_failed",
                extra={
                    "prediction_id": str(prediction_id),
                    "error": str(exc),
                },
            )
            return None

    def _compute_hash(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> str:
        """Compute SHA256 hash of input data."""
        data = json.dumps({"values": values, "timestamps": timestamps}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def _find_by_hash(self, input_hash: str) -> Optional[UUID]:
        """Find existing record by hash."""
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text("""
                        SELECT TOP 1 Id
                        FROM zenin_ml.prediction_inputs
                        WHERE InputHash = :input_hash
                    """),
                    {"input_hash": input_hash},
                ).fetchone()

            return UUID(result[0]) if result else None

        except Exception:
            return None
