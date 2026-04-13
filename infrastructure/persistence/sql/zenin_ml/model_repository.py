"""Repository for serialized ML models (IsolationForest, LOF, etc.)."""

from __future__ import annotations

import hashlib
import logging
import pickle
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
    ZeninDbConnection,
)

logger = logging.getLogger(__name__)


class ModelRepository:
    """Persists and loads trained sklearn models to/from zenin_ml.ml_models."""

    def __init__(self, engine: Optional[Engine] = None):
        self._engine = engine or ZeninDbConnection.get_engine()

    def save_model(
        self,
        model_name: str,
        series_id: str,
        domain_type: str,
        model_obj: Any,
        training_points: int,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_metrics: Optional[Dict[str, Any]] = None,
        training_duration_ms: Optional[int] = None,
    ) -> UUID:
        """Serialize and save a trained model.

        Args:
            model_name: Name of model ('isolation_forest', 'lof', etc.)
            series_id: Series identifier
            domain_type: Domain ('sensor', 'text', 'financial', etc.)
            model_obj: Trained sklearn model object
            training_points: Number of points used for training
            hyperparameters: Model hyperparameters as dict
            training_metrics: Training metrics as dict
            training_duration_ms: Training duration in milliseconds

        Returns:
            UUID of saved model record
        """
        try:
            # Serialize model
            model_blob = pickle.dumps(model_obj)

            # Deactivate previous versions
            self._deactivate_previous_versions(series_id, model_name)

            # Insert new model
            model_id = uuid4()
            import json

            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO zenin_ml.ml_models (
                            Id, ModelName, SeriesId, DomainType,
                            ModelBlob, ModelFormat,
                            TrainedAt, TrainingPoints, TrainingDuration,
                            Hyperparameters, TrainingMetrics,
                            Version, IsActive
                        ) VALUES (
                            :id, :model_name, :series_id, :domain_type,
                            :model_blob, 'pickle',
                            GETUTCDATE(), :training_points, :training_duration,
                            :hyperparameters, :training_metrics,
                            1, 1
                        )
                    """),
                    {
                        "id": str(model_id),
                        "model_name": model_name,
                        "series_id": series_id,
                        "domain_type": domain_type,
                        "model_blob": model_blob,
                        "training_points": training_points,
                        "training_duration": training_duration_ms,
                        "hyperparameters": json.dumps(hyperparameters) if hyperparameters else None,
                        "training_metrics": json.dumps(training_metrics) if training_metrics else None,
                    },
                )

            logger.info(
                "model_saved",
                extra={
                    "model_id": str(model_id),
                    "model_name": model_name,
                    "series_id": series_id,
                    "domain_type": domain_type,
                    "training_points": training_points,
                    "blob_size_kb": len(model_blob) // 1024,
                },
            )

            return model_id

        except Exception as exc:
            logger.error(
                "model_save_failed",
                extra={
                    "model_name": model_name,
                    "series_id": series_id,
                    "error": str(exc),
                },
            )
            raise

    def load_model(
        self,
        series_id: str,
        model_name: str,
    ) -> Optional[Any]:
        """Load the active trained model for a series.

        Args:
            series_id: Series identifier
            model_name: Model name

        Returns:
            Deserialized model object or None if not found
        """
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text("""
                        SELECT TOP 1 ModelBlob, TrainedAt
                        FROM zenin_ml.ml_models
                        WHERE SeriesId = :series_id
                          AND ModelName = :model_name
                          AND IsActive = 1
                        ORDER BY TrainedAt DESC
                    """),
                    {"series_id": series_id, "model_name": model_name},
                ).fetchone()

            if result is None:
                return None

            model_blob = result[0]
            model_obj = pickle.loads(model_blob)

            logger.debug(
                "model_loaded",
                extra={
                    "series_id": series_id,
                    "model_name": model_name,
                    "trained_at": result[1],
                },
            )

            return model_obj

        except Exception as exc:
            logger.error(
                "model_load_failed",
                extra={
                    "series_id": series_id,
                    "model_name": model_name,
                    "error": str(exc),
                },
            )
            return None

    def _deactivate_previous_versions(self, series_id: str, model_name: str) -> None:
        """Mark previous versions as inactive."""
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        UPDATE zenin_ml.ml_models
                        SET IsActive = 0, UpdatedAt = GETUTCDATE()
                        WHERE SeriesId = :series_id
                          AND ModelName = :model_name
                          AND IsActive = 1
                    """),
                    {"series_id": series_id, "model_name": model_name},
                )
        except Exception as exc:
            logger.warning(
                "deactivate_previous_versions_failed",
                extra={"series_id": series_id, "model_name": model_name, "error": str(exc)},
            )
