"""Repository for ensemble adaptive weights."""

from __future__ import annotations

import logging
from typing import Dict, Optional
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
    ZeninDbConnection,
)

logger = logging.getLogger(__name__)


class EnsembleWeightsRepository:
    """Persists and loads ensemble weights to/from zenin_ml.ensemble_weights."""

    def __init__(self, engine: Optional[Engine] = None):
        self._engine = engine or ZeninDbConnection.get_engine()

    def save_weights(
        self,
        series_id: str,
        domain_type: str,
        weights: Dict[str, float],
        errors: Optional[Dict[str, float]] = None,
    ) -> None:
        """Save or update ensemble weights for a series.

        Args:
            series_id: Series identifier
            domain_type: Domain type ('sensor', 'text', etc.)
            weights: Dict of {engine_name: weight}
            errors: Optional dict of {engine_name: avg_error}
        """
        try:
            with self._engine.begin() as conn:
                for engine_name, weight in weights.items():
                    avg_error = errors.get(engine_name) if errors else None

                    # MERGE (upsert)
                    conn.execute(
                        text("""
                            MERGE zenin_ml.ensemble_weights AS target
                            USING (SELECT :series_id AS SeriesId, :engine_name AS EngineName) AS source
                            ON target.SeriesId = source.SeriesId 
                               AND target.EngineName = source.EngineName
                            WHEN MATCHED THEN
                                UPDATE SET
                                    Weight = :weight,
                                    AvgError = :avg_error,
                                    ErrorCount = ErrorCount + 1,
                                    LastError = :avg_error,
                                    UpdatedAt = GETUTCDATE()
                            WHEN NOT MATCHED THEN
                                INSERT (Id, SeriesId, EngineName, DomainType, Weight, AvgError, ErrorCount, LastError)
                                VALUES (NEWID(), :series_id, :engine_name, :domain_type, :weight, :avg_error, 1, :avg_error);
                        """),
                        {
                            "series_id": series_id,
                            "engine_name": engine_name,
                            "domain_type": domain_type,
                            "weight": weight,
                            "avg_error": avg_error,
                        },
                    )

            logger.debug(
                "ensemble_weights_saved",
                extra={
                    "series_id": series_id,
                    "domain_type": domain_type,
                    "n_engines": len(weights),
                },
            )

        except Exception as exc:
            logger.error(
                "ensemble_weights_save_failed",
                extra={
                    "series_id": series_id,
                    "error": str(exc),
                },
            )
            raise

    def load_weights(
        self,
        series_id: str,
    ) -> Optional[Dict[str, float]]:
        """Load ensemble weights for a series.

        Args:
            series_id: Series identifier

        Returns:
            Dict of {engine_name: weight} or None if not found
        """
        try:
            with self._engine.begin() as conn:
                results = conn.execute(
                    text("""
                        SELECT EngineName, Weight
                        FROM zenin_ml.ensemble_weights
                        WHERE SeriesId = :series_id
                        ORDER BY UpdatedAt DESC
                    """),
                    {"series_id": series_id},
                ).fetchall()

            if not results:
                return None

            weights = {row[0]: row[1] for row in results}

            logger.debug(
                "ensemble_weights_loaded",
                extra={
                    "series_id": series_id,
                    "n_engines": len(weights),
                },
            )

            return weights

        except Exception as exc:
            logger.error(
                "ensemble_weights_load_failed",
                extra={
                    "series_id": series_id,
                    "error": str(exc),
                },
            )
            return None
