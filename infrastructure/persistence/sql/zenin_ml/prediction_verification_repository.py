"""Repository for prediction verifications — tracking pending vs verified predictions."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
    ZeninDbConnection,
)

logger = logging.getLogger(__name__)


class PredictionVerificationRepository:
    """Persists and queries prediction verifications for predict-and-verify loop."""

    def __init__(self, engine: Optional[Engine] = None) -> None:
        self._engine = engine or ZeninDbConnection.get_engine()
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create table if not exists."""
        try:
            with self._engine.begin() as conn:
                conn.execute(text("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables
                        WHERE name = 'prediction_verifications'
                        AND schema_id = SCHEMA_ID('dbo')
                    )
                    BEGIN
                        CREATE TABLE dbo.prediction_verifications (
                            prediction_id VARCHAR(36) NOT NULL PRIMARY KEY,
                            series_id VARCHAR(100) NOT NULL,
                            predicted_value FLOAT NOT NULL,
                            actual_value FLOAT NULL,
                            absolute_error FLOAT NULL,
                            target_timestamp DATETIME2 NOT NULL,
                            horizon_seconds INT NOT NULL,
                            engine_name VARCHAR(50) NOT NULL,
                            confidence FLOAT NOT NULL,
                            status VARCHAR(20) NOT NULL DEFAULT 'pending',
                            created_at DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
                            verified_at DATETIME2 NULL
                        );
                        CREATE INDEX idx_series_target
                            ON dbo.prediction_verifications (series_id, target_timestamp);
                        CREATE INDEX idx_status
                            ON dbo.prediction_verifications (status);
                    END
                """))
        except Exception as exc:
            logger.warning(
                "prediction_verifications_table_creation_failed",
                extra={"error": str(exc)},
            )

    def save_pending(
        self,
        series_id: str,
        predicted_value: float,
        target_timestamp: datetime,
        horizon_seconds: int,
        engine_name: str,
        confidence: float,
    ) -> str:
        """Save a new pending prediction and return its prediction_id."""
        prediction_id = str(uuid4())
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO dbo.prediction_verifications (
                            prediction_id, series_id, predicted_value,
                            target_timestamp, horizon_seconds, engine_name,
                            confidence, status, created_at
                        )
                        VALUES (
                            :prediction_id, :series_id, :predicted_value,
                            :target_timestamp, :horizon_seconds, :engine_name,
                            :confidence, 'pending', GETUTCDATE()
                        )
                    """),
                    {
                        "prediction_id": prediction_id,
                        "series_id": series_id,
                        "predicted_value": predicted_value,
                        "target_timestamp": target_timestamp.replace(tzinfo=None),
                        "horizon_seconds": horizon_seconds,
                        "engine_name": engine_name,
                        "confidence": confidence,
                    },
                )
            logger.debug(
                "prediction_pending_saved",
                extra={
                    "prediction_id": prediction_id,
                    "series_id": series_id,
                    "target_timestamp": target_timestamp.isoformat(),
                },
            )
            return prediction_id
        except Exception as exc:
            logger.warning(
                "prediction_pending_save_failed",
                extra={"series_id": series_id, "error": str(exc)},
            )
            raise

    def find_match(
        self,
        series_id: str,
        reading_timestamp: datetime,
        tolerance_seconds: int,
    ) -> Optional[Dict[str, object]]:
        """Find the closest pending prediction matching the reading timestamp."""
        try:
            with self._engine.begin() as conn:
                row = conn.execute(
                    text("""
                        SELECT TOP 1
                            prediction_id, predicted_value, target_timestamp,
                            horizon_seconds, engine_name, confidence
                        FROM dbo.prediction_verifications
                        WHERE series_id = :series_id
                          AND status = 'pending'
                          AND ABS(DATEDIFF(second, target_timestamp, :reading_ts))
                              <= :tolerance_seconds
                        ORDER BY ABS(DATEDIFF(second, target_timestamp, :reading_ts))
                    """),
                    {
                        "series_id": series_id,
                        "reading_ts": reading_timestamp.replace(tzinfo=None),
                        "tolerance_seconds": tolerance_seconds,
                    },
                ).fetchone()

                if row is None:
                    return None

                return {
                    "prediction_id": str(row[0]),
                    "predicted_value": float(row[1]),
                    "target_timestamp": row[2],
                    "horizon_seconds": int(row[3]),
                    "engine_name": str(row[4]),
                    "confidence": float(row[5]),
                }
        except Exception as exc:
            logger.warning(
                "prediction_match_query_failed",
                extra={"series_id": series_id, "error": str(exc)},
            )
            return None

    def mark_verified(
        self,
        prediction_id: str,
        actual_value: float,
        absolute_error: float,
    ) -> None:
        """Mark a pending prediction as verified with actual value and error."""
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        UPDATE dbo.prediction_verifications
                        SET
                            status = 'verified',
                            actual_value = :actual_value,
                            absolute_error = :absolute_error,
                            verified_at = GETUTCDATE()
                        WHERE prediction_id = :prediction_id
                    """),
                    {
                        "prediction_id": prediction_id,
                        "actual_value": actual_value,
                        "absolute_error": absolute_error,
                    },
                )
            logger.debug(
                "prediction_verified",
                extra={
                    "prediction_id": prediction_id,
                    "actual_value": actual_value,
                    "absolute_error": absolute_error,
                },
            )
        except Exception as exc:
            logger.warning(
                "prediction_verify_failed",
                extra={"prediction_id": prediction_id, "error": str(exc)},
            )

    def expire_old(self, max_age_seconds: int) -> int:
        """Mark stale pending predictions as expired. Returns count expired."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text("""
                        UPDATE dbo.prediction_verifications
                        SET status = 'expired'
                        WHERE status = 'pending'
                          AND target_timestamp < :cutoff
                    """),
                    {"cutoff": cutoff.replace(tzinfo=None)},
                )
                expired_count = result.rowcount if hasattr(result, "rowcount") else 0
                if expired_count:
                    logger.info(
                        "predictions_expired",
                        extra={"count": expired_count, "cutoff": cutoff.isoformat()},
                    )
                return expired_count
        except Exception as exc:
            logger.warning(
                "prediction_expire_failed",
                extra={"error": str(exc)},
            )
            return 0
