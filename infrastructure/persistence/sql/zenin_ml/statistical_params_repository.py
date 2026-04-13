"""Repository for statistical engine parameters."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
    ZeninDbConnection,
)

logger = logging.getLogger(__name__)


class StatisticalParamsRepository:
    """Persists and loads statistical engine parameters."""
    
    def __init__(self, engine: Optional[Engine] = None):
        self._engine = engine or ZeninDbConnection.get_engine()
        self._ensure_table()
    
    def _ensure_table(self) -> None:
        """Create table if not exists."""
        try:
            with self._engine.begin() as conn:
                conn.execute(text("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables 
                        WHERE name = 'statistical_params' 
                        AND schema_id = SCHEMA_ID('zenin_ml')
                    )
                    BEGIN
                        CREATE TABLE zenin_ml.statistical_params (
                            series_id VARCHAR(100) NOT NULL PRIMARY KEY,
                            alpha FLOAT NOT NULL DEFAULT 0.3,
                            beta FLOAT NOT NULL DEFAULT 0.1,
                            mae FLOAT NOT NULL DEFAULT 999.0,
                            n_samples INT NOT NULL DEFAULT 0,
                            updated_at DATETIME DEFAULT GETDATE()
                        )
                    END
                """))
        except Exception as exc:
            logger.warning(
                "statistical_params_table_creation_failed",
                extra={"error": str(exc)},
            )
    
    def load_params(
        self,
        series_id: str,
    ) -> Optional[Tuple[float, float, float]]:
        """Load parameters for a series.
        
        Args:
            series_id: Series identifier
            
        Returns:
            Tuple of (alpha, beta, mae) or None if not found
        """
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text("""
                        SELECT alpha, beta, mae
                        FROM zenin_ml.statistical_params
                        WHERE series_id = :series_id
                    """),
                    {"series_id": series_id},
                ).fetchone()
            
            if result:
                logger.debug(
                    "statistical_params_loaded",
                    extra={
                        "series_id": series_id,
                        "alpha": result[0],
                        "beta": result[1],
                        "mae": result[2],
                    },
                )
                return result[0], result[1], result[2]
            
            return None
            
        except Exception as exc:
            logger.warning(
                "statistical_params_load_failed",
                extra={"series_id": series_id, "error": str(exc)},
            )
            return None
    
    def save_params(
        self,
        series_id: str,
        alpha: float,
        beta: float,
        mae: float,
        n_samples: int,
    ) -> None:
        """Save or update parameters for a series.
        
        Args:
            series_id: Series identifier
            alpha: EMA smoothing factor
            beta: Trend smoothing factor
            mae: Mean absolute error
            n_samples: Number of samples used
        """
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        MERGE zenin_ml.statistical_params AS target
                        USING (SELECT :series_id AS series_id) AS source
                        ON target.series_id = source.series_id
                        WHEN MATCHED THEN
                            UPDATE SET
                                alpha = :alpha,
                                beta = :beta,
                                mae = :mae,
                                n_samples = :n_samples,
                                updated_at = GETDATE()
                        WHEN NOT MATCHED THEN
                            INSERT (series_id, alpha, beta, mae, n_samples)
                            VALUES (:series_id, :alpha, :beta, :mae, :n_samples);
                    """),
                    {
                        "series_id": series_id,
                        "alpha": alpha,
                        "beta": beta,
                        "mae": mae,
                        "n_samples": n_samples,
                    },
                )
            
            logger.debug(
                "statistical_params_saved",
                extra={
                    "series_id": series_id,
                    "alpha": alpha,
                    "beta": beta,
                    "mae": round(mae, 4),
                },
            )
            
        except Exception as exc:
            logger.warning(
                "statistical_params_save_failed",
                extra={
                    "series_id": series_id,
                    "error": str(exc),
                },
            )
