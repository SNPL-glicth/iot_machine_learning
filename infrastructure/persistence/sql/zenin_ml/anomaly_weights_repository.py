"""Repository for anomaly detector adaptive weights."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
    ZeninDbConnection,
)

logger = logging.getLogger(__name__)


class AnomalyWeightsRepository:
    """Persists and loads anomaly detector adaptive weights."""
    
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
                        WHERE name = 'anomaly_detector_weights' 
                        AND schema_id = SCHEMA_ID('zenin_ml')
                    )
                    BEGIN
                        CREATE TABLE zenin_ml.anomaly_detector_weights (
                            series_id VARCHAR(100) NOT NULL,
                            detector_name VARCHAR(50) NOT NULL,
                            weight FLOAT NOT NULL,
                            precision_score FLOAT NOT NULL DEFAULT 0.5,
                            n_outcomes INT NOT NULL DEFAULT 0,
                            updated_at DATETIME DEFAULT GETDATE(),
                            PRIMARY KEY (series_id, detector_name)
                        )
                    END
                """))
        except Exception as exc:
            logger.warning(
                "anomaly_weights_table_creation_failed",
                extra={"error": str(exc)},
            )
    
    def load_weights(
        self,
        series_id: str,
    ) -> Optional[Dict[str, float]]:
        """Load adaptive weights for a series.
        
        Args:
            series_id: Series identifier
            
        Returns:
            Dict of {detector_name: weight} or None if not found
        """
        try:
            with self._engine.begin() as conn:
                results = conn.execute(
                    text("""
                        SELECT detector_name, weight
                        FROM zenin_ml.anomaly_detector_weights
                        WHERE series_id = :series_id
                        ORDER BY updated_at DESC
                    """),
                    {"series_id": series_id},
                ).fetchall()
            
            if not results:
                return None
            
            weights = {row[0]: row[1] for row in results}
            
            logger.debug(
                "anomaly_weights_loaded",
                extra={
                    "series_id": series_id,
                    "n_detectors": len(weights),
                },
            )
            
            return weights
            
        except Exception as exc:
            logger.warning(
                "anomaly_weights_load_failed",
                extra={"series_id": series_id, "error": str(exc)},
            )
            return None
    
    def save_weights(
        self,
        series_id: str,
        weights: Dict[str, float],
        precision_scores: Dict[str, float],
        n_outcomes: Dict[str, int],
    ) -> None:
        """Save or update adaptive weights for a series.
        
        Args:
            series_id: Series identifier
            weights: Dict of {detector_name: weight}
            precision_scores: Dict of {detector_name: precision}
            n_outcomes: Dict of {detector_name: outcome_count}
        """
        try:
            with self._engine.begin() as conn:
                for detector_name, weight in weights.items():
                    precision = precision_scores.get(detector_name, 0.5)
                    n_out = n_outcomes.get(detector_name, 0)
                    
                    conn.execute(
                        text("""
                            MERGE zenin_ml.anomaly_detector_weights AS target
                            USING (SELECT :series_id AS series_id, :detector_name AS detector_name) AS source
                            ON target.series_id = source.series_id 
                               AND target.detector_name = source.detector_name
                            WHEN MATCHED THEN
                                UPDATE SET
                                    weight = :weight,
                                    precision_score = :precision,
                                    n_outcomes = :n_outcomes,
                                    updated_at = GETDATE()
                            WHEN NOT MATCHED THEN
                                INSERT (series_id, detector_name, weight, precision_score, n_outcomes)
                                VALUES (:series_id, :detector_name, :weight, :precision, :n_outcomes);
                        """),
                        {
                            "series_id": series_id,
                            "detector_name": detector_name,
                            "weight": weight,
                            "precision": precision,
                            "n_outcomes": n_out,
                        },
                    )
            
            logger.debug(
                "anomaly_weights_saved",
                extra={
                    "series_id": series_id,
                    "n_detectors": len(weights),
                },
            )
            
        except Exception as exc:
            logger.warning(
                "anomaly_weights_save_failed",
                extra={
                    "series_id": series_id,
                    "error": str(exc)},
            )
