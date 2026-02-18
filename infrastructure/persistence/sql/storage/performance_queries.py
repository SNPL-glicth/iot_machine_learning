"""Performance queries module."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convierte a float con guard contra None/NaN/Inf."""
    if value is None:
        return default
    try:
        import math
        f = float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


class PerformanceQueries:
    """Queries para métricas de performance."""
    
    def __init__(self, conn: Connection) -> None:
        self._conn = conn
    
    def record_prediction_error(
        self,
        series_id: str,
        engine_name: str,
        predicted_value: float,
        actual_value: float,
    ) -> None:
        """Record prediction error for adaptive learning."""
        absolute_error = abs(predicted_value - actual_value)
        
        try:
            self._conn.execute(
                text(
                    """
                    INSERT INTO dbo.prediction_errors (
                        series_id, engine_name,
                        predicted_value, actual_value,
                        absolute_error, timestamp
                    )
                    VALUES (
                        :series_id, :engine_name,
                        :predicted_value, :actual_value,
                        :absolute_error, SYSDATETIME()
                    )
                    """
                ),
                {
                    "series_id": series_id,
                    "engine_name": engine_name,
                    "predicted_value": predicted_value,
                    "actual_value": actual_value,
                    "absolute_error": absolute_error,
                },
            )
            
            logger.debug(
                "prediction_error_recorded",
                extra={
                    "series_id": series_id,
                    "engine": engine_name,
                    "error": absolute_error,
                },
            )
        except Exception as e:
            logger.warning(
                "failed_to_record_prediction_error",
                extra={"series_id": series_id, "engine": engine_name, "error": str(e)},
            )

    def get_rolling_performance(
        self,
        series_id: str,
        engine_name: str,
        window_size: int = 50,
    ) -> Optional[Dict[str, float]]:
        """Get rolling MAE and std_dev for adaptive weights."""
        try:
            row = self._conn.execute(
                text(
                    """
                    SELECT
                        AVG(absolute_error) as mae,
                        STDEV(absolute_error) as std_dev,
                        COUNT(*) as count
                    FROM (
                        SELECT TOP (:window_size) absolute_error
                        FROM dbo.prediction_errors
                        WHERE series_id = :series_id
                          AND engine_name = :engine_name
                        ORDER BY timestamp DESC
                    ) recent_errors
                    """
                ),
                {
                    "series_id": series_id,
                    "engine_name": engine_name,
                    "window_size": window_size,
                },
            ).fetchone()
            
            if not row or row[2] < 5:
                return None
            
            return {
                "mae": _safe_float(row[0], default=0.0),
                "std_dev": _safe_float(row[1], default=0.0),
                "count": int(row[2]),
            }
        except Exception as e:
            logger.warning(
                "failed_to_get_rolling_performance",
                extra={"series_id": series_id, "engine": engine_name, "error": str(e)},
            )
            return None

    def compute_confidence_interval(
        self,
        series_id: str,
        engine_name: str,
        predicted_value: float,
        confidence_level: float = 0.95,
    ) -> Optional[Dict[str, float]]:
        """Compute confidence interval for prediction."""
        perf = self.get_rolling_performance(series_id, engine_name)
        if not perf:
            return None
        
        std_dev = perf["std_dev"]
        
        if confidence_level >= 0.99:
            z_score = 2.576
        elif confidence_level >= 0.95:
            z_score = 1.96
        else:
            z_score = 1.645
        
        margin = z_score * std_dev
        
        return {
            "lower": predicted_value - margin,
            "upper": predicted_value + margin,
            "std_dev": std_dev,
            "z_score": z_score,
        }
