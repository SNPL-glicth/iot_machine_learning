"""Plasticity queries module - Contextual learning and engine health."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


class PlasticityQueries:
    """Queries para plasticidad contextual y salud de engines."""
    
    def __init__(self, conn: Connection) -> None:
        self._conn = conn
    
    def record_contextual_error(
        self,
        series_id: str,
        engine_name: str,
        predicted_value: float,
        actual_value: float,
        error: float,
        penalty: float,
        regime: str,
        noise_ratio: float,
        volatility: float,
        time_of_day: int,
        consecutive_failures: int,
        is_critical_zone: bool,
        context_key: str,
    ) -> None:
        """Record contextual prediction error."""
        try:
            self._conn.execute(
                text("""
                    INSERT INTO contextual_prediction_errors (
                        series_id, engine_name, predicted_value, actual_value,
                        absolute_error, penalized_error, regime, noise_ratio,
                        volatility, time_of_day, consecutive_failures,
                        is_critical_zone, context_key, timestamp
                    )
                    VALUES (
                        :series_id, :engine_name, :predicted_value, :actual_value,
                        :absolute_error, :penalized_error, :regime, :noise_ratio,
                        :volatility, :time_of_day, :consecutive_failures,
                        :is_critical_zone, :context_key, SYSDATETIME()
                    )
                """),
                {
                    "series_id": series_id,
                    "engine_name": engine_name,
                    "predicted_value": predicted_value,
                    "actual_value": actual_value,
                    "absolute_error": error,
                    "penalized_error": penalty,
                    "regime": regime,
                    "noise_ratio": noise_ratio,
                    "volatility": volatility,
                    "time_of_day": time_of_day,
                    "consecutive_failures": consecutive_failures,
                    "is_critical_zone": is_critical_zone,
                    "context_key": context_key,
                },
            )
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            logger.error(
                "record_contextual_error_failed",
                extra={"series_id": series_id, "engine_name": engine_name, "error": str(e)},
            )
    
    def get_contextual_performance(
        self,
        series_id: str,
        engine_name: str,
        context_key: str,
        window_size: int = 50,
    ) -> Optional[Dict[str, float]]:
        """Get contextual performance metrics."""
        try:
            result = self._conn.execute(
                text("""
                    SELECT 
                        AVG(absolute_error) as mae,
                        STDEV(absolute_error) as std_dev,
                        COUNT(*) as count
                    FROM (
                        SELECT TOP :window_size absolute_error
                        FROM contextual_prediction_errors
                        WHERE series_id = :series_id
                          AND engine_name = :engine_name
                          AND context_key = :context_key
                        ORDER BY timestamp DESC
                    ) recent_errors
                """),
                {
                    "series_id": series_id,
                    "engine_name": engine_name,
                    "context_key": context_key,
                    "window_size": window_size,
                },
            ).fetchone()
            
            if result and result.count and result.count >= 5:
                return {
                    "mae": float(result.mae) if result.mae else 0.0,
                    "std_dev": float(result.std_dev) if result.std_dev else 0.0,
                    "count": int(result.count),
                }
            
            return None
            
        except Exception as e:
            logger.error(
                "get_contextual_performance_failed",
                extra={"series_id": series_id, "engine_name": engine_name, "context_key": context_key, "error": str(e)},
            )
            return None
    
    def update_engine_health(
        self,
        series_id: str,
        engine_name: str,
        consecutive_failures: int,
        consecutive_successes: int,
        total_predictions: int,
        total_errors: int,
        last_error: float,
        failure_rate: float,
        is_inhibited: bool,
        inhibition_reason: Optional[str] = None,
        last_success_time: Optional[str] = None,
        last_failure_time: Optional[str] = None,
    ) -> None:
        """Update engine health status (UPSERT)."""
        try:
            self._conn.execute(
                text("""
                    MERGE engine_health_status AS target
                    USING (SELECT :series_id AS series_id, :engine_name AS engine_name) AS source
                    ON target.series_id = source.series_id AND target.engine_name = source.engine_name
                    WHEN MATCHED THEN
                        UPDATE SET
                            consecutive_failures = :consecutive_failures,
                            consecutive_successes = :consecutive_successes,
                            total_predictions = :total_predictions,
                            total_errors = :total_errors,
                            last_error = :last_error,
                            failure_rate = :failure_rate,
                            is_inhibited = :is_inhibited,
                            inhibition_reason = :inhibition_reason,
                            inhibited_at = CASE WHEN :is_inhibited = 1 AND target.is_inhibited = 0 
                                               THEN SYSDATETIME() 
                                               ELSE target.inhibited_at 
                                          END,
                            last_success_time = :last_success_time,
                            last_failure_time = :last_failure_time,
                            last_updated = SYSDATETIME()
                    WHEN NOT MATCHED THEN
                        INSERT (
                            series_id, engine_name, consecutive_failures, consecutive_successes,
                            total_predictions, total_errors, last_error, failure_rate,
                            is_inhibited, inhibition_reason, inhibited_at,
                            last_success_time, last_failure_time, last_updated
                        )
                        VALUES (
                            :series_id, :engine_name, :consecutive_failures, :consecutive_successes,
                            :total_predictions, :total_errors, :last_error, :failure_rate,
                            :is_inhibited, :inhibition_reason, 
                            CASE WHEN :is_inhibited = 1 THEN SYSDATETIME() ELSE NULL END,
                            :last_success_time, :last_failure_time, SYSDATETIME()
                        );
                """),
                {
                    "series_id": series_id,
                    "engine_name": engine_name,
                    "consecutive_failures": consecutive_failures,
                    "consecutive_successes": consecutive_successes,
                    "total_predictions": total_predictions,
                    "total_errors": total_errors,
                    "last_error": last_error,
                    "failure_rate": failure_rate,
                    "is_inhibited": is_inhibited,
                    "inhibition_reason": inhibition_reason,
                    "last_success_time": last_success_time,
                    "last_failure_time": last_failure_time,
                },
            )
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            logger.error(
                "update_engine_health_failed",
                extra={"series_id": series_id, "engine_name": engine_name, "error": str(e)},
            )
    
    def get_engine_health(
        self,
        series_id: str,
        engine_name: str,
    ) -> Optional[Dict[str, any]]:
        """Get engine health status."""
        try:
            result = self._conn.execute(
                text("""
                    SELECT 
                        consecutive_failures, consecutive_successes,
                        total_predictions, total_errors,
                        last_error, failure_rate,
                        is_inhibited, inhibition_reason,
                        inhibited_at, last_success_time,
                        last_failure_time, last_updated
                    FROM engine_health_status
                    WHERE series_id = :series_id
                      AND engine_name = :engine_name
                """),
                {"series_id": series_id, "engine_name": engine_name},
            ).fetchone()
            
            if result:
                return {
                    "consecutive_failures": int(result.consecutive_failures),
                    "consecutive_successes": int(result.consecutive_successes),
                    "total_predictions": int(result.total_predictions),
                    "total_errors": int(result.total_errors),
                    "last_error": float(result.last_error),
                    "failure_rate": float(result.failure_rate),
                    "is_inhibited": bool(result.is_inhibited),
                    "inhibition_reason": result.inhibition_reason,
                    "inhibited_at": result.inhibited_at.isoformat() if result.inhibited_at else None,
                    "last_success_time": result.last_success_time.isoformat() if result.last_success_time else None,
                    "last_failure_time": result.last_failure_time.isoformat() if result.last_failure_time else None,
                    "last_updated": result.last_updated.isoformat() if result.last_updated else None,
                }
            
            return None
            
        except Exception as e:
            logger.error(
                "get_engine_health_failed",
                extra={"series_id": series_id, "engine_name": engine_name, "error": str(e)},
            )
            return None
