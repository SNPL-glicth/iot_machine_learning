"""Metrics data types — extracted from performance_metrics.py for ≤180 lines."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MLMetrics:
    """ML service performance metrics."""

    # Prediction metrics
    total_predictions: int = 0
    predictions_per_minute: float = 0.0
    avg_prediction_time_ms: float = 0.0
    max_prediction_time_ms: float = 0.0

    # Reading processing metrics
    total_readings_processed: int = 0
    readings_per_minute: float = 0.0
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0

    # Error metrics
    total_errors: int = 0
    error_rate: float = 0.0

    # Persistence metrics
    persistence_successes: int = 0
    persistence_failures: int = 0

    # Anomaly detection metrics
    total_anomalies_detected: int = 0
    total_anomalies_normal: int = 0

    # Broker metrics
    broker_connected: bool = False
    broker_type: str = "unknown"

    # Uptime
    uptime_seconds: float = 0.0
    started_at: str = ""

    # FIX PROD-2: cognitive metrics
    cognitive_budget_exceeded: int = 0
    cognitive_phases_skipped: int = 0
    cognitive_fallbacks: int = 0

    def to_dict(self) -> dict:
        return {
            "predictions": {
                "total": self.total_predictions,
                "per_minute": round(self.predictions_per_minute, 2),
                "avg_time_ms": round(self.avg_prediction_time_ms, 2),
                "max_time_ms": round(self.max_prediction_time_ms, 2),
            },
            "readings": {
                "total": self.total_readings_processed,
                "per_minute": round(self.readings_per_minute, 2),
                "avg_time_ms": round(self.avg_processing_time_ms, 2),
                "max_time_ms": round(self.max_processing_time_ms, 2),
            },
            "errors": {
                "total": self.total_errors,
                "rate": round(self.error_rate, 4),
            },
            "persistence": {
                "successes": self.persistence_successes,
                "failures": self.persistence_failures,
            },
            "anomalies": {
                "detected": self.total_anomalies_detected,
                "normal": self.total_anomalies_normal,
            },
            "broker": {
                "connected": self.broker_connected,
                "type": self.broker_type,
            },
            "uptime": {
                "seconds": round(self.uptime_seconds, 2),
                "started_at": self.started_at,
            },
            "cognitive": {
                "budget_exceeded": self.cognitive_budget_exceeded,
                "phases_skipped": self.cognitive_phases_skipped,
                "fallbacks": self.cognitive_fallbacks,
            },
        }
