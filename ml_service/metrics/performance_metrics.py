"""Performance metrics for ML service.

Collects and exposes metrics for observability.
Thread-safe implementation for concurrent access.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Optional


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
        }


class MetricsCollector:
    """Thread-safe metrics collector for ML service."""
    
    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._start_time = time.time()
        self._started_at = datetime.now(timezone.utc)
        
        # Prediction tracking
        self._prediction_count = 0
        self._prediction_times: deque = deque(maxlen=1000)
        self._prediction_timestamps: deque = deque(maxlen=1000)
        
        # Reading tracking
        self._reading_count = 0
        self._reading_times: deque = deque(maxlen=1000)
        self._reading_timestamps: deque = deque(maxlen=1000)
        
        # Error tracking
        self._error_count = 0
        
        # Persistence tracking
        self._persistence_success_count = 0
        self._persistence_failure_count = 0
        
        # Anomaly tracking
        self._anomaly_detected_count = 0
        self._anomaly_normal_count = 0
        
        # Thread lock for updates
        self._update_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def record_prediction(self, duration_ms: float) -> None:
        """Record a prediction with its duration."""
        with self._update_lock:
            self._prediction_count += 1
            self._prediction_times.append(duration_ms)
            self._prediction_timestamps.append(time.time())
    
    def record_reading_processed(self, duration_ms: float) -> None:
        """Record a reading processed with its duration."""
        with self._update_lock:
            self._reading_count += 1
            self._reading_times.append(duration_ms)
            self._reading_timestamps.append(time.time())
    
    def record_error(self) -> None:
        """Record an error."""
        with self._update_lock:
            self._error_count += 1
    
    def record_persistence_success(self) -> None:
        """Record a successful persistence operation."""
        with self._update_lock:
            self._persistence_success_count += 1
    
    def record_persistence_failure(self) -> None:
        """Record a failed persistence operation."""
        with self._update_lock:
            self._persistence_failure_count += 1
    
    def record_anomaly_result(self, is_anomaly: bool) -> None:
        """Record an anomaly detection result."""
        with self._update_lock:
            if is_anomaly:
                self._anomaly_detected_count += 1
            else:
                self._anomaly_normal_count += 1
    
    def get_metrics(self) -> MLMetrics:
        """Get current metrics snapshot."""
        now = time.time()
        uptime = now - self._start_time
        
        with self._update_lock:
            # Calculate predictions per minute
            recent_predictions = sum(
                1 for ts in self._prediction_timestamps
                if now - ts < 60
            )
            predictions_per_minute = recent_predictions
            
            # Calculate readings per minute
            recent_readings = sum(
                1 for ts in self._reading_timestamps
                if now - ts < 60
            )
            readings_per_minute = recent_readings
            
            # Calculate prediction times
            avg_prediction_time = mean(self._prediction_times) if self._prediction_times else 0.0
            max_prediction_time = max(self._prediction_times) if self._prediction_times else 0.0
            
            # Calculate reading times
            avg_reading_time = mean(self._reading_times) if self._reading_times else 0.0
            max_reading_time = max(self._reading_times) if self._reading_times else 0.0
            
            # Calculate error rate
            total_ops = self._prediction_count + self._reading_count
            error_rate = self._error_count / total_ops if total_ops > 0 else 0.0
            
            # Get broker status
            broker_connected = False
            broker_type = "unknown"
            try:
                from ..broker import get_broker_health
                health = get_broker_health()
                broker_connected = health.get("connected", False)
                broker_type = health.get("type", "unknown")
            except Exception:
                pass
            
            return MLMetrics(
                total_predictions=self._prediction_count,
                predictions_per_minute=predictions_per_minute,
                avg_prediction_time_ms=avg_prediction_time,
                max_prediction_time_ms=max_prediction_time,
                total_readings_processed=self._reading_count,
                readings_per_minute=readings_per_minute,
                avg_processing_time_ms=avg_reading_time,
                max_processing_time_ms=max_reading_time,
                total_errors=self._error_count,
                error_rate=error_rate,
                persistence_successes=self._persistence_success_count,
                persistence_failures=self._persistence_failure_count,
                total_anomalies_detected=self._anomaly_detected_count,
                total_anomalies_normal=self._anomaly_normal_count,
                broker_connected=broker_connected,
                broker_type=broker_type,
                uptime_seconds=uptime,
                started_at=self._started_at.isoformat(),
            )


# Module-level functions for convenience
def get_metrics() -> MLMetrics:
    """Get current metrics."""
    return MetricsCollector.get_instance().get_metrics()


def record_prediction(duration_ms: float) -> None:
    """Record a prediction."""
    MetricsCollector.get_instance().record_prediction(duration_ms)


def record_reading_processed(duration_ms: float) -> None:
    """Record a reading processed."""
    MetricsCollector.get_instance().record_reading_processed(duration_ms)


def record_error() -> None:
    """Record an error."""
    MetricsCollector.get_instance().record_error()


def record_persistence_success() -> None:
    """Record a successful persistence operation."""
    MetricsCollector.get_instance().record_persistence_success()


def record_persistence_failure() -> None:
    """Record a failed persistence operation."""
    MetricsCollector.get_instance().record_persistence_failure()


def record_anomaly_result(is_anomaly: bool) -> None:
    """Record an anomaly detection result."""
    MetricsCollector.get_instance().record_anomaly_result(is_anomaly)
