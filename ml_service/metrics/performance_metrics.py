"""Performance metrics collector for ML service.

Thread-safe implementation for concurrent access.
MLMetrics dataclass lives in metrics_types.py.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from statistics import mean
from typing import Optional

from .metrics_types import MLMetrics


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
        
        # FIX PROD-2: cognitive phase tracking
        self._cognitive_phase_times: deque = deque(maxlen=5000)
        self._cognitive_budget_exceeded = 0
        self._cognitive_phases_skipped = 0
        self._cognitive_fallbacks = 0
        
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
    
    def record_cognitive_phase(self, phase_name: str, duration_ms: float) -> None:
        """FIX PROD-2: record per-phase cognitive timing."""
        with self._update_lock:
            self._cognitive_phase_times.append((phase_name, duration_ms))
    
    def record_cognitive_budget_exceeded(self) -> None:
        with self._update_lock:
            self._cognitive_budget_exceeded += 1
    
    def record_cognitive_phase_skipped(self) -> None:
        with self._update_lock:
            self._cognitive_phases_skipped += 1
    
    def record_cognitive_fallback(self) -> None:
        with self._update_lock:
            self._cognitive_fallbacks += 1
    
    def to_prometheus_text(self) -> str:
        """FIX P2-2: Serializa métricas al formato Prometheus exposition text."""
        from .prometheus_serializer import serialize_prometheus
        with self._update_lock:
            return serialize_prometheus(
                prediction_count=self._prediction_count,
                error_count=self._error_count,
                reading_count=self._reading_count,
                persistence_failure_count=self._persistence_failure_count,
                prediction_times=self._prediction_times,
                start_time=self._start_time,
            )

    def get_metrics(self) -> MLMetrics:
        """Get current metrics snapshot."""
        now = time.time()
        with self._update_lock:
            ppm = sum(1 for ts in self._prediction_timestamps if now - ts < 60)
            rpm = sum(1 for ts in self._reading_timestamps if now - ts < 60)
            total_ops = self._prediction_count + self._reading_count
            broker_connected, broker_type = False, "unknown"
            try:
                from ..broker import get_broker_health
                h = get_broker_health()
                broker_connected, broker_type = h.get("connected", False), h.get("type", "unknown")
            except Exception:
                pass
            return MLMetrics(
                total_predictions=self._prediction_count,
                predictions_per_minute=ppm,
                avg_prediction_time_ms=mean(self._prediction_times) if self._prediction_times else 0.0,
                max_prediction_time_ms=max(self._prediction_times) if self._prediction_times else 0.0,
                total_readings_processed=self._reading_count,
                readings_per_minute=rpm,
                avg_processing_time_ms=mean(self._reading_times) if self._reading_times else 0.0,
                max_processing_time_ms=max(self._reading_times) if self._reading_times else 0.0,
                total_errors=self._error_count,
                error_rate=self._error_count / total_ops if total_ops > 0 else 0.0,
                persistence_successes=self._persistence_success_count,
                persistence_failures=self._persistence_failure_count,
                total_anomalies_detected=self._anomaly_detected_count,
                total_anomalies_normal=self._anomaly_normal_count,
                broker_connected=broker_connected,
                broker_type=broker_type,
                uptime_seconds=now - self._start_time,
                started_at=self._started_at.isoformat(),
                cognitive_budget_exceeded=self._cognitive_budget_exceeded,
                cognitive_phases_skipped=self._cognitive_phases_skipped,
                cognitive_fallbacks=self._cognitive_fallbacks,
            )


# Module-level convenience functions
def get_metrics() -> MLMetrics:
    return MetricsCollector.get_instance().get_metrics()

def record_prediction(duration_ms: float) -> None:
    MetricsCollector.get_instance().record_prediction(duration_ms)

def record_reading_processed(duration_ms: float) -> None:
    MetricsCollector.get_instance().record_reading_processed(duration_ms)

def record_error() -> None:
    MetricsCollector.get_instance().record_error()

def record_persistence_success() -> None:
    MetricsCollector.get_instance().record_persistence_success()

def record_persistence_failure() -> None:
    MetricsCollector.get_instance().record_persistence_failure()

def record_anomaly_result(is_anomaly: bool) -> None:
    MetricsCollector.get_instance().record_anomaly_result(is_anomaly)

def record_cognitive_phase(phase_name: str, duration_ms: float) -> None:
    MetricsCollector.get_instance().record_cognitive_phase(phase_name, duration_ms)

def record_cognitive_budget_exceeded() -> None:
    MetricsCollector.get_instance().record_cognitive_budget_exceeded()

def record_cognitive_phase_skipped() -> None:
    MetricsCollector.get_instance().record_cognitive_phase_skipped()

def record_cognitive_fallback() -> None:
    MetricsCollector.get_instance().record_cognitive_fallback()
