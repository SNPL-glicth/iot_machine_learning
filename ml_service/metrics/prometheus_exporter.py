"""Prometheus metrics exporter — R-6 Observability + Fase 5 Data Drift.
Lightweight metrics for ZENIN without prometheus_client dependency.
Tracks: latency, weights, anomalies, concept drift detection.
"""
from __future__ import annotations
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
@dataclass
class MetricValue:
    value: float
    timestamp: float
    labels: Dict[str, str]

class DataDriftDetector:
    """Fase 5: Lightweight concept drift detection using mean/std tracking."""
    def __init__(self, window_size: int = 100, drift_threshold: float = 2.0) -> None:
        self._window_size = window_size
        self._drift_threshold = drift_threshold
        self._baseline: Optional[Tuple[float, float]] = None
        self._recent: List[float] = []
        self._lock = threading.Lock()
    def update(self, values: List[float]) -> float:
        with self._lock:
            if not values:
                return 0.0
            self._recent.extend(values)
            self._recent = self._recent[-self._window_size:]
            if len(self._recent) < 20:
                return 0.0
            recent_mean = sum(self._recent) / len(self._recent)
            recent_std = max((sum((x - recent_mean) ** 2 for x in self._recent) / len(self._recent)) ** 0.5, 0.001)
            if self._baseline is None:
                self._baseline = (recent_mean, recent_std)
                return 0.0
            baseline_mean, baseline_std = self._baseline
            drift_score = abs(recent_mean - baseline_mean) / baseline_std
            if drift_score < self._drift_threshold:
                self._baseline = (baseline_mean * 0.9 + recent_mean * 0.1, baseline_std * 0.9 + recent_std * 0.1)
            return drift_score
    def is_drift_detected(self) -> bool:
        with self._lock:
            if self._baseline is None or len(self._recent) < 20:
                return False
            return abs(sum(self._recent) / len(self._recent) - self._baseline[0]) / self._baseline[1] > self._drift_threshold
    def get_stats(self) -> Dict[str, float]:
        with self._lock:
            if self._baseline is None:
                return {"drift_score": 0.0, "baseline_mean": 0.0, "baseline_std": 0.0}
            recent_mean = sum(self._recent) / len(self._recent) if self._recent else 0.0
            return {"drift_score": abs(recent_mean - self._baseline[0]) / self._baseline[1], "baseline_mean": self._baseline[0], "baseline_std": self._baseline[1]}

class PrometheusExporter:
    """Lightweight Prometheus-compatible metrics exporter with data drift detection."""
    def __init__(self, max_history: int = 1000) -> None:
        self._max_history = max_history
        self._lock = threading.RLock()
        self._latency_gauge: Dict[str, List[MetricValue]] = defaultdict(list)
        self._weight_gauge: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._anomaly_counter: int = 0
        self._anomaly_label_counts: Dict[str, int] = defaultdict(int)
        self._prediction_counter: int = 0
        self._error_counter: int = 0
        self._inhibition_events: int = 0
        self._drift_detectors: Dict[str, DataDriftDetector] = {}
        self._drift_alerts: int = 0
    def record_latency(self, engine_name: str, latency_ms: float, series_id: Optional[str] = None) -> None:
        labels = {"engine_name": engine_name}
        if series_id:
            labels["series_id"] = series_id
        with self._lock:
            self._latency_gauge[engine_name].append(MetricValue(value=latency_ms, timestamp=time.time(), labels=labels))
            if len(self._latency_gauge[engine_name]) > self._max_history:
                self._latency_gauge[engine_name] = self._latency_gauge[engine_name][-self._max_history:]
    def record_weight(self, series_id: str, engine_name: str, weight: float) -> None:
        with self._lock:
            self._weight_gauge[series_id][engine_name] = weight
    def record_values(self, series_id: str, values: List[float]) -> float:
        """Fase 5: Record values for drift detection. Returns drift score."""
        with self._lock:
            if series_id not in self._drift_detectors:
                self._drift_detectors[series_id] = DataDriftDetector()
            drift_score = self._drift_detectors[series_id].update(values)
            if drift_score > 2.0:
                self._drift_alerts += 1
            return drift_score
    def is_concept_drift_detected(self, series_id: str) -> bool:
        with self._lock:
            detector = self._drift_detectors.get(series_id)
            return detector.is_drift_detected() if detector else False
    def increment_anomaly_override(self, reason: str = "smart_inhibition") -> None:
        with self._lock:
            self._anomaly_counter += 1
            self._anomaly_label_counts[reason] += 1
    def increment_predictions(self) -> None:
        with self._lock:
            self._prediction_counter += 1
    def increment_errors(self) -> None:
        with self._lock:
            self._error_counter += 1
    def increment_inhibition(self) -> None:
        with self._lock:
            self._inhibition_events += 1
    def get_latency_stats(self, engine_name: str) -> Dict[str, float]:
        with self._lock:
            values = self._latency_gauge.get(engine_name, [])
            if not values:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0, "count": 0}
            latencies = sorted([v.value for v in values])
            n = len(latencies)
            return {"p50": latencies[int(n * 0.50)], "p95": latencies[int(n * 0.95)] if n >= 20 else latencies[-1], "p99": latencies[int(n * 0.99)] if n >= 100 else latencies[-1], "avg": sum(latencies) / n, "count": n}
    def get_drift_stats(self, series_id: str) -> Dict[str, float]:
        with self._lock:
            detector = self._drift_detectors.get(series_id)
            return detector.get_stats() if detector else {"drift_score": 0.0}
    
    def get_all_weights(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return dict(self._weight_gauge)
    
    def export_prometheus_format(self) -> str:
        lines = []
        with self._lock:
            lines.extend(["# HELP zenin_prediction_latency_ms Prediction latency by engine", "# TYPE zenin_prediction_latency_ms gauge"])
            for engine, values in self._latency_gauge.items():
                if values:
                    lines.append(f'zenin_prediction_latency_ms{{engine="{engine}"}} {values[-1].value}')
            lines.extend(["# HELP zenin_engine_weight Engine weight by series", "# TYPE zenin_engine_weight gauge"])
            for series_id, engines in self._weight_gauge.items():
                for engine_name, weight in engines.items():
                    lines.append(f'zenin_engine_weight{{series_id="{series_id}",engine="{engine_name}"}} {weight}')
            lines.extend(["# HELP zenin_anomaly_overrides_total Anomaly override events", "# TYPE zenin_anomaly_overrides_total counter", f'zenin_anomaly_overrides_total {self._anomaly_counter}'])
            for reason, count in self._anomaly_label_counts.items():
                lines.append(f'zenin_anomaly_overrides_total{{reason="{reason}"}} {count}')
            lines.extend(["# HELP zenin_concept_drift_detected Concept drift detection flag", "# TYPE zenin_concept_drift_detected gauge"])
            for series_id, detector in self._drift_detectors.items():
                lines.append(f'zenin_concept_drift_detected{{series_id="{series_id}"}} {1.0 if detector.is_drift_detected() else 0.0}')
            lines.extend(["# HELP zenin_concept_drift_score Drift severity score", "# TYPE zenin_concept_drift_score gauge"])
            for series_id, detector in self._drift_detectors.items():
                lines.append(f'zenin_concept_drift_score{{series_id="{series_id}"}} {detector.get_stats()["drift_score"]:.4f}')
            lines.extend(["# HELP zenin_drift_alerts_total Total drift alert events", "# TYPE zenin_drift_alerts_total counter", f'zenin_drift_alerts_total {self._drift_alerts}'])
            lines.extend(["# HELP zenin_predictions_total Total predictions", "# TYPE zenin_predictions_total counter", f'zenin_predictions_total {self._prediction_counter}'])
            lines.extend(["# HELP zenin_errors_total Total errors", "# TYPE zenin_errors_total counter", f'zenin_errors_total {self._error_counter}'])
            lines.extend(["# HELP zenin_inhibition_events_total Engine inhibition events", "# TYPE zenin_inhibition_events_total counter", f'zenin_inhibition_events_total {self._inhibition_events}'])
        return "\n".join(lines)
    def get_metrics_summary(self) -> Dict:
        with self._lock:
            drift_detected = sum(1 for d in self._drift_detectors.values() if d.is_drift_detected())
            return {"latency_engines": list(self._latency_gauge.keys()), "tracked_series": len(self._weight_gauge), "drift_monitored": len(self._drift_detectors), "drift_detected": drift_detected, "anomaly_overrides": self._anomaly_counter, "predictions": self._prediction_counter, "errors": self._error_counter, "inhibitions": self._inhibition_events}
_global_exporter: Optional[PrometheusExporter] = None
_global_lock = threading.Lock()
def get_exporter() -> PrometheusExporter:
    global _global_exporter
    if _global_exporter is None:
        with _global_lock:
            if _global_exporter is None:
                _global_exporter = PrometheusExporter()
    return _global_exporter
def reset_exporter() -> None:
    global _global_exporter
    with _global_lock:
        _global_exporter = None
