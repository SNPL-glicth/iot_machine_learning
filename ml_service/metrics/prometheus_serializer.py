"""Prometheus exposition text serializer for MetricsCollector.

FIX P2-2: Serializa métricas al formato Prometheus sin dependencia de prometheus_client.
Extracted from performance_metrics.py for ≤180 lines per file.
"""
from __future__ import annotations

import time
from collections import deque


def serialize_prometheus(
    prediction_count: int,
    error_count: int,
    reading_count: int,
    persistence_failure_count: int,
    prediction_times: deque,
    start_time: float,
) -> str:
    """Generate Prometheus exposition text from raw metric values.

    Args:
        prediction_count: Total predictions executed.
        error_count: Total prediction errors.
        reading_count: Total readings processed (stream messages).
        persistence_failure_count: Total dropped messages.
        prediction_times: Deque of prediction durations in ms.
        start_time: Unix timestamp of service start.

    Returns:
        Prometheus exposition text ending with newline.
    """
    now = time.time()
    lines: list = []

    # Counters
    lines.append("# HELP zenin_predictions_total Total de predicciones ejecutadas")
    lines.append("# TYPE zenin_predictions_total counter")
    lines.append(f"zenin_predictions_total {prediction_count}")

    lines.append("# HELP zenin_prediction_errors_total Total de errores de predicción")
    lines.append("# TYPE zenin_prediction_errors_total counter")
    lines.append(f"zenin_prediction_errors_total {error_count}")

    lines.append("# HELP zenin_stream_messages_processed_total Lecturas procesadas")
    lines.append("# TYPE zenin_stream_messages_processed_total counter")
    lines.append(f"zenin_stream_messages_processed_total {reading_count}")

    lines.append("# HELP zenin_stream_messages_dropped_total Mensajes descartados por error")
    lines.append("# TYPE zenin_stream_messages_dropped_total counter")
    lines.append(f"zenin_stream_messages_dropped_total {persistence_failure_count}")

    # Histogram: prediction duration
    buckets = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    lines.append("# HELP zenin_prediction_duration_seconds Latencia de predicciones")
    lines.append("# TYPE zenin_prediction_duration_seconds histogram")
    times_sec = [t / 1000.0 for t in prediction_times]
    total_sum = sum(times_sec)
    count = len(times_sec)
    for le in buckets:
        cumulative = sum(1 for t in times_sec if t <= le)
        lines.append(f'zenin_prediction_duration_seconds_bucket{{le="{le}"}} {cumulative}')
    lines.append(f'zenin_prediction_duration_seconds_bucket{{le="+Inf"}} {count}')
    lines.append(f"zenin_prediction_duration_seconds_sum {total_sum:.6f}")
    lines.append(f"zenin_prediction_duration_seconds_count {count}")

    # Gauge: active sensors (approximate)
    lines.append("# HELP zenin_active_sensors Sensores activos (aprox)")
    lines.append("# TYPE zenin_active_sensors gauge")
    lines.append(f"zenin_active_sensors {prediction_count + reading_count}")

    # Gauge: uptime
    lines.append("# HELP zenin_uptime_seconds Segundos desde inicio")
    lines.append("# TYPE zenin_uptime_seconds gauge")
    lines.append(f"zenin_uptime_seconds {now - start_time:.1f}")

    return "\n".join(lines) + "\n"


def serialize_cognitive_prometheus(
    cognitive_budget_exceeded: int,
    cognitive_phases_skipped: int,
    cognitive_fallbacks: int,
    cognitive_phase_times: deque,
) -> str:
    """FIX PROD-2: Serialize cognitive metrics to Prometheus text."""
    lines: list = []
    lines.append("# HELP zenin_cognitive_budget_exceeded_total Cognitive budget exceeded count")
    lines.append("# TYPE zenin_cognitive_budget_exceeded_total counter")
    lines.append(f"zenin_cognitive_budget_exceeded_total {cognitive_budget_exceeded}")
    lines.append("# HELP zenin_cognitive_phases_skipped_total Skipped phases count")
    lines.append("# TYPE zenin_cognitive_phases_skipped_total counter")
    lines.append(f"zenin_cognitive_phases_skipped_total {cognitive_phases_skipped}")
    lines.append("# HELP zenin_cognitive_fallbacks_total Fallback count")
    lines.append("# TYPE zenin_cognitive_fallbacks_total counter")
    lines.append(f"zenin_cognitive_fallbacks_total {cognitive_fallbacks}")
    buckets = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    lines.append("# HELP zenin_cognitive_phase_duration_seconds Cognitive phase duration")
    lines.append("# TYPE zenin_cognitive_phase_duration_seconds histogram")
    times_sec = [t / 1000.0 for _, t in cognitive_phase_times]
    total_sum = sum(times_sec)
    count = len(times_sec)
    for le in buckets:
        cumulative = sum(1 for t in times_sec if t <= le)
        lines.append(f'zenin_cognitive_phase_duration_seconds_bucket{{le="{le}"}} {cumulative}')
    lines.append(f'zenin_cognitive_phase_duration_seconds_bucket{{le="+Inf"}} {count}')
    lines.append(f"zenin_cognitive_phase_duration_seconds_sum {total_sum:.6f}")
    lines.append(f"zenin_cognitive_phase_duration_seconds_count {count}")
    return "\n".join(lines) + "\n"
