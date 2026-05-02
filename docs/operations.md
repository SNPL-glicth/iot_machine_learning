## Monitoring & Observability

### Prometheus Metrics

ZENIN exports metrics at `/metrics`:

```
zenin_predictions_total{series_id,engine}
zenin_prediction_latency_ms{quantile}
zenin_prediction_confidence_avg

zenin_plasticity_updates_total{regime}
zenin_plasticity_weights{regime,engine}

zenin_anomalies_detected_total{severity}
zenin_anomaly_detection_latency_ms{quantile}

zenin_tool_executions_total{tool,guard}
zenin_tool_execution_failures_total{tool}

zenin_circuit_breaker_state{service}       # 0=closed, 1=open
zenin_concept_drift_score{series_id}
```

### Health Checks

| Endpoint | Purpose |
|----------|---------|
| `GET /health/live` | Liveness probe — Kubernetes restarts unhealthy pods |
| `GET /health/ready` | Readiness probe — checks Redis and SQL connectivity |
| `GET /health` | Full system status, version, component health |

### Logging

Structured JSON logging with correlation IDs:

```json
{
  "timestamp": "2026-04-02T14:30:00Z",
  "level": "INFO",
  "correlation_id": "abc-123-def",
  "component": "MetaCognitiveOrchestrator",
  "event": "prediction_completed",
  "series_id": "sensor_42",
  "predicted_value": 87.2,
  "confidence": 0.85,
  "latency_ms": 45.2,
  "regime": "TRENDING",
  "selected_engine": "taylor"
}
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Low confidence (< 0.5) | High noise / engine disagreement / insufficient data | Wait for stable signal; ensure 3–5 readings minimum |
| Amnesic mode | Circuit breaker opened for persistence | Check Redis/SQL connectivity; restart service |
| Plasticity not learning | `record_actual()` not called; Redis down | Verify `record_actual()` usage; confirm Redis is running |
| Decision scores wrong | `ML_DECISION_BASE_SCORES` JSON malformed | Validate JSON before deploying |
| Plasticity not adapting | `ML_PLASTICITY_REGIME_ALPHAS` not picked up | Confirm env var is set; hot-reload applies to new calls only |

### Debug Mode

```bash
export ZENIN_LOG_LEVEL=DEBUG
export ZENIN_LOG_STRUCTURED=true
```

View plasticity state:

```python
from infrastructure.ml.cognitive.plasticity.base import PlasticityTracker

tracker = PlasticityTracker(redis_client=redis)
weights = tracker.get_weights("TRENDING")
print(weights)  # {'taylor': 0.7, 'baseline': 0.3, 'statistical': 0.0}
```

---

## Performance Tuning

**High Latency (> 100ms):**
- Disable iterative mode: `ML_ENABLE_ITERATIVE=false`
- Reduce engine count (remove ensemble if not needed)
- Enable sliding windows: `ML_STREAM_USE_SLIDING_WINDOW=true`
- Increase Redis cache TTL

**Memory Pressure:**
- Reduce `ML_SLIDING_WINDOW_MAX_SENSORS` (default: 1000)
- Lower `ML_SLIDING_WINDOW_TTL_SECONDS` (default: 3600)
- Disable cognitive memory: `ML_ENABLE_COGNITIVE_MEMORY=false`

**Database Load:**
- Enable batch processing: `ML_BATCH_PARALLEL_WORKERS=4`
- Disable stream predictions: `ML_STREAM_PREDICTIONS_ENABLED=false`
- Use write-behind caching for plasticity
