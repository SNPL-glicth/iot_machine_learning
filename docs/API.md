# Zenin ML Service — API Reference

> **Base URL:** `http://localhost:8002`  
> **Auth:** All endpoints require `X-API-Key` header.  
> **Version:** 0.2.0

---

## Quick Start

```bash
# 1. Start the service
docker compose -f iot_machine_learning/docker-compose.yml up -d

# 2. Health check
curl -H "X-API-Key: $ML_API_KEY" http://localhost:8002/health

# 3. Make a prediction
curl -X POST http://localhost:8002/ml/predict \
  -H "X-API-Key: $ML_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"sensor_id": 1, "horizon_minutes": 10, "window": 60}'
```

---

## Endpoints

### Core Prediction

#### `POST /ml/predict`

Generate a prediction for a sensor using the cognitive ML pipeline.

**Request:**
```json
{
  "sensor_id": 1,
  "horizon_minutes": 10,
  "window": 60,
  "dedupe_minutes": 10
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sensor_id` | int (>0) | required | Sensor ID |
| `horizon_minutes` | int (1-1440) | 10 | Prediction horizon in minutes |
| `window` | int (1-1000) | 60 | Historical data window size |
| `dedupe_minutes` | int (1-1440) | 10 | Event deduplication window |

**Response (200):**
```json
{
  "sensor_id": 1,
  "model_id": 1,
  "prediction_id": 12345,
  "predicted_value": 23.456,
  "confidence": 0.87,
  "target_timestamp": "2026-05-19T20:00:00Z",
  "horizon_minutes": 10,
  "window": 60,
  "trend": "up",
  "engine_used": "taylor_ls",
  "confidence_level": "high",
  "processing_time_ms": 45.2,
  "audit_trace_id": "aud_abc123",
  "explanation_summary": "Taylor (least-squares) selected: stable regime, low noise",
  "structural_analysis": {
    "regime": "stable",
    "slope": 0.12,
    "curvature": 0.003,
    "noise_ratio": 0.05,
    "stability": 0.15,
    "trend_strength": 0.72,
    "mean": 23.1,
    "std": 0.8,
    "n_points": 60
  },
  "metacognitive": {
    "certainty": "high",
    "disagreement": "consensus",
    "cognitive_stability": "stable",
    "overfit_risk": "low",
    "engine_conflict": "none"
  },
  "decision": "normal",
  "verdict": "Operating within expected range",
  "severity": "info",
  "action_required": false,
  "action": null
}
```

**Async mode:** Add `?async=true` or header `X-Async-Prediction: true` to get `202 Accepted` with a `prediction_id` for polling.

#### `POST /ml/predict/batch`

Predict multiple sensors in a single request with configurable concurrency.

**Request:**
```json
{
  "predictions": [
    {"sensor_id": 1, "horizon_minutes": 10, "window": 60},
    {"sensor_id": 2, "horizon_minutes": 10, "window": 60}
  ],
  "max_concurrency": 10
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `predictions` | List[PredictRequest] | required | Up to 100 prediction requests |
| `max_concurrency` | int (1-50) | 10 | Max parallel predictions within batch |

**Response (200):**
```json
{
  "total": 2,
  "succeeded": 2,
  "failed": 0,
  "results": [
    {"sensor_id": 1, "success": true, "result": { ... }, "elapsed_ms": 45.2},
    {"sensor_id": 2, "success": true, "result": { ... }, "elapsed_ms": 38.1}
  ],
  "total_elapsed_ms": 52.3
}
```

#### `GET /ml/predict/{prediction_id}/status`

Poll async prediction result.

**Response (202 - pending):**
```json
{"status": "pending"}
```

**Response (200 - completed):**
```json
{"status": "completed", "result": { ... }}
```

---

### Health & Readiness

#### `GET /health`

Liveness probe with component health.

**Response (200):**
```json
{
  "status": "ok",
  "degraded": false,
  "broker": {"connected": true, "type": "redis"},
  "poller": {"enabled": true, "healthy": true, "total_processed": 1500},
  "sql_pool": {"checked_out": 2, "pool_size": 20, "overflow": 0, "healthy": true},
  "tsdb_circuit": "closed",
  "dist_window_circuit": "closed"
}
```

Returns `503` if any critical component is unhealthy.

#### `GET /ready`

Readiness probe — checks DB connectivity. Returns `503` if not ready.

---

### Observability

#### `GET /ml/metrics`

Performance metrics. Add `?format=prometheus` for Prometheus exposition format.

#### `GET /ml/diagnostics`

Cognitive pipeline diagnostics: slowest phases, budget exceeded count, fallback rate.

```json
{
  "slowest_phases": [
    {"phase": "fuse_phase", "max_ms": 12.5},
    {"phase": "perception_phase", "max_ms": 8.3}
  ],
  "cognitive_stats": {
    "budget_exceeded": 0,
    "phases_skipped": 2,
    "fallbacks": 0,
    "total_recorded": 150
  }
}
```

#### `GET /ml/broker/health`

Redis stream broker health status.

#### `GET /ml/observability/metrics`

Detailed ML engine metrics (prediction counts, latencies, error rates).

#### `GET /ml/observability/fallback-rate`

Fallback-to-baseline rate by engine and time window.

#### `GET /ml/observability/engine-distribution`

Distribution of engine usage (which engine won, how often).

#### `GET /ml/observability/semantic-impact`

Semantic analysis impact metrics.

#### `GET /ml/observability/silent-failures`

Predictions that succeeded but with degraded confidence.

#### `GET /ml/activation/status`

Feature flag activation status for all ML features.

---

### Cognitive Memory

#### `POST /ml/index-document`

Index a document into cognitive memory (Weaviate).

**Request:**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "content_type": "text",
  "normalized_payload": {
    "title": "Sensor maintenance report",
    "body": "Temperature sensor 42 recalibrated..."
  }
}
```

#### `POST /ml/semantic-search`

Semantic search over indexed documents.

**Request:**
```json
{
  "query": "temperature sensor calibration",
  "tenant_id": "tenant_abc",
  "limit": 10
}
```

---

### Telemetry

#### `GET /telemetry/ml-features/latest/{sensor_id}`

Latest computed ML features for a sensor (min, max, fluctuation, thresholds).

---

### Governance

#### `GET /governance/status`

Complete governance system state: parameter registry, convergence detectors, bounds violations.

#### `GET /governance/parameters`

List all registered ML parameters with category, scope, and current value.

#### `GET /governance/history?parameter_name=ML_BAYES_ALPHA&limit=50`

Audit trail of parameter changes.

#### `POST /governance/reset-convergence`

Reset convergence detectors (useful post-deploy).

---

### Query

#### `POST /query`

Natural language query over ML analysis results.

---

## Architecture Overview

```
Client Request
    │
    ▼
FastAPI (uvicorn, 4 workers)
    │
    ├── /ml/predict ──► PredictionService
    │                       │
    │                       ▼
    │               MetaCognitiveOrchestrator
    │                   │
    │                   ├── SignalAnalyzer (regime detection)
    │                   ├── TaylorEngine (3 derivative methods)
    │                   ├── StatisticalEngine (EMA/Holt)
    │                   ├── BaselineEngine (persistence model)
    │                   ├── InhibitionGate (suppress unstable engines)
    │                   ├── PlasticityTracker (per-regime weight learning)
    │                   └── WeightedFusion (Σ prediction_i × weight_i)
    │
    ├── /health ────► SQL pool + Redis + Broker checks
    │
    └── /ml/metrics ► Prometheus-compatible metrics
```

## Configuration (Environment Variables)

### Core
| Variable | Default | Description |
|----------|---------|-------------|
| `ML_API_KEY` | required | API authentication key |
| `ML_API_WORKERS` | 4 | Uvicorn worker count |
| `ML_ENV` | production | Environment (dev/production) |
| `ML_LOG_LEVEL` | INFO | Log level |

### Database
| Variable | Default | Description |
|----------|---------|-------------|
| `IOT_DB_POOL_SIZE` | 20 | SQL connection pool size |
| `IOT_DB_MAX_OVERFLOW` | 30 | Max overflow connections |
| `ZENIN_DB_POOL_SIZE` | 20 | Zenin DB pool size |
| `ZENIN_DB_MAX_OVERFLOW` | 30 | Zenin DB max overflow |

### Redis
| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | redis://localhost:6379/0 | Redis connection URL |
| `REDIS_MAX_CONNECTIONS` | 150 | General pool max connections |
| `REDIS_STREAM_MAX_CONNECTIONS` | 50 | Stream pool max connections |

### ML Pipeline
| Variable | Default | Description |
|----------|---------|-------------|
| `ML_USE_COGNITIVE_ORCHESTRATOR` | false | Enable cognitive pipeline |
| `ML_BATCH_MAX_WORKERS` | 4 | Batch runner thread pool |
| `ML_BATCH_PARALLEL_WORKERS` | 8 | Parallel sensor processing |
| `ML_ORCHESTRATOR_WORKERS` | 4 | Orchestrator thread pool |
| `ML_ORCHESTRATOR_TIMEOUT_S` | 5.0 | Per-prediction timeout |

### Cognitive Memory
| Variable | Default | Description |
|----------|---------|-------------|
| `ML_ENABLE_COGNITIVE_MEMORY` | false | Enable Weaviate integration |
| `ML_COGNITIVE_MEMORY_URL` | "" | Weaviate URL |

## Error Codes

| HTTP | Meaning |
|------|---------|
| 200 | Success |
| 202 | Accepted (async prediction) |
| 400 | Invalid request (bad sensor_id, out-of-range params) |
| 401 | Missing or invalid API key |
| 404 | Prediction not found (async poll) |
| 500 | Internal server error |
| 503 | Service unhealthy / not ready |
