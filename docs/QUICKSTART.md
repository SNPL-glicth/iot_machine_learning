# ZENIN ML Service — Quickstart Guide

## Prerequisites

- Python 3.11+
- Redis 7+
- SQL Server (or compatible) with `zenin_db` schema
- (Optional) Weaviate for cognitive memory features

## Local Development

```bash
# Clone and enter
cd iot_machine_learning

# Create virtual env
python -m venv .venv && source .venv/bin/activate

# Install deps
pip install -r requirements.txt -r requirements-dev.txt

# Configure env
cp .env.example .env
# Edit .env with your DB/Redis credentials

# Run tests
pytest tests/ -q

# Start dev server
uvicorn iot_machine_learning.ml_service.main:app --reload --port 8002
```

## Docker (Standalone ML)

```bash
# From workspace root:
docker compose -f iot_machine_learning/docker-compose.yml up --build
```

## Full Platform Demo

```bash
# From workspace root:
cp .env.demo .env
docker compose -f docker-compose.demo.yml up --build
```

Services available:
| Service | Port | URL |
|---------|------|-----|
| ML Prediction API | 8002 | http://localhost:8002 |
| Telemetry API | 3000 | http://localhost:3000 |
| Ingest Service | 8001 | http://localhost:8001 |
| Monitor Backend | 3001 | http://localhost:3001 |
| MQTT Broker | 1883 | mqtt://localhost:1883 |
| EMQX Dashboard | 18083 | http://localhost:18083 |
| Redis | 6379 | redis://localhost:6379 |

## First Prediction

```bash
curl -X POST http://localhost:8002/ml/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-change-in-production" \
  -d '{
    "sensor_id": 1,
    "horizon_minutes": 10,
    "window": 60
  }'
```

Response:
```json
{
  "sensor_id": 1,
  "predicted_value": 24.7,
  "confidence": 0.87,
  "trend": "stable",
  "engine_used": "cognitive_orchestrator",
  "decision": { "action": "monitor", "urgency": "low" },
  "meta_diagnostic": { ... }
}
```

## Health Check

```bash
curl http://localhost:8002/health
# {"status":"ok","degraded":false,"engines":["taylor","statistical","cognitive"]}
```

## Key Endpoints

See [API Documentation](./API.md) for full reference.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ml/predict` | Synchronous prediction |
| POST | `/ml/predict?async=true` | Async prediction (202) |
| GET | `/ml/predict/{id}/status` | Poll async result |
| GET | `/health` | Service health |
| GET | `/ml/metrics` | Performance metrics |
| GET | `/ml/diagnostics` | Engine diagnostics |
| POST | `/ml/index-document` | Index document for cognitive memory |
| POST | `/ml/semantic-search` | Semantic search over indexed docs |
| GET | `/governance/status` | Governance observability |
| GET | `/telemetry/ml-features/latest/{sensor_id}` | Telemetry features |

## Feature Flags

Control ML engine behavior via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_USE_COGNITIVE_ORCHESTRATOR` | `true` | Enable meta-cognitive engine selection |
| `ML_USE_BAYESIAN_WEIGHTS` | `true` | Bayesian weight tracking per sensor |
| `ML_USE_MOE_ENGINE` | `false` | Mixture-of-Experts gating |
| `ML_USE_COMPLIANCE_AUDIT` | `true` | HMAC audit trail for predictions |
| `ML_COGNITIVE_MEMORY_ENABLED` | `false` | Weaviate-backed semantic memory |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     ML Prediction API                        │
│  FastAPI (uvicorn, 4 workers)                               │
├─────────────────────────────────────────────────────────────┤
│  Application Layer                                          │
│  ├── PredictionService (use cases)                          │
│  ├── DecisionService (action recommendations)               │
│  └── DocumentAnalyzer (text cognitive engine)               │
├─────────────────────────────────────────────────────────────┤
│  Domain Layer                                               │
│  ├── Entities (Prediction, SensorReading, TimeSeries)       │
│  ├── Ports (PredictionPort, StoragePort, AuditPort)         │
│  └── Services (Severity, Actions, Plasticity)               │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                       │
│  ├── ML Engines (Taylor, Statistical, Cognitive, MoE)       │
│  ├── Persistence (Redis cache, SQL storage, Weaviate)       │
│  ├── Adapters (Calibrators, Drift detectors, Compliance)    │
│  └── Neural (Attention, Feedforward, Hybrid)                │
└─────────────────────────────────────────────────────────────┘
```

## Running Tests

```bash
# All tests
pytest tests/ -q

# Unit tests only (fast)
pytest tests/unit/ -q

# Coverage report
pytest tests/ --cov=iot_machine_learning --cov-report=html

# Specific module
pytest tests/ -k "cognitive" -q
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: dotenv` | `pip install python-dotenv` |
| `Connection refused (Redis)` | Ensure Redis is running on configured port |
| `pyodbc.OperationalError` | Check DB credentials in `.env` |
| `ZENIN_DB_PASSWORD is required` | Set the env variable or add to `.env` |
| Low confidence predictions | Increase `window` parameter or feed more historical data |
