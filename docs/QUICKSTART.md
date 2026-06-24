# ZENIN ML Service — Quickstart Guide

## Prerequisites

- Python 3.10+
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
pip install -r requirements.txt

# Configure env
cp .env.example .env
# Edit .env with your DB/Redis credentials

# Run tests
pytest tests/ -q

# Start dev server
uvicorn ml_service.main:app --reload --port 8002
```

## Docker (Full Stack)

```bash
docker compose -f docker-compose.yml up --build
```

Services available:
| Service | Port | URL |
|---------|------|-----|
| ML Prediction API | 8002 | http://localhost:8002 |
| Redis | 6380 | redis://localhost:6380 |
| Weaviate | 8080 | http://localhost:8080 |
| t2v-transformers | — | Embedding service for Weaviate |

## First Prediction

```bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "series_id": "PAST-01",
    "values": [71.2, 71.4, 71.3, 71.5, 71.4],
    "timestamps": [1700000000, 1700000300, 1700000600, 1700000900, 1700001200]
  }'
```

Response:
```json
{
  "series_id": "PAST-01",
  "predicted_value": 71.35,
  "confidence": 0.55,
  "regime": "STABLE",
  "decision": "MONITOR",
  "top_expert": "baseline",
  "reasoning_phases": ["sanitize", "perceive", "predict", "fuse", "explain"]
}
```

## Health Check

```bash
curl http://localhost:8002/health
# {"service":"iot-ml-service","status":"ok"}
```

## Key Endpoints

See [API Documentation](./API.md) for full reference.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Synchronous prediction |
| POST | `/cognitive/predict` | Cognitive pipeline prediction |
| POST | `/batch/predict` | Batch prediction |
| POST | `/predict/async` | Async prediction (202) |
| GET | `/health` | Service health |
| GET | `/governance/status` | Governance observability |
| GET | `/observability/metrics` | Engine metrics |
| GET | `/telemetry/ml-features/latest/{sensor_id}` | Telemetry features |
| POST | `/query` | Natural language query |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     ML Service (FastAPI)                     │
│  25+ fases cognitivas, 8 engines, 7 anomaly detectors       │
├─────────────────────────────────────────────────────────────┤
│  Application Layer                                          │
│  ├── Use cases (predict, detect anomalies, analyze patterns)│
│  ├── DecisionService, Evaluation, Explainability            │
├─────────────────────────────────────────────────────────────┤
│  Domain Layer                                               │
│  ├── Entities (Prediction, Anomaly, Explanation, etc.)      │
│  ├── Ports (23 interfaces), Services (33), Policies, Tools  │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                       │
│  ├── ML Engines (Taylor, Statistical, Kalman, Baseline,     │
│  │             LightGBM, Adaptive Ensemble, Multivariate)   │
│  ├── Cognitive (25+ phases, bayesian weights, drift, Hampel)│
│  ├── Anomaly (7 detectors v2.0, RUL estimator)              │
│  ├── MoE (gating tree, sparse fusion, expert registry)      │
│  ├── Inference (MLE, Bayes, Naive Bayes, Platt scaling)     │
│  ├── Optimization (SGD, L-BFGS, genetic, PSO)              │
│  ├── Persistence (Redis, SQL Server, Weaviate vector DB)    │
│  └── Governance (9 componentes: registry, bounds, tuning)   │
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
