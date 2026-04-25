# ZENIN

**Cognitive Decision Engine for Real-Time Data Analysis**

ZENIN is a production-grade cognitive engine that reads signals, detects anomalies, fuses multiple ML predictors, and produces **actionable decisions with full explanations** — all within a 500ms latency budget. It is designed for mission-critical domains where understanding *why* a decision was made matters as much as the decision itself.

Unlike black-box ML pipelines, ZENIN operates as a transparent cognitive loop: it perceives signal structure, predicts across multiple specialized engines, learns which engine to trust in each context, suppresses unreliable outputs, and explains its reasoning in human-readable traces. Every prediction carries a full audit trail.

---

## Why ZENIN Matters

### The Problem

Traditional ML pipelines for time-series and sensor data share three critical weaknesses:

1. **Black-box predictions** — they output a number without explaining *why*, making operators distrust automated decisions in mission-critical environments.
2. **Static models** — a single model trained offline cannot adapt when a sensor shifts from stable to volatile, or when a new operational regime emerges.
3. **No safety boundaries** — when a prediction triggers an autonomous action (shut a valve, page an on-call engineer), there is no gate to prevent bad calls from executing.

### The ZENIN Difference

ZENIN replaces the single-model pipeline with a **cognitive decision loop** that perceives, predicts, learns, and explains — all in real time.

| Advantage | Why it matters |
|-----------|---------------|
| **Signal Inhibition** | Noisy or contradictory engine outputs are suppressed *before* they reach the final decision, preventing contaminated predictions |
| **Multi-Engine Fusion** | Four specialized predictors (Taylor, seasonal FFT, statistical, baseline) compete; plasticity learns which wins in each signal regime, so accuracy improves without retraining |
| **Cognitive Orchestration** | The pipeline is not a fixed DAG — phases execute conditionally based on runtime signal characteristics and feature flags, keeping latency within budget |
| **Built-In Explainability** | Every prediction carries a reasoning trace (regime, engine contributions, confidence, recommended action), making ZENIN auditable by human operators and regulators |
| **Safety Guardrails** | Every autonomous action passes through AUTO/ASK/DENY levels; critical severity executes immediately, ambiguous cases escalate to human review |
| **Real-Time by Design** | p99 latency under 150ms for the full cognitive pipeline (perception → fusion), proven at 1,000+ sensors |

### Proven Effectiveness

| Metric | Value | Evidence |
|--------|-------|----------|
| Test coverage | 1,800+ automated tests | Unit, integration, and architectural meta-tests |
| Latency (p99) | < 150ms | Per-phase `PipelineTimer` instrumentation with 500ms budget guard |
| Online learning | Zero retraining | Bayesian weight updates per regime via `PlasticityTracker` |
| Outlier rejection | Hampel filter (≈3σ) | Removes rogue engine outputs before fusion |
| Concurrent engines | ThreadPool with per-engine timeout | `ML_PREDICT_MAX_WORKERS` / `ML_PREDICT_ENGINE_TIMEOUT_MS` |
| Architecture | Hexagonal + clean ports | Domain layer has zero external dependencies; fully mockable for testing |

---

## 🚀 Quick Start

### Installation

```bash
cd /home/Linux/Documentos/Iot_System/iot_machine_learning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start Dependencies

```bash
# Redis (required for streaming + plasticity)
docker run -d -p 6379:6379 redis:7-alpine

# SQL Server (required for persistence)
docker run -d -p 1434:1434 \
  -e SA_PASSWORD=YourPassword123 \
  -e ACCEPT_EULA=Y \
  mcr.microsoft.com/mssql/server:2022-latest
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Run Service

```bash
# Start ML API service
uvicorn ml_service.main:app --host 0.0.0.0 --port 8002 --reload

# Verify
curl http://localhost:8002/
# {"service": "iot-ml-service", "version": "0.3.0-GOLD", "status": "ok"}
```

> **Security note:** API key authentication via `X-API-Key` header.
> Set `ML_API_KEY` in `.env`. Empty value disables auth (dev mode only).
> Do not expose to the internet without VPN or reverse proxy.

---

## Key Features

| Capability | What it does |
|------------|-------------|
| **Multi-Engine Prediction** | Combines Taylor polynomial, seasonal FFT, statistical, and baseline engines; selects the best fit per signal regime |
| **Signal Regime Detection** | Automatically classifies signals as STABLE, TRENDING, VOLATILE, NOISY, or TRANSITIONAL |
| **Plasticity (Online Learning)** | Bayesian weight updates per regime — no offline training required |
| **Anomaly Detection** | Voting ensemble (Z-Score, IQR, Isolation Forest, LOF) with configurable consensus thresholds |
| **Decision Guardrails** | AUTO / ASK / DENY safety levels for every autonomous action |
| **Full Explainability** | Every prediction carries a reasoning trace: regime, engine contributions, confidence, and recommended actions |
| **Universal Analysis** *(v0.3.0)* | Text, numeric, tabular, and mixed-input analysis through a unified 5-phase cognitive pipeline |

---

## Architecture Overview

ZENIN is built on a **hexagonal architecture** with clean separation between domain logic and infrastructure:

```
┌─────────────────────────────────────────┐
│           Domain Layer                  │
│  (Entities, Value Objects, Ports)       │
│  Zero external dependencies             │
├─────────────────────────────────────────┤
│         Application Layer               │
│  (Use Cases, DTOs, Orchestration)       │
├─────────────────────────────────────────┤
│        Infrastructure Layer             │
│  (Engines, Adapters, ML Models)         │
├─────────────────────────────────────────┤
│         Interface Layer                 │
│  (FastAPI, MQTT, Redis Streams)       │
└─────────────────────────────────────────┘
```

**Design principles:**
- **No magic numbers** — all thresholds are runtime-configurable via environment variables
- **Hot-reload flags** — change behavior without restarting the service
- **Circuit breakers** — graceful degradation when dependencies fail
- **Correlation IDs** — full traceability across distributed calls
- **Test-driven** — 1400+ tests, hexagonal architecture enables mocking at port boundaries

## How It Works

### Cognitive Pipeline

```
PERCEIVE → PREDICT → ADAPT → INHIBIT → FUSE → EXPLAIN → DECIDE → ACT
```

| Phase | What happens |
|-------|-------------|
| **PERCEIVE** | Analyze signal structure — classify regime, measure noise, detect trends |
| **PREDICT** | Run all capable engines concurrently (Taylor, seasonal FFT, statistical, baseline) |
| **ADAPT** | Retrieve per-regime weights from plasticity tracker (Redis) |
| **INHIBIT** | Suppress engines with high recent error or low confidence |
| **FUSE** | Hampel-filter outliers, then weighted consensus — selected engine gets primary weight |
| **EXPLAIN** | Build reasoning trace: regime, contributions, confidence |
| **DECIDE** | Map prediction to recommended action with business impact |
| **ACT** | Execute through safety guardrails (AUTO / ASK / DENY) |

**Assembly** consolidates everything into a single `cognitive_trace` inside the final `PredictionResult`.

### Regime Types

Regimes are defined as `RegimeType` enum in `domain/entities/series/structural_analysis.py`:
`STABLE`, `TRENDING`, `VOLATILE`, `NOISY`, `TRANSITIONAL`.

---

## Use Cases

| Domain | Application |
|--------|------------|
| **IoT / Infrastructure** | Predictive maintenance, threshold alerting, anomaly detection on sensor streams |
| **Cybersecurity** | Behavioral baselining, regime-shift detection, automated response with guardrails |
| **Operational Monitoring** | Real-time KPI tracking, SLA breach prediction, automated ticket creation |
| **Document Intelligence** | Crisis report analysis, urgency classification, entity extraction from unstructured text |

## Example Output

### Prediction Response

```json
{
  "predicted_value": 87.2,
  "confidence": 0.85,
  "trend": "rising",
  "metadata": {
    "cognitive_diagnostic": {
      "regime": "TRENDING",
      "final_weights": {"taylor": 0.7, "baseline": 0.3}
    },
    "cognitive_trace": {
      "drift_score": 0.15,
      "circuit_breaker_status": "closed",
      "amnesic_mode": false
    },
    "explanation": {
      "series_id": "sensor_42",
      "signal": {"regime": "TRENDING", "slope": 0.82},
      "outcome": {"confidence": 0.85, "trend": "rising"}
    }
  }
}
```

### Anomaly Detection Response

```json
{
  "anomalies": [
    {
      "index": 4,
      "value": 92.5,
      "severity": "WARNING",
      "methods": ["z_score", "iqr"],
      "confidence": 0.78
    }
  ],
  "series_stats": {
    "mean": 85.46,
    "std": 3.92,
    "regime": "TRENDING"
  }
}
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI, Uvicorn |
| **ML / Math** | NumPy, scikit-learn (optional) |
| **State** | Redis (streams, caching, sliding windows) |
| **Persistence** | SQL Server |
| **Deployment** | Docker, Kubernetes-ready |
| **Observability** | Prometheus metrics, structured JSON logging |

---

## Learning & Adaptation

### How Plasticity Works

Plasticity is **regime-contextual weight learning** — the system learns which engines perform best in specific signal regimes.

**Mechanism:**
1. **Regime detection:** SignalAnalyzer classifies into STABLE, TRENDING, VOLATILE, NOISY, TRANSITIONAL
2. **Error tracking:** After each prediction, `record_actual()` computes |predicted - actual|
3. **Bayesian update:** Inverse error updates Gaussian prior for (regime, engine) pair
4. **Weight computation:** Weights = normalized posterior means
5. **TTL decay:** Unused regimes decay to uniform weights over time

### Key parameters (configurables via feature flags)

All plasticity parameters are now runtime-configurable without redeployment.
Set them as environment variables or in your `.env` file:

| Flag | Default | Controls |
|------|---------|----------|
| `ML_PLASTICITY_ALPHA` | `0.15` | Base EMA smoothing factor |
| `ML_PLASTICITY_REGIME_ALPHAS` | JSON string | Per-regime learning rates |
| `ML_PLASTICITY_MIN_WEIGHT` | `0.05` | Floor to prevent total suppression |
| `ML_PLASTICITY_MAX_REGIMES` | `10` | LRU eviction threshold |
| `ML_PLASTICITY_REGIME_TTL_SECONDS` | `86400.0` | Unused regime decay (1 day) |
| `ML_PLASTICITY_NOISE_THRESHOLD` | `0.3` | Noise penalty activation |
| `ML_PLASTICITY_LR_FACTORS` | JSON string | Per-regime LR multipliers |

**JSON flag example** — override regime learning rates at runtime:
```bash
export ML_PLASTICITY_REGIME_ALPHAS='{"STABLE":0.10,"TRENDING":0.20,"VOLATILE":0.35,"NOISY":0.05}'
export ML_PLASTICITY_LR_FACTORS='{"STABLE":1.0,"TRENDING":1.2,"VOLATILE":1.5,"NOISY":0.8}'
```

Hot-reload: every call to `_get_flags()` reads the current singleton.
Changes take effect on the next prediction without restarting the service.

### Why Redis is Used

**Problem:** Multi-worker deployments need shared plasticity state

**Solution:** Redis-backed shared plasticity
- **Writes:** Each `update()` writes to Redis hash `plasticity:{regime}`
- **Reads:** `get_weights()` reads from Redis with 60s local cache
- **Consistency:** All workers see same weights for same regime
- **Performance:** Local cache prevents Redis round-trips

### Redis key registry

All Redis key patterns are centralized in `infrastructure/redis_keys.py`.
Never hardcode key strings — always use `RedisKeys`:

```python
from infrastructure.redis_keys import RedisKeys

# Plasticity weights
key = RedisKeys.plasticity("STABLE")          # → "plasticity:STABLE"

# Error history
key = RedisKeys.error_history("s42", "taylor") # → "error_history:s42:taylor"

# Anomaly tracking
key = RedisKeys.anomaly_track("s42")           # → "anomaly_track:s42"
key = RedisKeys.anomaly_consecutive("s42")     # → "anomaly_consecutive:s42"

# Alert suppression
key = RedisKeys.last_alert("s42")              # → "last_alert:s42"
key = RedisKeys.suppressed("s42")              # → "suppressed:s42"
```

Changing a key pattern requires editing only `RedisKeys` — all consumers
update automatically. This enables full key auditability (ISO 27001 A.12.4.1).

### How the System "Learns" Over Time

**Short-term (within session):**
- In-memory accuracy tracking per regime
- Immediate weight adjustment after each prediction

**Long-term (across restarts):**
- Optional repository persistence (SQL Server)
- Batched writes every 10 updates (performance optimization)
- State reload on initialization

**Example Learning Curve:**
```
Hour 0 (cold start):
  STABLE: {taylor: 0.5, baseline: 0.5}

Hour 1 (after 100 predictions):
  STABLE: {taylor: 0.72, baseline: 0.28}  # Taylor better in stable

Hour 2 (pattern change to TRENDING):
  TRENDING: {taylor: 0.45, statistical: 0.55}  # Statistical better for trends
```

---

## Project Structure

```
iot_machine_learning/
├── domain/                              # Pure business logic (no I/O)
│   ├── entities/
│   │   ├── results/                     # Prediction, AnomalyResult
│   │   ├── series/                      # StructuralAnalysis, SeriesProfile, RegimeType (enum)
│   │   ├── explainability/              # Explanation, ReasoningTrace
│   │   ├── patterns/                    # PatternResult, ChangePoint
│   │   └── iot/                         # Legacy SensorReading
│   ├── ports/                           # Abstract interfaces
│   │   ├── prediction_port.py
│   │   ├── storage_port.py
│   │   ├── audit_port.py
│   │   └── anomaly_detection_port.py
│   ├── services/                        # Domain orchestration
│   │   ├── prediction_domain_service.py
│   │   ├── anomaly_domain_service.py
│   │   ├── memory_recall_enricher.py
│   │   └── cognitive_constants.py      # Shared constants (CONFIDENCE_REDUCTION_*)
│   ├── tools/                           # Tool system
│   │   ├── tool.py                      # Tool ABC
│   │   ├── tool_base.py                 # Base implementation
│   │   ├── tool_registry.py             # Central registry
│   │   ├── tool_executor.py             # Execution logic
│   │   ├── tool_guard.py                # SafetyLevel, GuardResult
│   │   ├── tool_metrics.py              # Execution tracking
│   │   └── iot_tools.py                 # Concrete tools
│   └── validators/                      # Input validation
│
├── infrastructure/                      # Concrete implementations
│   ├── redis_keys.py                   # Centralized Redis key patterns
│   ├── ml/
│   │   ├── engines/                     # Prediction engines
│   │   │   ├── taylor/                  # Taylor polynomial
│   │   │   ├── seasonal/                # FFT-based cycle detection
│   │   │   ├── statistical/             # Statistical predictors
│   │   │   ├── baseline/                # Moving average heuristics
│   │   │   ├── ensemble/                # Model ensemble
│   │   │   └── core/                    # Engine factory, base types
│   │   ├── anomaly/                     # Anomaly detection
│   │   │   ├── voting_anomaly_detector.py
│   │   │   └── detectors/               # Sub-detector implementations
│   │   ├── cognitive/                   # Cognitive pipeline
│   │   │   ├── orchestration/
│   │   │   │   ├── orchestrator.py      # MetaCognitiveOrchestrator
│   │   │   │   ├── pipeline_executor.py # execute_pipeline()
│   │   │   │   ├── iterative_controller.py # CognitiveLoopController
│   │   │   │   └── weight_resolution_service.py
│   │   │   ├── analysis/
│   │   │   │   ├── signal_analyzer.py   # Regime detection
│   │   │   │   └── types.py             # MetaDiagnostic, PipelineTimer
│   │   │   ├── universal/                 # Universal Analysis Engine (NEW 0.3.0)
│   │   │   │   ├── analysis/
│   │   │   │   │   ├── engine.py        # UniversalAnalysisEngine
│   │   │   │   │   ├── types.py         # UniversalResult, UniversalInput
│   │   │   │   │   ├── pipeline/        # 5-phase pipeline
│   │   │   │   │   │   ├── perceive.py  # Input detection, domain classification
│   │   │   │   │   │   ├── analyze.py   # Perception collection
│   │   │   │   │   │   ├── remember.py  # Cognitive memory recall
│   │   │   │   │   │   ├── reason.py    # Fusion, inhibition, severity
│   │   │   │   │   │   └── explain.py   # Narrative generation
│   │   │   │   │   ├── perception_collector.py
│   │   │   │   │   ├── input_detector.py
│   │   │   │   │   └── monte_carlo.py   # Uncertainty quantification
│   │   │   │   ├── text/                # Text analysis components
│   │   │   │   │   ├── severity_mapper.py    # Urgency-aware severity
│   │   │   │   │   ├── impact_detector.py    # Hard signal detection
│   │   │   │   │   ├── pattern_interpreter.py # Pattern interpretation
│   │   │   │   │   └── analyzers/       # Text analyzers (urgency, sentiment)
│   │   │   │   └── comparative.py       # Comparative engine
│   │   │   ├── plasticity/
│   │   │   │   └── base.py              # PlasticityTracker
│   │   │   ├── fusion/
│   │   │   │   └── weighted_fusion.py   # WeightedFusion
│   │   │   ├── inhibition/
│   │   │   │   └── inhibition_gate.py   # InhibitionGate
│   │   │   ├── explanation/
│   │   │   │   └── explanation_builder.py
│   │   │   └── decision/
│   │   │       └── decision_executor.py
│   │   └── interfaces.py                # PredictionEngine ABC
│   ├── adapters/                        # Port implementations
│   │   ├── sqlserver_storage.py
│   │   ├── redis_cache.py
│   │   └── weaviate_cognitive_adapter.py
│   └── persistence/                     # Database connections
│
├── application/                         # Use cases
│   ├── use_cases/
│   │   ├── predict_sensor_value.py      # PredictSensorValueUseCase
│   │   ├── analyze_document.py          # AnalyzeDocumentUseCase (NEW 0.3.0)
│   │   └── detect_anomalies.py          # DetectAnomaliesUseCase
│   ├── dto/
│   │   ├── prediction_dto.py
│   │   ├── text_decision_output.py      # Document analysis DTOs
│   │   └── anomaly_dto.py
│   └── explainability/
│       └── explanation_renderer.py
│
├── ml_service/                          # FastAPI service
│   ├── api/
│   │   ├── routes.py                    # HTTP endpoints
│   │   ├── schemas.py                   # Pydantic models
│   │   └── services/
│   │       ├── prediction_service.py    # Numeric predictions
│   │       ├── document_analyzer.py     # Document analysis (NEW 0.3.0)
│   │       └── analysis/                # Universal analysis components
│   │           ├── universal_bridge.py      # Bridge to Universal Engine
│   │           ├── conclusion_formatter.py  # Rich output formatting
│   │           ├── entity_extractor.py      # Entity extraction
│   │           ├── decision_engine_service.py
│   │           └── document_analyzer_factory.py
│   ├── config/
│   │   ├── feature_flags.py             # Main facade
│   │   ├── flags.py                     # FeatureFlags model
│   │   └── loader.py                    # Environment loading
│   ├── consumers/
│   │   └── stream_consumer.py           # Redis Streams consumer
│   ├── workers/
│   │   └── zenin_queue_poller.py        # Queue-based text processing
│   ├── broker/
│   │   └── broker_factory.py            # Redis broker
│   └── main.py                          # FastAPI entry point
│
└── tests/                               # Test suite
    ├── unit/
    ├── integration/
    └── e2e/
```

---

## API Reference

### Prediction

**POST** `/api/v1/predict`

```json
// Request
{
  "series_id": "sensor_42",
  "values": [82.1, 83.5, 84.2, 85.0, 85.5],
  "timestamps": [1741234500, 1741234515, 1741234530, 1741234545, 1741234560],
  "threshold": 90.0
}
```

### Anomaly Detection

**POST** `/api/v1/detect-anomalies`

```json
// Request
{
  "series_id": "sensor_42",
  "values": [82.1, 83.5, 84.2, 85.0, 92.5],
  "timestamps": [1741234500, 1741234515, 1741234530, 1741234545, 1741234560]
}
```

### Document Analysis

**POST** `/api/v1/analyze-document`

```json
// Request
{
  "document_id": "crisis_report_001",
  "content": "ALERTA: Rack B-07 temperatura 94°C (límite 80°C). Servidores offline. 77% capacidad perdida.",
  "tenant_id": "acme_corp",
  "content_type": "text"
}
```

**Response:**
```json
{
  "document_id": "crisis_report_001",
  "tenant_id": "acme_corp",
  "classification": "infrastructure",
  "severity": "critical",
  "confidence": 0.65,
  "entities": ["94°C", "80°C"],
  "actions": [
    "Restart affected node immediately",
    "Check sensor readings and thresholds",
    "Reduce system load to prevent cascade failure"
  ],
  "processing_time_ms": 145.2
}
```
---

## Configuration

### Feature Flags Reference

Flags are loaded via `get_feature_flags()` on every call (hot-reload).
All flags read from environment variables. Safe defaults are conservative
(features off, low resource usage). Set in `.env` or export directly.

#### Core toggles
| Flag | Default | Description |
|------|---------|-------------|
| `ML_ROLLBACK_TO_BASELINE` | `false` | Panic button — force all to baseline |
| `ML_USE_TAYLOR_PREDICTOR` | `false` | Enable Taylor engine |
| `ML_USE_KALMAN_FILTER` | `false` | Enable Kalman filtering |
| `ML_ENABLE_AB_TESTING` | `false` | Compare baseline vs Taylor |
| `ML_ENABLE_PLASTICITY` | `false` | Enable weight learning |
| `ML_ENABLE_ITERATIVE` | `false` | Enable cognitive loop |
| `ML_ENABLE_COGNITIVE_MEMORY` | `false` | Weaviate integration |
| `ML_ENABLE_MEMORY_RECALL` | `false` | Query historical explanations |
| `ML_ENABLE_PREDICTION_CACHE` | `false` | Cache predictions |
| `ML_ENABLE_VOTING_ANOMALY` | `false` | Voting anomaly detection |
| `ML_ENABLE_CHANGE_POINT_DETECTION` | `false` | Change point detection |
| `ML_ENABLE_EXPLAINABILITY` | `false` | Explanation builder |
| `ML_ENABLE_ENSEMBLE_PREDICTOR` | `false` | Model ensemble |
| `ML_ENABLE_DELTA_SPIKE_DETECTION` | `false` | Delta spike detection |
| `ML_ENABLE_REGIME_DETECTION` | `false` | Regime detection |
| `ML_ENABLE_AUDIT_LOGGING` | `false` | Audit logging |

#### Plasticity tuning
| Flag | Default | Description |
|------|---------|-------------|
| `ML_PLASTICITY_ALPHA` | `0.15` | Base EMA smoothing factor |
| `ML_PLASTICITY_MIN_WEIGHT` | `0.05` | Floor to prevent total suppression |
| `ML_PLASTICITY_MAX_REGIMES` | `10` | LRU eviction threshold |
| `ML_PLASTICITY_REGIME_TTL_SECONDS` | `86400.0` | Unused regime decay (1 day) |
| `ML_PLASTICITY_NOISE_THRESHOLD` | `0.3` | Noise penalty activation |
| `ML_PLASTICITY_PERSIST_EVERY_N` | `10` | Batch writes every N updates |
| `ML_PLASTICITY_IMMEDIATE_PERSIST_THRESHOLD` | `0.15` | Accuracy change threshold for immediate persist |
| `ML_PLASTICITY_REDIS_CACHE_TTL_SECONDS` | `60.0` | Redis cache TTL |
| `ML_PLASTICITY_REGIME_ALPHAS` | JSON | Per-regime learning rates |
| `ML_PLASTICITY_LR_FACTORS` | JSON | Per-regime LR multipliers |

#### Decision engine tuning
| Flag | Default | Description |
|------|---------|-------------|
| `ML_ENABLE_DECISION_ENGINE` | `true` | Enable decision engine |
| `ML_DECISION_ENGINE` | `simple` | Engine type: simple/contextual/conservative/aggressive/cost_optimized |
| `ML_DECISION_CONSERVATIVE_THRESHOLD` | `0.8` | Confidence threshold for "intervene" |
| `ML_DECISION_CONSERVATIVE_SAFETY_MARGIN` | `1.2` | Risk multiplier for worst-case analysis |
| `ML_DECISION_CONFIDENCE_FLOOR` | `0.6` | Minimum confidence |
| `ML_DECISION_CONFIDENCE_CEILING` | `0.95` | Maximum confidence |
| `ML_DECISION_ESCALATION_THRESHOLD` | `5` | Consecutive anomalies for escalation |
| `ML_DECISION_ATT_STABLE_DRIFT_THRESHOLD` | `0.10` | Drift threshold for stable attenuator |
| `ML_DECISION_CONFIDENCE_REDUCTION_SPARSE` | `0.9` | Confidence reduction with sparse evidence |
| `ML_DECISION_BASE_SCORES` | JSON | Severity to score mappings |
| `ML_DECISION_AMP_THRESHOLDS` | JSON | Amplifier thresholds |
| `ML_DECISION_AMP_CONSECUTIVE_5` | `1.35` | Multiplier for 5+ consecutive anomalies |
| `ML_DECISION_AMP_CONSECUTIVE_3` | `1.20` | Multiplier for 3+ consecutive anomalies |
| `ML_DECISION_AMP_RATE_HIGH` | `1.20` | Multiplier for high anomaly rate |
| `ML_DECISION_AMP_RATE_MED` | `1.10` | Multiplier for medium anomaly rate |
| `ML_DECISION_AMP_VOLATILE` | `1.15` | Multiplier for volatile regime |
| `ML_DECISION_AMP_NOISY` | `1.10` | Multiplier for noisy regime |
| `ML_DECISION_AMP_DRIFT_HIGH` | `1.20` | Multiplier for high drift |
| `ML_DECISION_AMP_DRIFT_MED` | `1.10` | Multiplier for medium drift |
| `ML_DECISION_ATT_STABLE` | `0.85` | Attenuator for stable regime |
| `ML_DECISION_ATT_LOW_CRITICALITY` | `0.80` | Attenuator for low criticality |
| `ML_DECISION_ATT_NO_CONTEXT` | `0.90` | Attenuator for no recent context |
| `ML_DECISION_SUPPRESSION_WINDOW_MINUTES` | `5.0` | Alert suppression window |
| `ML_DECISION_THRESHOLD_ESCALATE` | `0.85` | Score threshold to escalate |
| `ML_DECISION_THRESHOLD_INVESTIGATE` | `0.65` | Score threshold to investigate |
| `ML_DECISION_THRESHOLD_MONITOR` | `0.40` | Score threshold to monitor |

#### Anomaly tracking
| Flag | Default | Description |
|------|---------|-------------|
| `ML_ANOMALY_TTL_SECONDS` | `7200.0` | Anomaly entry TTL (2 hours) |
| `ML_ANOMALY_MAX_ENTRIES_PER_SERIES` | `500` | Max entries per series |
| `ML_ANOMALY_KEY_TTL_SECONDS` | `3600` | Redis key TTL (1 hour) |
| `ML_ANOMALY_TRACKER_BACKEND` | `memory` | Backend: memory or redis |
| `ML_ANOMALY_VOTING_THRESHOLD` | `0.5` | Voting threshold |
| `ML_ANOMALY_CONTAMINATION` | `0.1` | Contamination factor |

#### Performance & streaming
| Flag | Default | Description |
|------|---------|-------------|
| `ML_BATCH_MAX_WORKERS` | `4` | Parallel batch workers |
| `ML_BATCH_CIRCUIT_BREAKER_THRESHOLD` | `10` | Circuit breaker threshold |
| `ML_STREAM_USE_SLIDING_WINDOW` | `true` | Use in-memory windows |
| `ML_ENTERPRISE_USE_PRELOADED_DATA` | `true` | Reduce DB queries |
| `ML_MQTT_ASYNC_PROCESSING` | `true` | Enable async MQTT processing |
| `ML_MQTT_QUEUE_SIZE` | `1000` | Max queue depth |
| `ML_MQTT_NUM_WORKERS` | `4` | ThreadPool workers |
| `ML_CACHE_TTL_SECONDS` | `60` | Cache TTL |
| `ML_CACHE_MAX_ENTRIES` | `1000` | Max cache entries |
| `ML_SLIDING_WINDOW_MAX_SENSORS` | `1000` | LRU eviction threshold |
| `ML_SLIDING_WINDOW_TTL_SECONDS` | `3600` | TTL eviction (1 hour) |

#### Circuit breaker & infrastructure
| Flag | Default | Description |
|------|---------|-------------|
| `ML_INGEST_CIRCUIT_BREAKER_ENABLED` | `true` | Enable circuit breaker |
| `ML_INGEST_CB_FAILURE_THRESHOLD` | `5` | Failure threshold |
| `ML_INGEST_CB_TIMEOUT_SECONDS` | `30` | Timeout in seconds |
| `ML_PIPELINE_BUDGET_MS` | `500` | Pipeline budget in ms |
| `ML_COHERENCE_CHECK_ENABLED` | `false` | Prediction coherence check |
| `ML_DOMAIN_BOUNDARY_ENABLED` | `false` | Validate input domain |
| `ML_ACTION_GUARD_ENABLED` | `false` | Enable action guards |
| `ML_DECISION_ARBITER_ENABLED` | `false` | Enable decision arbiter |
| `ML_CONFIDENCE_CALIBRATION_ENABLED` | `false` | Confidence calibration |
| `ML_NARRATIVE_UNIFICATION_ENABLED` | `false` | Narrative unification |
| `ML_PREDICT_MAX_WORKERS` | `3` | ThreadPool workers for concurrent engine execution |
| `ML_PREDICT_ENGINE_TIMEOUT_MS` | `400` | Per-engine timeout (ms) within 500ms pipeline budget |
| `ML_HAMPEL_ENABLED` | `true` | Outlier rejection before weighted fusion |
| `ML_HAMPEL_K` | `3.0` | Hampel filter sensitivity (≈3σ Gaussian) |

### JSON Flags (Dict-Type Parameters)

Some flags accept JSON strings to configure dictionaries at runtime:

```bash
# Override all severity→score mappings for the decision engine
export ML_DECISION_BASE_SCORES='{"critical":0.95,"high":0.75,"medium":0.50,"low":0.25,"info":0.05,"warning":0.50}'

# Override amplifier thresholds
export ML_DECISION_AMP_THRESHOLDS='{"count_high":3,"count_medium":2,"ratio_high":0.55,"ratio_low":0.25}'

# Override plasticity regime alphas
export ML_PLASTICITY_REGIME_ALPHAS='{"STABLE":0.08,"TRENDING":0.18,"VOLATILE":0.30,"NOISY":0.05}'
```

JSON flags are parsed with `json.loads()` on each `_get_flags()` call.
Invalid JSON falls back to the hardcoded default silently — validate before deploying.

### Backpressure & Limits

```bash
# MQTT Async Processing
export ML_MQTT_ASYNC_PROCESSING=true           # Enable async processing
export ML_MQTT_QUEUE_SIZE=1000                 # Max queue depth
export ML_MQTT_NUM_WORKERS=4                   # ThreadPool workers

# Sliding Window Limits
export ML_SLIDING_WINDOW_MAX_SENSORS=1000      # LRU eviction threshold
export ML_SLIDING_WINDOW_TTL_SECONDS=3600      # TTL eviction (1 hour)

# Circuit Breaker
export ML_INGEST_CIRCUIT_BREAKER_ENABLED=true
export ML_INGEST_CB_FAILURE_THRESHOLD=5
export ML_INGEST_CB_TIMEOUT_SECONDS=30
```

### Panic Button

```bash
# Force all predictions to baseline engine, bypass cognitive pipeline
export ML_ROLLBACK_TO_BASELINE=true
```

### Architecture Decisions: Why Flags Over Constants

| Decision | Rationale |
|----------|-----------|
| **All numeric thresholds in flags** | Adjustable in <5 min without code change (ISO 27001 A.12.1.2) |
| **JSON flags for dicts** | Compatible with env vars in Kubernetes/docker-compose |
| **`_get_flags()` on every call** | Hot-reload: changes propagate without restart |
| **`RegimeType` enum** | Eliminates magic strings across 10+ files |
| **`RedisKeys` registry** | Single point of change for key patterns; enables access auditing |
| **Fallback to `FeatureFlags()`** | Service stays alive even if config system fails |

---

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

### Performance Tuning

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

---

## Development Guide

### Adding a New Prediction Engine

```python
from infrastructure.ml.interfaces import PredictionEngine

class MyCustomEngine(PredictionEngine):
    @property
    def name(self) -> str:
        return "my_custom"
    
    def predict(self, window: TimeSeriesWindow) -> PredictionResult:
        # Your prediction logic here
        return PredictionResult(
            predicted_value=self._compute(window),
            confidence=0.8,
            trend="stable"
        )
```

Register in orchestrator:

```python
orchestrator = MetaCognitiveOrchestrator(
    engines=[TaylorEngine(), BaselineEngine(), MyCustomEngine()],
    enable_plasticity=True
)
```

### Adding a New Tool

```python
from domain.tools.tool import Tool
from domain.tools.tool_guard import SafetyLevel

class MyCustomTool(Tool):
    @property
    def name(self) -> str:
        return "custom_action"
    
    @property
    def parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            }
        }
    
    def can_execute(self, context) -> GuardResult:
        # Return AUTO, ASK, or DENY
        return GuardResult(SafetyLevel.AUTO)
    
    def execute(self, params, context):
        # Execution logic
        return ToolResult(success=True, data={})
```

---

## Current Limitations

### Known Constraints

**Heuristic Tool Mapping:**
- DecisionEngine uses rule-based mapping from predictions to tool calls
- No learned policy optimization (e.g., RL) yet
- Mapping rules are hardcoded per domain

**Limited Refinement Strategies:**
- Iterative loop has placeholder `_refine()` method
- Currently returns input unchanged
- Future: window expansion, outlier removal, smoothing

**No Full Reinforcement Learning:**
- Plasticity uses Bayesian updates (supervised-style)
- No trial-and-error learning from action outcomes
- No exploration vs exploitation tradeoff

**Single-Node Plasticity:**
- Redis shared state works for multi-worker
- No distributed consensus for weight updates
- Edge deployment requires state synchronization

**Anomaly Integration:**
- Anomaly detection runs separate from prediction pipeline
- No unified narrative between prediction and anomaly
- Narrative unification is placeholder-only

### Performance Limits

| Metric | Tested Limit | Theoretical Limit |
|--------|---------------|-------------------|
| Sensors | 1,000 | 10,000 (with sharding) |
| Latency (p99) | 150ms | 50ms (with caching) |
| Throughput | 10K msgs/sec | 100K (with batching) |
| Plasticity regimes | 10 | 100 (with LRU tuning) |

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **Scalability** | Horizontal scaling, partitioned sliding windows, async DB writes | Planned |
| **Distributed Learning** | Federated plasticity, model compression for edge, A/B testing | Planned |
| **Causal Reasoning** | DoWhy integration, counterfactual explanations | Future |
| **Natural Language Interface** | Text commands to tool execution | Future |

For detailed design decisions, see `ARCHITECTURE.md`. For legacy API migration status, see `MIGRATION_SCORECARD.md`.

---

## License

Internal ZENIN project. All rights reserved.
