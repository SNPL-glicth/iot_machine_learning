# ZENIN Cognitive IoT Analytics Engine — GOLD 0.2.1

**Version:** 0.2.1-GOLD | **Tests:** ~1260 passed | **Architecture:** Hexagonal + Cognitive  
**Status:** Production-Ready | **License:** Internal ZENIN Project

---

## 🎯 What is ZENIN?

ZENIN transforms raw IoT sensor data into **intelligent decisions** through a transparent, multi-phase cognitive pipeline. Unlike traditional ML systems that are black boxes, ZENIN provides full explainability of every prediction and action.

### The Cognitive Pipeline

```
PERCEIVE → PREDICT → ADAPT → INHIBIT → FUSE → EXPLAIN → DECIDE → ACT
```

- **PERCEIVE:** Analyze signal structure (regime, noise, trends)
- **PREDICT:** Multi-engine forecasting (Taylor, Baseline, Statistical)
- **ADAPT:** Learn which engine works best in each context (Plasticity)
- **INHIBIT:** Suppress unreliable engines
- **FUSE:** Weighted consensus prediction
- **EXPLAIN:** Build reasoning trace
- **DECIDE:** Map to recommended action with business impact
- **ACT:** Execute with safety guardrails (AUTO/ASK/DENY)

### Key Differentiators

| Feature | Traditional ML | ZENIN |
|---------|-----------------|-------|
| **Reasoning** | Black box model | Transparent cognitive phases |
| **Learning** | Retrain models | Real-time weight adaptation |
| **Safety** | Post-hoc monitoring | Built-in guard system |
| **Actions** | Manual interpretation | Autonomous with approval gates |
| **Explainability** | SHAP/LIME approximations | Full trace per prediction |

---

## 🚀 Quick Start

### 1. Installation

```bash
cd /home/nicolas/Documentos/Iot_System/iot_machine_learning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Dependencies

```bash
# Redis (required for streaming + plasticity)
docker run -d -p 6379:6379 redis:7-alpine

# SQL Server (required for persistence)
docker run -d -p 1434:1434 \
  -e SA_PASSWORD=YourPassword123 \
  -e ACCEPT_EULA=Y \
  mcr.microsoft.com/mssql/server:2022-latest
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 4. Run Service

```bash
# Start ML API service
uvicorn ml_service.main:app --host 0.0.0.0 --port 8002 --reload

# Verify
curl http://localhost:8002/
# {"service": "iot-ml-service", "version": "0.2.1-GOLD", "status": "ok"}
```

---

## 📊 GOLD 0.2.1 Release Highlights

### Critical Bug Fixes

| Issue | Fix | Impact |
|-------|-----|--------|
| **Storage initialization** | `orchestrator.py` now properly initializes `_storage` | Prevents AttributeError on persistence operations |
| **Confidence inversion** | `plasticity/base.py` corrected accuracy formula to `1.0 / (1.0 + abs(error))` | Weights now correctly reflect engine performance |
| **Race conditions** | `_last_diagnostic` now protected with `_state_lock` | Thread-safe diagnostic access |
| **Backward compat** | Removed broken `_weight_service`/`_weight_mediator` properties | Cleaner API, no crashes |

### Performance Optimizations

- **NumPy import moved to top-level:** ~100μs latency reduction per prediction
- **Redis pipeline support:** Batch weight updates for multi-engine scenarios
- **Circuit breaker integration:** Automatic fallback to "Modo Amnésico" (RAM-only) on persistence failures

### New Features

- **Cognitive Trace Metadata:** Unified `cognitive_trace` field in PredictionResult combining:
  - `drift_score`: Concept drift detection
  - `shadow_performance`: Shadow evaluation results
  - `circuit_breaker_status`: "closed"/"open"/"half_open"
  - `amnesic_mode`: RAM-only fallback indicator

- **Deprecated sensor_id:** All `sensor_id: int` references marked as `@deprecated`. Use `series_id: str` instead.

---

## 2. Key Capabilities

### 2.1 Cognitive Reasoning (Iterative Loop)
The system can execute multiple prediction passes until confidence threshold is met:
- **Max iterations:** Configurable (default: 3)
- **Confidence threshold:** Target confidence to stop iterating (default: 0.85)
- **Time budget:** Hard limit to prevent runaway loops (default: 5000ms)
- **Best-result tracking:** Keeps highest confidence result across iterations

### 2.2 Deterministic Predictions
Multi-engine prediction with weighted fusion:
- **Engines:** Baseline (moving average), Taylor (polynomial extrapolation), Ensemble, Statistical
- **Fusion:** Weighted average with inhibition-based suppression
- **Inhibition:** Unstable engines get weight reduced (min floor: 0.05)
- **Selection:** Best engine chosen per regime based on historical accuracy

### 2.3 Action Execution via Tools
Typed tool system replaces string-based actions:
- `send_alert` — Notifications to operators
- `adjust_threshold` — Dynamic sensor threshold adjustment
- `request_maintenance` — Create maintenance tickets

### 2.4 Real-Time Processing
- **Redis Streams:** Message broker for high-throughput ingestion
- **Sliding Windows:** Per-sensor state with LRU+TTL eviction
- **Async Processing:** ThreadPool-based message handling (configurable workers)
- **Backpressure:** Queue size limits prevent memory exhaustion

### 2.5 Safety & Guard System
Three-level guard system for all autonomous actions:
- **AUTO:** Execute without human intervention
- **ASK:** Request human approval before execution
- **DENY:** Block execution entirely

**Safety Rules:**
- CRITICAL severity → AUTO (immediate action needed)
- WARNING severity → ASK (important but not urgent)
- Low confidence (< 0.6) → ASK
- Too frequent adjustments (>3/hour) → DENY

---

## 3. Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL INTERFACES                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   MQTT/HTTP  │  │  Ingest API  │  │   WebSocket  │  │   Prometheus │    │
│  │   Ingestion  │  │   (REST)     │  │   (Realtime) │  │   /metrics   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REDIS LAYER (State & Messaging)                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Streams        │  Plasticity       │  Sliding        │  Circuit      │   │
│  │  (ingestion)    │  (weights)        │  Windows        │  Breaker      │   │
│  │                 │                   │                 │               │   │
│  │  readings:raw   │  plasticity:*     │  window:{id}    │  state:{svc}  │   │
│  │  readings:proc   │  (per-regime)     │  (per-series)   │               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COGNITIVE PIPELINE (Phase Execution)                    │
│                                                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │PERCEIVE │──▶│ PREDICT │──▶│  ADAPT  │──▶│ INHIBIT │──▶│  FUSE   │     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│       │            │            │            │            │                │
│       ▼            ▼            ▼            ▼            ▼                │
│  Signal        Engine         Weight       Inhibition   Weighted          │
│  Analysis      Perception     Resolution   Gate         Fusion            │
│                                                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐                       │
│  │ EXPLAIN │──▶│ DECIDE  │──▶│  GUARD  │──▶│   ACT   │                       │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘                       │
│       │            │            │            │                               │
│       ▼            ▼            ▼            ▼                               │
│  Reasoning     Decision       Safety       Tool                              │
│  Builder       Engine         Check        Execution                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ASSEMBLY PHASE (GOLD)                            │   │
│  │  • Cognitive trace (drift, shadow, circuit breaker)                 │   │
│  │  • Metadata consolidation                                           │   │
│  │  • Confidence interval computation                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PERSISTENCE LAYER (GOLD Features)                      │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐           │
│  │   SQL Server     │  │    Redis         │  │   Hybrid         │           │
│  │   (primary)      │  │   (cache)        │  │   (adaptive)     │           │
│  │                  │  │                  │  │                  │           │
│  │  • predictions   │  │  • weights       │  │  • Auto-failover │           │
│  │  • anomalies     │  │  • windows       │  │  • Mode switch   │           │
│  │  • audit_logs    │  │  • circuit       │  │  • Amnesic mode  │           │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Layer Architecture (Hexagonal)

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│  Use Cases, DTOs, API Routes, Consumers                     │
│  (FastAPI, Redis Streams, Background Jobs)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     DOMAIN LAYER (Pure)                     │
│  Entities, Value Objects, Domain Services, Ports            │
│  (Zero I/O dependencies, 100% testable)                   │
│                                                             │
│  Entities: Prediction, Anomaly, Explanation, Tool           │
│  Services: PredictionService, AnomalyService                │
│  Ports: PredictionPort, StoragePort, AuditPort               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                       │
│  Adapters implementing Domain Ports                         │
│                                                             │
│  ML: Engines, Anomaly Detectors, Cognitive Pipeline        │
│  Persistence: SQL Server, Redis, Weaviate                   │
│  Streaming: Redis Streams, MQTT                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow Architecture

**Ingestion Flow:**
```
MQTT/HTTP → Redis Stream → Consumer → Sliding Window → 
Signal Analysis → Prediction → Persistence → Action
```

**Learning Flow:**
```
Prediction Made → Record Actual → Compute Error → 
Update Weights → Redis Cache → Optional DB Persistence
```

**Query Flow:**
```
API Request → Load Window → Run Prediction → 
Apply Plasticity → Return Result + Metadata
```

### 3.2 Layer Explanation

#### Domain Layer (`domain/`)
Pure business logic with zero I/O dependencies:
- **Entities:** `Prediction`, `AnomalyResult`, `Explanation`, `StructuralAnalysis`
- **Ports:** Abstract interfaces (`PredictionPort`, `StoragePort`, `AuditPort`)
- **Services:** Domain orchestration (`PredictionDomainService`, `AnomalyDomainService`)
- **Tools:** Tool abstraction with guard system (`Tool`, `ToolRegistry`, `ToolExecutor`)

#### Infrastructure Layer (`infrastructure/`)
Concrete implementations of domain ports:
- **ML Engines:** Taylor, Baseline, Ensemble, Statistical predictors
- **Anomaly Detection:** Voting-based detector (IF, Z-score, IQR, LOF)
- **Cognitive:** `MetaCognitiveOrchestrator`, `PlasticityTracker`, `InhibitionGate`
- **Adapters:** SQL Server, Redis implementations (Weaviate adapter exists but not production-ready)

#### Application Layer (`application/`)
Use cases and DTOs:
- **Use Cases:** `PredictSensorValueUseCase`, `DetectAnomaliesUseCase`
- **DTOs:** `PredictionDTO`, `AnomalyDTO`, `ExplanationRenderer`

### 3.4 Cognitive Pipeline Stages (Detailed)

| Phase | Component | Responsibility | Key Output |
|-------|-----------|----------------|------------|
| **PERCEIVE** | `SignalAnalyzer` | Analyze signal structure (regime, noise, slope) | `MetaDiagnostic` with regime classification |
| **PREDICT** | `EnginePerception` | Collect predictions from all engines | `List[EnginePerception]` with confidence scores |
| **ADAPT** | `PlasticityTracker` | Adjust weights based on regime history | `RegimeWeights` from Redis cache |
| **INHIBIT** | `InhibitionGate` | Suppress unstable engines | `InhibitionResult` with suppressed flags |
| **FUSE** | `WeightedFusion` | Combine predictions with learned weights | `FusedPrediction` with weighted value |
| **EXPLAIN** | `ExplanationBuilder` | Build reasoning trace | `Explanation` with phase trace |
| **DECIDE** | `DecisionExecutor` | Map to tool calls | `Recommendation` with tool and guard level |
| **GUARD** | `ToolGuard` | Apply safety rules | `GuardResult` (AUTO/ASK/DENY) |
| **ACT** | `ToolExecutor` | Execute tool | `ToolResult` with execution status |
| **ASSEMBLY** | `AssemblyPhase` | Consolidate metadata | `PredictionResult` with `cognitive_trace` |

### 3.5 GOLD Assembly Phase

The Assembly Phase (new in GOLD 0.2.1) creates a unified `cognitive_trace` in the final PredictionResult:

```python
{
    "cognitive_trace": {
        "drift_score": 0.15,                    # Concept drift detection
        "shadow_performance": {                 # Shadow evaluation results
            "engines_tested": 3,
            "results": [{"engine": "experimental", "error": 0.02}],
            "sampled": True
        },
        "circuit_breaker_status": "closed",   # "closed"/"open"/"half_open"
        "amnesic_mode": False,                  # RAM-only fallback
        "assembly_timestamp": 1741234567.89
    }
}
```

---

## 4. Cognitive Pipeline Flow

### 4.1 Step-by-Step Execution

```
Input (sensor readings)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. INGEST                                                       │
│    • Parse MQTT/HTTP payload                                     │
│    • Validate and normalize                                    │
│    • Write to Redis Stream "readings:raw"                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. ANALYZE (SignalAnalyzer)                                     │
│    • Compute slope, curvature, acceleration                     │
│    • Classify regime (STABLE, TRENDING, VOLATILE, NOISY)       │
│    • Calculate noise ratio, stability indicator                 │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. PREDICT (Multi-Engine)                                       │
│    • Taylor: Polynomial extrapolation                          │
│    • Baseline: Moving average                                    │
│    • Statistical: EMA/Holt-Winters                             │
│    • Ensemble: Model averaging                                   │
│    → Returns: List[EnginePerception]                             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. FUSE (WeightedFusion)                                        │
│    • Apply plasticity weights (regime-specific)                │
│    • Inhibit unstable engines                                    │
│    • Weighted average of predictions                           │
│    • Majority-vote for trend                                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. EVALUATE (Iterative - Optional)                              │
│    • Check confidence >= threshold (default: 0.85)             │
│    • If low: refine input and retry (max 3 iterations)         │
│    • Track best result across iterations                        │
│    • Respect time budget (default: 5000ms)                     │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. DECIDE (DecisionExecutor)                                    │
│    • Map prediction to recommended action                        │
│    • Evaluate guard level (AUTO/ASK/DENY)                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. ACT (Tool System)                                            │
│    • send_alert: Notify operators                                │
│    • adjust_threshold: Update sensor limits                      │
│    • request_maintenance: Create tickets                         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Metadata Generation

Every prediction includes rich metadata:

```python
{
    "cognitive_diagnostic": {
        "signal_profile": {"regime": "STABLE", "noise_ratio": 0.02},
        "perceptions": [{"engine": "taylor", "confidence": 0.92}],
        "inhibition_states": [{"engine": "baseline", "suppressed": False}],
        "final_weights": {"taylor": 0.7, "baseline": 0.3},
        "selected_engine": "taylor"
    },
    "explanation": {
        "series_id": "sensor_42",
        "signal": {"regime": "STABLE", "slope": 0.15},
        "contributions": {"n_engines": 2, "consensus_spread": 0.8},
        "trace": {"phases": ["PERCEIVE", "PREDICT", "ADAPT", "INHIBIT", "FUSE"]},
        "outcome": {"confidence": 0.92, "trend": "stable"}
    },
    "pipeline_timing": {
        "perceive_ms": 12.5,
        "predict_ms": 45.2,
        "adapt_ms": 3.1,
        "inhibit_ms": 1.8,
        "fuse_ms": 2.3,
        "explain_ms": 0.9,
        "total_ms": 65.8
    },
    "cognitive_loop": {  # If iterative mode enabled
        "iterations_used": 2,
        "converged": True,
        "threshold": 0.85,
        "iteration_history": [
            {"iteration": 0, "confidence": 0.78},
            {"iteration": 1, "confidence": 0.92}
        ]
    }
}
```

---

## 5. Tool System

### 5.1 Why String Actions Were Replaced

**Problem:** String-based actions (`action: "alert"`) lack:
- Type safety
- Parameter validation
- Safety guardrails
- Execution history
- Testability

**Solution:** Typed tool system with:
- JSON Schema parameter definitions
- Safety level guards (AUTO/ASK/DENY)
- Execution metrics and history
- Idempotent operations

### 5.2 Tool Structure

```python
class Tool(ABC):
    @property
    def name(self) -> str: ...
    
    @property
    def parameters(self) -> Dict[str, Any]:  # JSON Schema
        ...
    
    def can_execute(self, context: ToolContext) -> GuardResult:
        # Return AUTO, ASK, or DENY with reason
        ...
    
    def execute(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        # Actual execution logic
        ...
```

### 5.3 Guard System

| Level | Condition | Example |
|-------|-----------|---------|
| **AUTO** | Critical severity | Alert when temperature > 100°C |
| **ASK** | Warning severity OR low confidence (< 0.6) | Threshold adjustment with 0.55 confidence |
| **DENY** | Rate limit exceeded OR unsafe state | >3 adjustments in 1 hour |

### 5.4 Execution Flow

```
Recommendation from DecisionEngine
    │
    ▼
ToolRegistry.lookup(tool_name)
    │
    ▼
Tool.can_execute(context) → GuardResult
    │
    ├── AUTO ──▶ Tool.execute(params, context) ──▶ Record metrics
    │
    ├── ASK ───▶ (With auto_approve=True) ──▶ Tool.execute()
    │            (With auto_approve=False) ─▶ Return pending approval
    │
    └── DENY ──▶ Return failure with reason
```

---

## 6. Determinism & Learning

### 6.1 How Plasticity Works

Plasticity is **regime-contextual weight learning** — the system learns which engines perform best in specific signal regimes.

**Mechanism:**
1. **Regime detection:** SignalAnalyzer classifies into STABLE, TRENDING, VOLATILE, NOISY, TRANSITIONAL
2. **Error tracking:** After each prediction, `record_actual()` computes |predicted - actual|
3. **Bayesian update:** Inverse error updates Gaussian prior for (regime, engine) pair
4. **Weight computation:** Weights = normalized posterior means
5. **TTL decay:** Unused regimes decay to uniform weights over time

**Key Parameters:**
```python
_ALPHA = 0.15  # Default EMA smoothing
_REGIME_ALPHA = {
    "STABLE": 0.15,        # Low adaptation (stable patterns)
    "TRENDING": 0.22,      # Medium adaptation
    "VOLATILE": 0.45,       # High adaptation (unstable)
    "NOISY": 0.08,         # Low adaptation (trust less)
    "TRANSITIONAL": 0.20,  # Medium adaptation
}
_MIN_WEIGHT = 0.05  # Floor to prevent total suppression
```

### 6.2 Why Redis is Used

**Problem:** Multi-worker deployments need shared plasticity state

**Solution:** Redis-backed shared plasticity
- **Writes:** Each `update()` writes to Redis hash `plasticity:{regime}`
- **Reads:** `get_weights()` reads from Redis with 60s local cache
- **Consistency:** All workers see same weights for same regime
- **Performance:** Local cache prevents Redis round-trips

**Redis Schema:**
```
HSET plasticity:STABLE taylor 0.85
HSET plasticity:STABLE baseline 0.15
HSET plasticity:TRENDING taylor 0.60
HSET plasticity:TRENDING statistical 0.40
```

### 6.3 How the System "Learns" Over Time

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

## 7. Project Structure

```
iot_machine_learning/
├── domain/                              # Pure business logic (no I/O)
│   ├── entities/
│   │   ├── results/                     # Prediction, AnomalyResult
│   │   ├── series/                      # StructuralAnalysis, SeriesProfile
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
│   │   └── memory_recall_enricher.py
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
│   ├── ml/
│   │   ├── engines/                     # Prediction engines
│   │   │   ├── taylor/                  # Taylor polynomial
│   │   │   ├── baseline.py              # Moving average
│   │   │   └── ensemble.py              # Model ensemble
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
│   │   └── predict_sensor_value.py      # PredictSensorValueUseCase
│   ├── dto/
│   │   └── prediction_dto.py
│   └── explainability/
│       └── explanation_renderer.py
│
├── ml_service/                          # FastAPI service
│   ├── api/
│   │   ├── routes.py                    # HTTP endpoints
│   │   ├── schemas.py                   # Pydantic models
│   │   └── services/
│   │       └── prediction_service.py
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

## 8. How to Run

### 8.1 Local Setup

```bash
# Clone and setup
cd /home/nicolas/Documentos/Iot_System/iot_machine_learning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
# Edit .env with your database credentials
```

### 8.2 Dependencies

**Core:**
- Python 3.11+
- FastAPI + Uvicorn
- Pydantic V2
- NumPy, SciPy
- scikit-learn (for anomaly detection)

**Infrastructure:**
- SQL Server 2022 (primary persistence)
- Redis 7+ (streaming, plasticity, state management)
- Weaviate 1.24+ (optional, experimental cognitive memory)
- MQTT broker (optional, for IoT ingestion)

**Install:**
```bash
pip install fastapi uvicorn pydantic numpy scipy scikit-learn redis asyncpg
```

### 8.3 Services Required

**SQL Server (Required):**
```bash
docker run -d -p 1434:1434 \
  -e SA_PASSWORD=YourPassword123 \
  -e ACCEPT_EULA=Y \
  mcr.microsoft.com/mssql/server:2022-latest
```

**Redis (Required):**
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

**Weaviate (Optional - Experimental):**
```bash
# Note: Weaviate integration is for experimental cognitive memory features
# Not required for core prediction/anomaly functionality
docker run -d -p 8080:8080 \
  semitechnologies/weaviate:1.24.0
```

**MQTT (Optional):**
```bash
docker run -d -p 1883:1883 -p 9001:9001 \
  eclipse-mosquitto:2
```

### 8.4 Starting the Service

```bash
# Start ML service
uvicorn ml_service.main:app --host 0.0.0.0 --port 8002 --reload

# Start batch runner (for background processing)
python -m iot_ingest_services.jobs.ml_batch_runner

# Start stream consumer (for real-time processing)
python -m ml_service.consumers.stream_consumer
```

**Verify:**
```bash
curl http://localhost:8002/
# Response: {"service": "iot-ml-service", "version": "0.2.0", "status": "ok"}
```

---

## 9. API Reference

### 9.1 Prediction Endpoint

**POST** `/api/v1/predict`

Request:
```json
{
  "series_id": "sensor_42",
  "values": [82.1, 83.5, 84.2, 85.0, 85.5],
  "timestamps": [1741234500, 1741234515, 1741234530, 1741234545, 1741234560],
  "threshold": 90.0
}
```

Response:
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

### 9.2 Anomaly Detection Endpoint

**POST** `/api/v1/detect-anomalies`

Request:
```json
{
  "series_id": "sensor_42",
  "values": [82.1, 83.5, 84.2, 85.0, 92.5],
  "timestamps": [1741234500, 1741234515, 1741234530, 1741234545, 1741234560]
}
```

Response:
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

### 9.3 Health Check

**GET** `/health`

Response:
```json
{
  "status": "healthy",
  "version": "0.2.1-GOLD",
  "services": {
    "redis": "connected",
    "sqlserver": "connected",
    "circuit_breaker": "closed"
  },
  "metrics": {
    "predictions_total": 15234,
    "plasticity_updates": 8765,
    "active_windows": 142
  }
}
```

---

## 10. Configuration

### 9.1 Feature Flags

All features are **disabled by default** for safety. Enable via environment variables:

```bash
# Core Features
export ML_USE_TAYLOR_PREDICTOR=true          # Enable Taylor engine
export ML_USE_KALMAN_FILTER=true             # Enable Kalman filtering
export ML_ENABLE_PLASTICITY=true               # Enable weight learning

# Iterative Mode
export ML_ENABLE_ITERATIVE=true                # Enable cognitive loop
export ML_ITERATIVE_MAX_ITERATIONS=3           # Max iterations
export ML_ITERATIVE_CONFIDENCE_THRESHOLD=0.85  # Stop when reached
export ML_ITERATIVE_TIME_BUDGET_MS=5000      # Hard limit

# Cognitive Memory
export ML_ENABLE_COGNITIVE_MEMORY=true         # Weaviate integration
export ML_COGNITIVE_MEMORY_URL=http://localhost:8080
export ML_ENABLE_MEMORY_RECALL=true            # Query historical explanations

# Safety & Guards
export ML_ACTION_GUARD_ENABLED=true            # Enable action guards
export ML_DOMAIN_BOUNDARY_ENABLED=true         # Validate input domain
export ML_COHERENCE_CHECK_ENABLED=true           # Prediction coherence check

# Performance
export ML_STREAM_USE_SLIDING_WINDOW=true       # Use in-memory windows
export ML_ENTERPRISE_USE_PRELOADED_DATA=true   # Reduce DB queries
export ML_BATCH_PARALLEL_WORKERS=4             # Parallel batch processing

# Rollback (Panic Button)
export ML_ROLLBACK_TO_BASELINE=true            # Force all to baseline
```

### 9.2 Backpressure Settings

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

---

## 10. Example Flow

### 10.1 Temperature Sensor Alert

**Input:**
```json
{
  "sensor_id": 42,
  "value": 85.5,
  "timestamp": "2026-03-31T14:30:00Z",
  "type": "temperature"
}
```

**Pipeline Execution:**

1. **INGEST:** MQTT message received, parsed, written to Redis Stream

2. **PERCEIVE:** SignalAnalyzer detects regime
   ```
   Regime: TRENDING (slope: +2.5°C/min, accelerating)
   Noise ratio: 0.03 (low)
   ```

3. **PREDICT:** Engines provide predictions
   ```
   Taylor: 87.2°C (confidence: 0.88, trend: rising)
   Baseline: 86.1°C (confidence: 0.72, trend: stable)
   Statistical: 87.0°C (confidence: 0.81, trend: rising)
   ```

4. **ADAPT:** Plasticity retrieves weights for TRENDING regime
   ```
   Weights from Redis: {taylor: 0.6, statistical: 0.3, baseline: 0.1}
   ```

5. **INHIBIT:** Check stability
   ```
   Baseline inhibited: high recent error in TRENDING regime
   ```

6. **FUSE:** Weighted combination
   ```
   Fused: 87.1°C (confidence: 0.85, trend: rising)
   Selected engine: taylor (primary weight contributor)
   ```

7. **EVALUATE (Iterative):**
   ```
   Iteration 0: confidence 0.85 >= threshold 0.85 → CONVERGED
   ```

8. **DECIDE:** DecisionEngine evaluates
   ```
   Prediction: 87.1°C (current: 85.5°C)
   Threshold: 80.0°C
   Recommended action: send_alert (critical: approaching limit)
   Guard level: AUTO (critical severity)
   ```

9. **ACT:** Tool executes
   ```
   send_alert executed
   Channels: push, email
   Severity: critical
   Message: "Temperature sensor 42 approaching critical threshold: 87.1°C predicted"
   ```

**Output:**
```json
{
  "predicted_value": 87.1,
  "confidence": 0.85,
  "trend": "rising",
  "selected_engine": "taylor",
  "metadata": {
    "cognitive_diagnostic": {
      "regime": "TRENDING",
      "final_weights": {"taylor": 0.6, "statistical": 0.3, "baseline": 0.1}
    },
    "cognitive_loop": {
      "iterations_used": 1,
      "converged": true
    },
    "action_taken": {
      "tool": "send_alert",
      "guard": "AUTO",
      "executed": true
    }
  }
}
```

---

## 11. Monitoring & Metrics

### 11.1 Prometheus Metrics

ZENIN exports Prometheus metrics at `/metrics`:

```
# Prediction metrics
zenin_predictions_total{series_id="sensor_42",engine="taylor"} 15234
zenin_prediction_latency_ms{quantile="0.99"} 45.2
zenin_prediction_confidence_avg 0.82

# Plasticity metrics
zenin_plasticity_updates_total{regime="TRENDING"} 8765
zenin_plasticity_weights{regime="TRENDING",engine="taylor"} 0.70

# Anomaly metrics
zenin_anomalies_detected_total{severity="WARNING"} 234
zenin_anomaly_detection_latency_ms{quantile="0.95"} 23.1

# Tool execution metrics
zenin_tool_executions_total{tool="send_alert",guard="AUTO"} 189
zenin_tool_execution_failures_total{tool="adjust_threshold"} 3

# Circuit breaker metrics
zenin_circuit_breaker_state{service="redis"} 0  # 0=closed, 1=open
zenin_circuit_breaker_failures_total{service="sqlserver"} 12

# Data drift metrics
zenin_concept_drift_score{series_id="sensor_42"} 0.15
zenin_drift_alerts_triggered_total 5
```

### 11.2 Health Checks

**Liveness Probe:** `GET /health/live`
- Returns 200 if service is running
- Used by Kubernetes to restart unhealthy pods

**Readiness Probe:** `GET /health/ready`
- Returns 200 if service can accept traffic
- Checks Redis, SQL Server connectivity

**Deep Health:** `GET /health`
- Returns full system status including:
  - Service version (0.2.1-GOLD)
  - Component connectivity (redis, sqlserver)
  - Circuit breaker states
  - Recent metrics (predictions_total, active_windows)

### 11.3 Logging

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

## 12. Troubleshooting

### 12.1 Common Issues

**Issue: "AttributeError: 'MetaCognitiveOrchestrator' object has no attribute '_storage'"**
- **Cause:** Using pre-GOLD code
- **Fix:** Upgrade to 0.2.1-GOLD where `_storage` is properly initialized

**Issue: Low confidence scores (< 0.5)**
- **Diagnosis:** Check `metadata.cognitive_diagnostic.perceptions`
- **Causes:**
  - High noise ratio → Wait for more stable signal
  - Engine disagreement → Plasticity will learn over time
  - Insufficient data → Need at least 3-5 readings

**Issue: "Modo Amnesico" (Amnesic Mode)**
- **Cause:** Circuit breaker opened for persistence service
- **Status:** `cognitive_trace.amnesic_mode: true`
- **Impact:** Plasticity weights stored in RAM only (lost on restart)
- **Fix:** Check Redis/SQL Server connectivity, restart service

**Issue: Plasticity not learning**
- **Check:** Are you calling `record_actual()` after predictions?
- **Check:** Is Redis running? (Required for multi-worker shared state)
- **Check:** Are predictions happening frequently enough? (Need error signals)

### 12.2 Debug Mode

Enable verbose logging:

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

### 12.3 Performance Tuning

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

## 13. Development Guide

### 13.1 Adding a New Prediction Engine

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

### 13.2 Adding a New Tool

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

## 14. Current Limitations

### 14.1 Known Constraints

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

### 14.2 Performance Limits

| Metric | Tested Limit | Theoretical Limit |
|--------|---------------|-------------------|
| Sensors | 1,000 | 10,000 (with sharding) |
| Latency (p99) | 150ms | 50ms (with caching) |
| Throughput | 10K msgs/sec | 100K (with batching) |
| Plasticity regimes | 10 | 100 (with LRU tuning) |

---

## 15. Roadmap

### Phase 4: Distributed Learning Improvements
- [ ] Distributed plasticity with consensus (Raft/Paxos)
- [ ] Federated learning across edge nodes
- [ ] Model compression for edge deployment
- [ ] A/B testing framework for engine selection

### Phase 5: Scalability
- [ ] Horizontal scaling with consistent hashing
- [ ] Partitioned sliding windows
- [ ] Async database writes with write-behind cache
- [ ] Circuit breaker for all external calls

### Phase 6: UTSAE Full Integration
- [x] `series_id: str` throughout (GOLD 0.2.1)
- [x] Canonical `Reading` type across all services
- [ ] Schema registry for versioned message formats
- [ ] Multi-tenant isolation at domain layer

### Future: Advanced Cognitive Capabilities
- [ ] **Causal Inference:** DoWhy integration for causal reasoning
- [ ] **Counterfactual Explanations:** "What if sensor X was different?"
- [ ] **Full RL:** Bandit algorithms for engine selection
- [ ] **Natural Language Interface:** Text commands to tool execution
- [ ] **Visual Analytics:** Interactive explanation exploration

---

## Contributing

See `ARCHITECTURE.md` for detailed design decisions and `MIGRATION_SCORECARD.md` for legacy API sunset plan.

## License

Internal ZENIN project. All rights reserved.

---

**Last Updated:** 2026-04-02  
**System Status:** GOLD 0.2.1 — Production-ready  
**Contact:** ML Platform Team  
**Support:** support@zenin.ai
