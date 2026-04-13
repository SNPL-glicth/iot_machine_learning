# ZENIN Cognitive Analytics Engine — GOLD 0.3.0

**Version:** 0.3.0-GOLD | **Tests:** ~1400+ passed | **Architecture:** Hexagonal + Cognitive + Temporal + Zero Magic Numbers  
**Status:** Production-Ready | **License:** Internal ZENIN Project

**What's New in 0.3.0:**
- ✅ Universal Analysis Engine (text, numeric, tabular, mixed inputs)
- ✅ Document Analysis Pipeline with full cognitive phases
- ✅ Consistent severity classification (urgency-aware)
- ✅ Enhanced entity detection (temperatures, metrics without prefixes)
- ✅ Real-time pattern interpretation (no hardcoded values)
- ✅ End-to-end explainability for document analysis

---

## 🎯 What is ZENIN?

ZENIN transforms raw **sensor data AND documents** into **intelligent decisions** through a transparent, multi-phase cognitive pipeline. Unlike traditional ML systems that are black boxes, ZENIN provides full explainability of every prediction, anomaly, and document analysis.

### The Cognitive Pipeline (Numeric & Time-Series)

```
PERCEIVE → PREDICT → ADAPT → INHIBIT → FUSE → EXPLAIN → DECIDE → ACT
```

- **PERCEIVE:** Analyze signal structure (regime, noise, trends)
- **PREDICT:** Multi-engine forecasting (Taylor, Baseline, Statistical)
- **ADAPT:** Learn which engine works best in each context (Bayesian weight tracking — not RL)
- **INHIBIT:** Suppress unreliable engines
- **FUSE:** Weighted consensus prediction
- **EXPLAIN:** Build reasoning trace
- **DECIDE:** Map to recommended action with business impact
- **ACT:** Execute with safety guardrails (AUTO/ASK/DENY)

### The Universal Analysis Pipeline (Text & Documents)

```
INPUT → PERCEIVE → ANALYZE → REMEMBER → REASON → EXPLAIN → DECIDE
```

- **PERCEIVE:** Auto-detect input type, classify domain, extract entities
- **ANALYZE:** Text urgency, sentiment, readability, pattern detection
- **REMEMBER:** Recall similar past analyses (cognitive memory)
- **REASON:** Inhibit unreliable signals, adapt weights, fuse perceptions
- **EXPLAIN:** Build narrative with detected entities, severity, actions
- **DECIDE:** Classify severity (critical/warning/info) with urgency-aware logic

### Key Differentiators

| Feature | Traditional ML | ZENIN |
|---------|-----------------|-------|
| **Input Types** | Numeric only | Numeric + Text + Documents + Tabular |
| **Reasoning** | Black box model | Transparent cognitive phases |
| **Learning** | Retrain models | Real-time weight adaptation |
| **Safety** | Post-hoc monitoring | Built-in guard system |
| **Actions** | Manual interpretation | Autonomous with approval gates |
| **Explainability** | SHAP/LIME approximations | Full trace per prediction |
| **Severity** | Fixed thresholds | Urgency-aware dynamic classification |

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
# {"service": "iot-ml-service", "version": "0.3.0-GOLD", "status": "ok"}
```

> **Security note:** The service uses API key authentication
> (`X-API-Key` header). Set `ML_API_KEY` in your `.env` file.
> If `ML_API_KEY` is empty, authentication is disabled (dev mode only).
> **Do not expose this service to the internet without VPN or
> a reverse proxy until Phase 2 auth (JWT + tenant isolation) is complete.**

---

## 📊 GOLD 0.3.0 Release Highlights

### Universal Analysis Engine

| Feature | Description | Impact |
|---------|-------------|--------|
| **Multi-Input Support** | Analyze text, numeric, tabular, and mixed inputs | Single engine for all data types |
| **Document Analysis** | Full 5-phase cognitive pipeline for text documents | Crisis reports, logs, emails fully analyzed |
| **Urgency-Aware Severity** | Dynamic severity classification based on urgency (0.45 weight) + sentiment (0.20) + impact (0.35) | Critical urgency (≥0.85) + negative sentiment = CRITICAL |
| **Real-Time Patterns** | Pattern detection uses actual urgency/sentiment values | No more hardcoded "Operación estable" for crisis documents |
| **Enhanced Entity Detection** | Standalone temperature values detected (e.g., "94°C") | Captures metrics without explicit context |

### Critical Bug Fixes (Document Analysis)

| Issue | Fix | Impact |
|-------|-----|--------|
| **Severity inconsistency** | Urgency 0.88 now correctly maps to CRITICAL | Crisis documents properly flagged |
| **Pattern hardcoding** | Eliminated hardcoded pattern values in universal_bridge.py | Dynamic pattern generation based on real data |
| **Impact detection** | Added standalone metric patterns for °C/°F values | Temperatures like "94°C" now detected |
| **Severity formula** | Updated weights: urgency 0.45, sentiment 0.20, impact 0.35 | Urgency has appropriate influence on severity |

### GOLD 0.2.1 Release Highlights

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

### 2.6 Universal Analysis & Document Processing (NEW in 0.3.0)

**Input-Agnostic Analysis:** The UniversalAnalysisEngine handles any input type:
- **TEXT:** Reports, logs, emails, alerts with urgency/sentiment analysis
- **NUMERIC:** Time-series sensor data with regime detection
- **TABULAR:** Structured data with first-column analysis
- **MIXED:** Hybrid documents with multiple data types

**Document Analysis Pipeline:**
```
Document Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. UNIVERSAL BRIDGE                                         │
│    • Auto-detect content type                               │
│    • Extract entities (temperatures, costs, percentages)      │
│    • Compute urgency (0-1) and sentiment (pos/neg/neutral)  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PERCEIVE PHASE                                           │
│    • Classify domain (infrastructure/security/finance)      │
│    • Build signal profile with metadata                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. ANALYZE PHASE                                            │
│    • Text urgency engine (keyword-based + context)            │
│    • Sentiment analyzer (positive/negative/neutral)          │
│    • Pattern detector (regime changes, spikes)            │
│    • Impact signal scanner (SLA breaches, extreme metrics)  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. REASON PHASE (Severity Classification)                     │
│    • Fuse multiple perceptions with plasticity weights       │
│    • Classify severity: critical/warning/info               │
│    • Urgency-aware: 0.85+ urgency + negative = critical     │
│    • Impact-aware: Extreme metrics elevate severity         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. EXPLAIN PHASE                                            │
│    • Build rich narrative with entities detected            │
│    • Pattern interpretation (escalation/degradation/stable) │
│    • Context-specific recommendations                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
AnalysisOutput with conclusion, severity, entities, actions
```

**Example Document Analysis:**
```
Input: "ALERTA: Rack B-07 temperatura 94°C (límite 80°C). Servidores offline."

Output:
  Domain: infrastructure
  Urgency: 0.88 (critical)
  Sentiment: negative
  Entities: ["94°C", "80°C"]
  Severity: Critical (HIGH)
  Pattern: Escalada narrativa
  Actions: Restart affected node immediately, Check sensor readings
```

**Severity Classification Formula:**
```
composite = urgency×0.45 + sentiment×0.20 + impact×0.35

Overrides:
- urgency ≥ 0.85 + negative sentiment → critical
- urgency ≥ 0.75 → warning (minimum)
- 3+ impact categories hit → critical
```

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

> **Regime types** are defined as `RegimeType` enum in
> `domain/entities/series/structural_analysis.py`. Use
> `RegimeType.STABLE`, `RegimeType.TRENDING`, etc. in code —
> never string literals.

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
│    • If low: re-runs pipeline (input unchanged — refinement    │
│      not yet implemented)                                       │
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

### 6.2 Why Redis is Used

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

### 9.3 Document Analysis Endpoint (NEW in 0.3.0)

**POST** `/api/v1/analyze-document`

Analyze text documents (reports, logs, alerts) with full cognitive pipeline.

Request:
```json
{
  "document_id": "crisis_report_001",
  "content": "ALERTA: Rack B-07 temperatura 94°C (límite 80°C). Servidores web-prod-11 al web-prod-18 offline. 77% de capacidad perdida.",
  "tenant_id": "acme_corp",
  "content_type": "text"
}
```

Response:
```json
{
  "document_id": "crisis_report_001",
  "tenant_id": "acme_corp",
  "classification": "infrastructure",
  "conclusion": "Infrastructure incident — Critical (HIGH) | Confidence: 65.0%\n635 words. Entities: 94°C, 80°C\nUrgency: 0.88 | Sentiment: negative\nPatrón: Escalada narrativa: El documento muestra progresión de alertas menores hacia incidente crítico.",
  "confidence": 0.65,
  "severity": "critical",
  "analysis": {
    "domain": "infrastructure",
    "urgency_score": 0.88,
    "sentiment": "negative",
    "entities": ["94°C", "80°C"],
    "word_count": 635,
    "patterns": {
      "urgency_regime": "critical",
      "has_escalation": true
    }
  },
  "explanation": {
    "narrative": "Escalada narrativa: El documento muestra progresión de alertas menores hacia incidente crítico. Típico de situaciones que no recibieron atención temprana.",
    "context": "Típico de incidentes de infraestructura que no recibieron respuesta temprana. Revisa logs de sistema y métricas de rendimiento.",
    "confidence": 0.95
  },
  "actions": [
    "Restart affected node immediately",
    "Check sensor readings and thresholds",
    "Reduce system load to prevent cascade failure"
  ],
  "processing_time_ms": 145.2,
  "cached": false
}
```

### 9.4 Health Check

**GET** `/health`

Response:
```json
{
  "status": "healthy",
  "version": "0.3.0-GOLD",
  "services": {
    "redis": "connected",
    "sqlserver": "connected",
    "circuit_breaker": "closed"
  },
  "metrics": {
    "predictions_total": 15234,
    "plasticity_updates": 8765,
    "active_windows": 142,
    "documents_analyzed": 3847
  }
}
```

---

## 10. Configuration

### 10.1 Feature flags reference

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
| `ML_PIPELINE_BUDGET_MS` | `1000` | Pipeline budget in ms |
| `ML_COHERENCE_CHECK_ENABLED` | `false` | Prediction coherence check |
| `ML_DOMAIN_BOUNDARY_ENABLED` | `false` | Validate input domain |
| `ML_ACTION_GUARD_ENABLED` | `false` | Enable action guards |
| `ML_DECISION_ARBITER_ENABLED` | `false` | Enable decision arbiter |
| `ML_CONFIDENCE_CALIBRATION_ENABLED` | `false` | Confidence calibration |
| `ML_NARRATIVE_UNIFICATION_ENABLED` | `false` | Narrative unification |

### 10.2 JSON flags (dict-type parameters)

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

### 10.3 Backpressure & limits

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

### 10.4 Panic button

```bash
# Force all predictions to baseline engine, bypass cognitive pipeline
export ML_ROLLBACK_TO_BASELINE=true
```

### 10.5 Architecture decisions: why flags over constants

| Decision | Rationale |
|----------|-----------|
| **All numeric thresholds in flags** | Adjustable in <5 min without code change (ISO 27001 A.12.1.2) |
| **JSON flags for dicts** | Compatible with env vars in Kubernetes/docker-compose |
| **`_get_flags()` on every call** | Hot-reload: changes propagate without restart |
| **`RegimeType` enum** | Eliminates magic strings across 10+ files |
| **`RedisKeys` registry** | Single point of change for key patterns; enables access auditing |
| **Fallback to `FeatureFlags()`** | Service stays alive even if config system fails |

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

4. **ADAPT:** Bayesian weight tracker retrieves weights for TRENDING regime
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
  - Service version (0.2.2-GOLD)
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

**Issue: Unexpected decision scores after flag change**
- **Cause:** `ML_DECISION_BASE_SCORES` JSON is malformed
- **Symptom:** All confidence scores at default (no error thrown)
- **Fix:** Validate JSON before exporting: `echo $ML_DECISION_BASE_SCORES | python -m json.tool`
- **Check:** Run `python -c "from ml_service.config.feature_flags import get_feature_flags; print(get_feature_flags().ML_DECISION_BASE_SCORES)"`

**Issue: Plasticity not adapting to new learning rate**
- **Cause:** `ML_PLASTICITY_REGIME_ALPHAS` not picked up
- **Note:** Hot-reload applies to new predictions only. In-flight predictions
  use the snapshot captured at call start.
- **Fix:** Confirm with: `export ML_PLASTICITY_ALPHA=0.30 && curl http://localhost:8002/health`

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

---

## Changelog

### GOLD 0.3.0 (2026-04-12)

**Major Features:**
- **Universal Analysis Engine:** Input-agnostic cognitive pipeline supporting text, numeric, tabular, and mixed inputs
- **Document Analysis Pipeline:** Full 5-phase analysis (PERCEIVE→ANALYZE→REMEMBER→REASON→EXPLAIN) for documents
- **Urgency-Aware Severity Classification:** Dynamic severity based on urgency score (0.45 weight) + sentiment (0.20) + impact (0.35)
- **Enhanced Entity Detection:** Standalone temperature detection (e.g., "94°C" without "temperature" prefix)
- **Real-Time Pattern Interpretation:** Dynamic pattern detection based on actual urgency/sentiment (no hardcoded values)

**Critical Fixes:**
- **Fix:** Eliminated hardcoded pattern values in universal_bridge.py (lines 169-178)
- **Fix:** Severity inconsistency - urgency 0.88 now correctly maps to CRITICAL
- **Fix:** Pattern detection - crisis documents now show "Escalada narrativa" instead of "Operación estable"
- **Fix:** Impact detector now captures standalone temperature values with °C/°F suffixes
- **Fix:** Reason phase now considers max(fused_score, urgency_score) for severity classification

**Architecture Improvements:**
- Added `SeverityResult` with urgency overrides (≥0.85 + negative = critical)
- Enhanced `ImpactSignalDetector` with 5 categories: critical markers, SLA breaches, extreme metrics, temporal risk, cascade
- Refactored `UniversalPerceptionCollector` to use real urgency scores for pattern generation
- Updated `classify_text_severity()` with urgency-weighted composite formula

**Files Modified:**
- `ml_service/api/services/analysis/universal_bridge.py` - Dynamic pattern generation
- `infrastructure/ml/cognitive/text/severity_mapper.py` - Urgency-aware classification
- `infrastructure/ml/cognitive/text/impact_detector.py` - Standalone metric detection
- `infrastructure/analysis/pipeline/reason_helpers.py` - Urgency-based severity

### GOLD 0.2.2 (2026-04-11)

**Features:**
- Cognitive pipeline with 8 phases
- Plasticity learning (Bayesian weight updates)
- Tool system with safety guards (AUTO/ASK/DENY)
- Redis-backed shared state
- Circuit breaker for persistence

**Stats:** ~1337 tests passed

## License

Internal ZENIN project. All rights reserved.

---

**Last Updated:** 2026-04-12  
**System Status:** GOLD 0.3.0 — Production-ready with Universal Analysis  
**Contact:** ML Platform Team  
**Support:** support@zenin.ai  
**Document Analysis:** Fully operational — tested with crisis scenarios
