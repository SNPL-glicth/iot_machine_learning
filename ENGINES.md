# ZENIN Engines — Deep Architecture Reference

This document describes **every engine** in ZENIN: how it works, what it predicts, where it plugs into the pipeline, and how it complements other engines. If you are extending ZENIN or debugging predictions, start here.

---

## Table of Contents

- [Overview](#overview)
- [Prediction Engines (Time-Series)](#prediction-engines-time-series)
  - [Engine Factory](#engine-factory)
  - [TaylorPolynomialEngine](#taylorengine)
  - [SeasonalPredictorEngine](#seasonalengine)
  - [StatisticalPredictionEngine](#statisticalengine)
  - [BaselineMovingAverageEngine](#baselineengine)
  - [EnsembleWeightedPredictor](#ensemblepredictor)
- [Cognitive Orchestration](#cognitive-orchestration)
  - [MetaCognitiveOrchestrator](#metacognitiveorchestrator)
  - [TextCognitiveEngine](#textcognitiveengine)
- [Integration Points](#integration-points)
- [How Engines Complement Each Other](#how-engines-complement-each-other)
- [Adding a New Engine](#adding-a-new-engine)

---

## Overview

ZENIN runs two families of engines:

| Family | Purpose | Lives in |
|--------|---------|----------|
| **Prediction Engines** | Forecast the next numeric value from a time-series window | `infrastructure/ml/engines/` |
| **Cognitive Engines** | Analyze signals, fuse predictions, and produce decisions with explanations | `infrastructure/ml/cognitive/` |

All prediction engines implement the same interface: `PredictionEngine` (defined in `infrastructure/ml/interfaces.py`). They expose:

- `name` — registry identifier (e.g. `"taylor_polynomial"`)
- `can_handle(n_points)` — minimum window length required
- `predict(values, timestamps)` → `PredictionResult(predicted_value, confidence, trend, metadata)`

The cognitive layer consumes these `PredictionResult`s, runs them through signal-inhibition, fusion, and explanation building.

---

## Prediction Engines (Time-Series)

### Engine Factory

**File:** `infrastructure/ml/engines/core/factory.py`

Every prediction engine is registered in a central `EngineFactory`:

```python
EngineFactory.register("taylor_polynomial", TaylorPredictionEngine)
EngineFactory.register("seasonal_fft", SeasonalPredictorEngine)
EngineFactory.register("statistical_ema_holt", StatisticalPredictionEngine)
EngineFactory.register("baseline_moving_average", BaselineMovingAverageEngine)
```

You can create any engine by name:

```python
engine = EngineFactory.create("taylor_polynomial", series_id="temp_42")
```

If the name is unknown, the factory silently falls back to `BaselineMovingAverageEngine`. This is the **fail-open** guarantee: a missing engine never crashes the pipeline, it just degrades to the simplest predictor.

**Auto-registration** (decorator):

```python
from infrastructure.ml.engines.core import register_engine
from infrastructure.ml.interfaces import PredictionEngine

@register_engine("my_custom_engine")
class MyCustomEngine(PredictionEngine):
    @property
    def name(self) -> str:
        return "my_custom_engine"
```

**Plugin discovery:** `discover_engines("my_package.engines")` imports every module in a package to trigger `@register_engine` decorators at import time.

---

### TaylorPolynomialEngine

**File:** `infrastructure/ml/engines/taylor/engine.py` (orchestrator) + `taylor/prediction_pipeline.py` (execution)

**What it does:**
Fits a local polynomial to the time-series window using numerical derivatives, then projects the next point forward.

**How it works:**
1. **Sanitize inputs** — filters NaN and `inf` values
2. **Detect gaps** — if timestamps have gaps > 2x median Δt, uses the largest continuous segment
3. **Compute Δt** — robust time-step estimation (gap-aware)
4. **Load hyperparameters** — per-series α/β from Redis-backed `HyperparameterAdaptor`
5. **Compute Taylor coefficients** — backward, central, or least-squares derivative estimation
6. **Project** — `h = horizon × dt`, evaluate polynomial at `t + h`
7. **Clamp** — prediction is clamped to `±20%` of the value range (configurable)
8. **Physical bounds** — optional `physical_min` / `physical_max` override clamp
9. **Diagnose** — stability indicator, local fit error, structural analysis
10. **Cache** — `TaylorCoefficientCache` memoizes coefficients per `(series_id, window_hash)`

**Metadata output:**
```json
{
  "order": 3,
  "derivatives": {"d0": 42.0, "d1": 0.5, "d2": -0.01},
  "dt": 60.0,
  "clamped": false,
  "physical_clamp_applied": false,
  "diagnostic": {"stability_indicator": 0.92, "local_fit_error": 0.003},
  "structural_analysis": {"regime": "stable", "volatility": "low"},
  "cache_hit": true
}
```

**When it shines:**
- Smooth, low-noise signals where derivatives are meaningful
- Signals with known physical bounds (temperature, pressure)
- When gap detection matters (irregular sensor sampling)

**When it struggles:**
- High-volatility noise (derivatives amplify noise)
- Very short windows (< 3 points → `can_handle` returns `False`)
- Seasonal/cyclical patterns (Taylor is local, not periodic)

**Integration:**
- Called by `MetaCognitiveOrchestrator` in the PREDICT phase
- If `can_handle` returns `False`, the orchestrator skips it
- Metadata feeds into `WeightedFusion` for confidence-weighted averaging

---

### SeasonalPredictorEngine

**File:** `infrastructure/ml/engines/seasonal/engine.py`

**What it does:**
Detects periodic cycles via Fast Fourier Transform (FFT) and predicts the next value by matching the phase position from previous cycles.

**How it works:**
1. **Resample to uniform grid** — if timestamps are irregular, interpolates to equal spacing
2. **NaN guard** — filters non-finite values before FFT
3. **FFT cycle detection** — finds the dominant frequency (period)
4. **Confidence threshold** — needs `confidence ≥ min_confidence` (default 0.6)
5. **Project by phase** — finds the same position in the previous 2 cycles and averages those values
6. **Trend classification** — compares current value with same-phase value from last cycle

**Metadata output:**
```json
{
  "engine": "seasonal_fft",
  "detected_period": 24,
  "latency_ms": 4.2
}
```

**Fallback:** If insufficient data (< 2 cycles), FFT fails, or confidence is too low → returns last value with confidence 0.3.

**When it shines:**
- Daily cycles (temperature, energy consumption)
- Weekly patterns (traffic, sales)
- Any signal with clear periodicity

**When it struggles:**
- Aperiodic signals (random walk, shocks)
- Windows shorter than 2 cycles
- Highly irregular sampling (resampling introduces artifacts)

**Integration:**
- Registered in `EngineFactory` as `"seasonal_fft"`
- Called alongside Taylor, Statistical, and Baseline in parallel
- Its `predicted_value` competes in `WeightedFusion`
- If seasonal is confident and Taylor is not, the fusion weights shift toward seasonal

---

### StatisticalPredictionEngine

**File:** `infrastructure/ml/engines/statistical/engine.py`

**What it does:**
Double exponential smoothing (Holt's method) with auto-tuned α and β parameters.

**How it works:**
1. **Load hyperparameters** — per-series α (smoothing) and β (trend) from Redis
2. **Holt's method** — `level = α × value + (1-α) × (level + trend)`; `trend = β × (level - prev_level) + (1-β) × trend`
3. **Predict** — `prediction = level + trend × horizon`
4. **Residual analysis** — EMA of residuals gives noise ratio
5. **Confidence** — `1.0 - noise_ratio`, clamped to [0.2, 0.95]
6. **Trend classification** — `up` / `down` / `stable` based on trend magnitude vs residual std

**Auto-optimization (deferred):**
- Every 50 predictions, the engine marks itself as "needs reoptimization"
- The orchestrator calls `optimize()` between requests (safe moment)
- `StatisticalParamOptimizer` runs a grid search over α ∈ [0.1, 1.0], β ∈ [0.0, 0.5]
- If new MAE improves > 5%, saves new params via `HyperparameterAdaptor` → Redis

**Metadata output:**
```json
{
  "level": 42.5,
  "trend_component": 0.3,
  "alpha": 0.35,
  "beta": 0.1,
  "residual_std": 0.05,
  "diagnostic": {
    "stability_indicator": 0.15,
    "local_fit_error": 0.05,
    "method": "ema_holt"
  }
}
```

**When it shines:**
- Noisy signals where smoothing helps
- Signals with slow, persistent trends
- When you need adaptive parameter tuning without retraining

**When it struggles:**
- Sudden regime changes (Holt lags behind)
- Non-linear acceleration (Taylor handles this better)
- Very short windows (< 3 points)

**Integration:**
- Called in parallel with Taylor and Seasonal
- Its `alpha`/`beta` are hot-reloaded from Redis per request
- Reoptimization is externalized to avoid mutable-state writes during prediction

---

### BaselineMovingAverageEngine

**File:** `infrastructure/ml/engines/core/factory.py` (embedded) + `infrastructure/ml/engines/baseline/engine.py` (pure function)

**What it does:**
Simple moving average — the last line of defense.

**How it works:**
1. Sanitizes NaN/inf
2. Takes the mean of the last N values (default `min(20, len(values))`)
3. Confidence is derived from variance of the window
4. Trend is always `"stable"`

**Why it matters:**
- **Never fails** — `can_handle` returns `True` for any `n_points ≥ 1`
- **Factory fallback** — if any engine name is misspelled or unregistered, the factory returns this
- **Reference point** — ensemble weights use Baseline as a sanity check; if Taylor is wilder than Baseline, its weight drops

**Metadata output:**
```json
{
  "window": 10,
  "fallback": null,
  "clamped": false
}
```

**Integration:**
- Always present in the ensemble
- Provides the "anchor" that prevents exotic engines from drifting too far

---

### EnsembleWeightedPredictor

**File:** `infrastructure/ml/engines/ensemble/predictor.py`

**⚠️ Deprecated:** `EnsembleWeightedPredictor` is **never instantiated in production**. The orchestrator now handles fusion directly via `WeightedFusion`. This class remains in the codebase as a reference implementation and for standalone use cases.

**What it does:**
Combines N `PredictionPort` instances (not raw `PredictionEngine`s) using inverse-error weighting.

**How it works:**
1. Run all engines in parallel (or sequentially)
2. Filter out failed/skipped engines
3. Compute weighted average of predictions: `weight_i ∝ 1 / (error_i + 1e-9)`
4. Trend by majority vote (weighted)
5. Every 10 updates, recalculate weights and persist to DB (`zenin_ml.ensemble_weights`)

**Circuit breaker:** If an engine fails N times consecutively, its weight drops to 0 until reset.

**Integration:**
- Not used by `MetaCognitiveOrchestrator` (replaced by `WeightedFusion` + `InhibitionGate`)
- Can be used standalone for experiments or non-cognitive pipelines

---

## Cognitive Orchestration

### MetaCognitiveOrchestrator

**File:** `infrastructure/ml/cognitive/orchestration/orchestrator.py`

**What it does:**
The brain of ZENIN. Takes a sensor window, runs it through perception → prediction → inhibition → adaptation → fusion → explanation, and returns a `Decision`.

**Pipeline phases:**

| Phase | What happens | Engines involved |
|-------|-------------|------------------|
| **PERCEIVE** | Analyze signal structure (volatility, stationarity, trend) | `SignalAnalyzer` |
| **PREDICT** | Run all registered prediction engines in parallel | Taylor, Seasonal, Statistical, Baseline |
| **INHIBIT** | Suppress low-confidence or contradictory predictions | `InhibitionGate` |
| **ADAPT** | Update per-regime weights via Bayesian plasticity | `PlasticityTracker` |
| **FUSE** | Weighted average of surviving predictions + Hampel outlier rejection | `WeightedFusion` + `hampel_filter` |
| **EXPLAIN** | Build human-readable reasoning trace | `ExplanationBuilder` |

**Latency budget:** 500ms total. `PipelineTimer` tracks each phase. If PERCEIVE + PREDICT exceeds budget, cuts to Baseline fallback.

**Integration:**
- Called by `predict_sensor_value` use case
- Consumes `SensorWindow` from the sliding window store
- Produces `Prediction` domain entity with full metadata
- Persists weights to Redis (`zenin_ml.plasticity_weights`)

---

### TextCognitiveEngine

**File:** `infrastructure/ml/cognitive/text/engine.py`

**What it does:**
Deep cognitive analysis for text documents — sentiment, urgency, readability, semantic search, and decision classification.

**Pipeline phases (same interface as MetaCognitiveOrchestrator):**

| Phase | What happens |
|-------|-------------|
| **PERCEIVE** | Build signal profile from pre-computed text metrics (sentiment, urgency, readability scores) |
| **ANALYZE** | Map sub-analyzer scores to `EnginePerception[]` |
| **REMEMBER** | Recall similar past documents from cognitive memory (Weaviate) |
| **REASON** | Inhibit unreliable engines, fuse, classify severity |
| **EXPLAIN** | Assemble `Explanation` domain object |

**Key design principle:** No imports from `ml_service`. It receives pre-computed scores via `TextAnalysisInput`.

**Integration:**
- Called by `DocumentAnalyzer` in the ML Service
- Input comes from text analyzers (sentiment, urgency, readability) that run BEFORE the cognitive engine
- Output is an `Explanation` with severity, confidence, and reasoning trace
- Results are stored in `zenin_docs.analysis_results`

**Differences from MetaCognitiveOrchestrator:**
- Works on text metrics, not numeric time-series
- Memory phase recalls similar documents, not past sensor readings
- Domain-agnostic: works for logs, contracts, reports, trading notes

---

## Integration Points

### IoT / Sensor Data Flow

```
MQTT / HTTP ingest
       ↓
StreamConsumer (ingest service)
       ↓
SlidingWindowStore (Redis-backed LRU, max 1000 sensors)
       ↓
MetaCognitiveOrchestrator
  ├─ PERCEIVE → SignalAnalyzer
  ├─ PREDICT  → EngineFactory.create_all() [Taylor, Seasonal, Statistical, Baseline]
  ├─ INHIBIT  → InhibitionGate
  ├─ ADAPT    → PlasticityTracker (Redis)
  ├─ FUSE     → WeightedFusion + hampel_filter
  └─ EXPLAIN  → ExplanationBuilder
       ↓
Prediction (domain entity)
       ↓
SQL Server (zenin_ml.predictions)
       ↓
.NET Backend (read-only) → Frontend
```

### Document Analysis Flow

```
Frontend upload
       ↓
.NET Backend (parse + enqueue to zenin_docs.ingestion_queue)
       ↓
ML Service poller (zenin_queue_poller.py)
       ↓
DocumentAnalyzer (text analyzers: sentiment, urgency, readability)
       ↓
TextCognitiveEngine
  ├─ PERCEIVE → text metrics profile
  ├─ ANALYZE  → EnginePerception[]
  ├─ REMEMBER → Weaviate semantic search
  ├─ REASON   → InhibitionGate + fusion
  └─ EXPLAIN  → Explanation
       ↓
zenin_docs.analysis_results (SQL Server)
       ↓
.NET Backend (read-only) → Frontend (polling)
```

---

## How Engines Complement Each Other

| Signal Type | Best Engine | Why | Backup |
|-------------|-------------|-----|--------|
| Smooth, bounded | **Taylor** | Polynomial projection respects physical limits | Baseline |
| Cyclic / seasonal | **Seasonal FFT** | FFT detects cycles Taylor cannot see | Baseline |
| Noisy with trend | **Statistical** | EMA/Holt smooths noise, adapts α/β | Baseline |
| Chaotic / shock | **Baseline** | Never overfits, always stable | — |
| High volatility | **Ensemble fusion** | Weights shift toward the engine with lowest recent error | Baseline |

**Complementarity examples:**

1. **Temperature sensor (daily cycle):** Seasonal FFT detects the 24h cycle. Taylor captures the short-term drift within the cycle. Statistical smooths sensor noise. Fusion weights: Seasonal 0.5, Taylor 0.3, Statistical 0.2.

2. **Pressure sensor (smooth, bounded):** Taylor dominates (0.7 weight) because derivatives are clean and physical bounds prevent clamping. Seasonal is skipped (no periodicity). Statistical provides sanity check (0.2). Baseline anchors (0.1).

3. **Random walk / noise:** All engines return low confidence. InhibitionGate suppresses Taylor and Seasonal. WeightedFusion falls back to Baseline. PlasticityTracker records the regime as "unpredictable" and lowers all engine weights.

4. **Shock / step change:** Taylor detects the acceleration (`d2` spike). Statistical lags (Holt's trend takes time to adapt). Seasonal is confused (phase mismatch). Fusion weights shift to Taylor (0.6) + Baseline (0.4). After 50 predictions, Statistical reoptimizes α to be more reactive.

---

## Adding a New Engine

1. **Implement `PredictionEngine`**:

```python
from infrastructure.ml.interfaces import PredictionEngine, PredictionResult
from infrastructure.ml.engines.core import register_engine

@register_engine("my_engine")
class MyEngine(PredictionEngine):
    @property
    def name(self) -> str:
        return "my_engine"

    def can_handle(self, n_points: int) -> bool:
        return n_points >= 5

    def predict(self, values, timestamps=None):
        # Your logic here
        return PredictionResult(
            predicted_value=42.0,
            confidence=0.8,
            trend="up",
            metadata={"engine": "my_engine"}
        )
```

2. **Add to `select_engine_for_sensor`** (if it should be auto-selected for certain regimes):

```python
# in application/use_cases/select_engine.py
if regime == "my_regime":
    return EngineFactory.create("my_engine", series_id=series_id)
```

3. **Add tests** in `tests/unit/infrastructure/test_my_engine.py`

4. **Document** in this file (update the tables and integration diagrams)

---

## Files Reference

| Engine | File |
|--------|------|
| EngineFactory | `infrastructure/ml/engines/core/factory.py` |
| Taylor | `infrastructure/ml/engines/taylor/engine.py` + `prediction_pipeline.py` |
| Seasonal | `infrastructure/ml/engines/seasonal/engine.py` |
| Statistical | `infrastructure/ml/engines/statistical/engine.py` |
| Baseline | `infrastructure/ml/engines/baseline/engine.py` + `core/factory.py` |
| Ensemble (deprecated) | `infrastructure/ml/engines/ensemble/predictor.py` |
| MetaCognitiveOrchestrator | `infrastructure/ml/cognitive/orchestration/orchestrator.py` |
| TextCognitiveEngine | `infrastructure/ml/cognitive/text/engine.py` |
| PredictionEngine interface | `infrastructure/ml/interfaces.py` |
| Engine selection logic | `application/use_cases/select_engine.py` |

---

*Last updated: 2026-04-24*
