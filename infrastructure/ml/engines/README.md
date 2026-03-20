# infrastructure/ml/engines

Prediction engines — time-series forecasting implementations.

**Reorganized:** 2026-03-20

## Package Structure

### 📁 core/
Factory + registry + auto-discovery (2 files)
- `factory.py` (212 lines) — `EngineFactory` central registry + `@register_engine` decorator + `discover_engines()` plugin discovery + `BaselineMovingAverageEngine` embedded fallback

### 📁 baseline/
Simple moving average engine (2 files)
- `engine.py` (62 lines) — `predict_moving_average()` pure function + `BaselineConfig` + `BaselineMetadata`
- `adapter.py` (82 lines) — **DEPRECATED** `BaselinePredictionAdapter` (PredictionPort wrapper)

### 📁 taylor/
Taylor series prediction engine (9 files)
- `engine.py` (172 lines) — `TaylorPredictionEngine` orchestrator
- `adapter.py` (142 lines) — **DEPRECATED** `TaylorPredictionAdapter` + `KalmanFilterAdapter`
- `math.py` (60 lines) — Backward-compat facade re-exporting all math functions
- `types.py` (117 lines) — `TaylorCoefficients`, `TaylorDiagnostic`, `DerivativeMethod`
- `derivatives.py` (103 lines) — `estimate_derivatives()` (backward, central, least_squares)
- `polynomial.py` (65 lines) — `project()`, `compute_local_fit_error()`
- `diagnostics.py` (75 lines) — `compute_diagnostic()`, stability analysis
- `time_step.py` (34 lines) — `compute_dt()` robust Δt estimation
- `least_squares.py` (83 lines) — Least-squares derivative estimation

### 📁 statistical/
EMA/Holt-based forecasting (1 file)
- `engine.py` (179 lines) — `StatisticalPredictionEngine` double exponential smoothing

### 📁 ensemble/
Weighted combination of multiple engines (1 file)
- `predictor.py` (291 lines) — `EnsembleWeightedPredictor` (implements `PredictionPort`, NOT `PredictionEngine`)

---

## Folder Structure

```
engines/
├── __init__.py                    ← Public API (backward compatible)
├── README.md
├── core/                          ← Factory + registry
│   ├── __init__.py
│   └── factory.py                 ← EngineFactory + register_engine + discover_engines
├── baseline/                      ← Baseline engine
│   ├── __init__.py
│   ├── engine.py                  ← predict_moving_average (pure function)
│   └── adapter.py                 ← DEPRECATED BaselinePredictionAdapter
├── taylor/                        ← Taylor series engine + math
│   ├── __init__.py
│   ├── engine.py                  ← TaylorPredictionEngine orchestrator
│   ├── adapter.py                 ← DEPRECATED TaylorPredictionAdapter + KalmanFilterAdapter
│   ├── math.py                    ← Backward-compat facade
│   ├── types.py                   ← TaylorCoefficients, TaylorDiagnostic, DerivativeMethod
│   ├── derivatives.py             ← estimate_derivatives (backward, central, least_squares)
│   ├── polynomial.py              ← project(), compute_local_fit_error()
│   ├── diagnostics.py             ← compute_diagnostic(), stability analysis
│   ├── time_step.py               ← compute_dt() robust Δt estimation
│   └── least_squares.py           ← Least-squares derivative estimation
├── statistical/                   ← Statistical engine
│   ├── __init__.py
│   └── engine.py                  ← StatisticalPredictionEngine (EMA/Holt)
└── ensemble/                      ← Ensemble predictor
    ├── __init__.py
    └── predictor.py               ← EnsembleWeightedPredictor (PredictionPort, not PredictionEngine)
```

**NOT in engines/:** `infrastructure/ml/interfaces.py` — stays at ml/ root (cross-cutting)

---

## Import Examples

```python
# Public API (unchanged - backward compatible)
from infrastructure.ml.engines import (
    EngineFactory,           # Factory + registry
    register_engine,         # Decorator for auto-registration
    discover_engines,        # Plugin discovery
    BaselineMovingAverageEngine,  # Embedded in factory
    TaylorPredictionEngine,
    StatisticalPredictionEngine,
    EnsembleWeightedPredictor,
)

# Subpackage imports (new paths)
from infrastructure.ml.engines.core import EngineFactory, register_engine
from infrastructure.ml.engines.baseline import predict_moving_average, BaselineConfig
from infrastructure.ml.engines.taylor import (
    TaylorPredictionEngine,
    estimate_derivatives,
    compute_dt,
    project,
)
from infrastructure.ml.engines.statistical import StatisticalPredictionEngine
from infrastructure.ml.engines.ensemble import EnsembleWeightedPredictor
```

---

## Engine Registration

```python
# Auto-registration with decorator (recommended)
from infrastructure.ml.engines.core import register_engine
from infrastructure.ml.interfaces import PredictionEngine, PredictionResult

@register_engine("my_custom_engine")
class MyCustomEngine(PredictionEngine):
    @property
    def name(self) -> str:
        return "my_custom_engine"
    
    def can_handle(self, n_points: int) -> bool:
        return n_points >= 5
    
    def predict(self, values, timestamps=None) -> PredictionResult:
        # ... implementation
        pass

# Manual registration (alternative)
from infrastructure.ml.engines import EngineFactory
EngineFactory.register("my_engine", MyCustomEngine)

# Plugin discovery
from infrastructure.ml.engines import discover_engines
discovered = discover_engines("my_package.engines")
```

---

## Deprecated Adapters

**Deprecated adapters removed (2026-03-20):**

**baseline/adapter.py** — `BaselinePredictionAdapter` ❌ DELETED
- **Migrated to:** `EngineFactory.create("baseline_moving_average").as_port()`
- **Files updated:** `ml_service/runners/wiring/container.py`, `ml_service/api/services/prediction_service.py`

**taylor/adapter.py** — `TaylorPredictionAdapter` ❌ DELETED
- **Migrated to:** `TaylorPredictionEngine(...).as_port()`
- **Files updated:** `tests/unit/infrastructure/test_taylor_adapter.py`

**taylor/adapter.py** — `KalmanFilterAdapter` ✅ MOVED
- **New location:** `infrastructure/ml/filters/kalman_adapter.py`
- **Import:** `from infrastructure.ml.filters import KalmanFilterAdapter`

---

## Architecture Notes

**Why `ensemble/` is special:**
- `EnsembleWeightedPredictor` implements `PredictionPort`, NOT `PredictionEngine`
- Wraps N `PredictionPort` instances (not raw values/timestamps)
- Cannot be used with `EngineFactory.create()` 
- Separate folder signals this architectural difference

**Why `interfaces.py` stays at ml/ root:**
- Defines `PredictionEngine` (used by engines/)
- Defines `SignalFilter` (used by filters/)
- Defines `PredictionEnginePortBridge` (used everywhere)
- Cross-cutting interface for ALL of ml/ (engines, filters, cognitive, anomaly)
- Moving it into `engines/core/` would create wrong dependency direction

**Taylor package design:**
- Math modules (`types`, `derivatives`, `polynomial`, `diagnostics`, `time_step`, `least_squares`) are pure functions
- `engine.py` orchestrates the math (delegates to math modules)
- `adapter.py` bridges to domain layer (PredictionPort)
- `math.py` is a backward-compat facade for old `from .taylor_math import ...` callers
