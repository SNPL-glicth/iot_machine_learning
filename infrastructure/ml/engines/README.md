# infrastructure/ml/engines

Prediction engines — time-series forecasting implementations.

**Reorganized:** 2026-03-20

## Package Structure

### 📁 core/
Factory + registry + auto-discovery (2 files)
- `factory.py` (212 lines) — `EngineFactory` central registry + `@register_engine` decorator + `discover_engines()` plugin discovery + `BaselineMovingAverageEngine` embedded fallback

### 📁 baseline/
Simple moving average engine with adaptive window (2 files)
- `engine.py` (62 lines) — `predict_moving_average()` pure function + `BaselineConfig` + `BaselineMetadata`
- `factory.py` (embedded) — `BaselineMovingAverageEngine` with adaptive window (P3) + `record_actual()` error tracking (P1)
- `adapter.py` (82 lines) — **DEPRECATED** `BaselinePredictionAdapter` (PredictionPort wrapper)

### 📁 taylor/
Taylor series prediction engine with scale-relative threshold and Savitzky-Golay smoothing (9 files)
- `engine.py` (195 lines) — `TaylorPredictionEngine` orchestrator with optional `smooth_window` (P2)
- `engine_helpers.py` (189 lines) — `sanitize_inputs()`, `classify_trend()` with scale-relative threshold (P2)
- `prediction_pipeline.py` (188 lines) — Pipeline with Savitzky-Golay pre-smoothing when `smooth_window >= 3` (P2)
- `types.py` (132 lines) — `TaylorCoefficients`, `TaylorDiagnostic`, `DerivativeMethod`
- `derivatives.py` (129 lines) — backward_differences, central_differences, least_squares_fit
- `polynomial.py` (86 lines) — `project()`, `compute_local_fit_error()`
- `diagnostics.py` (95 lines) — `compute_accel_variance`, `compute_stability_indicator`
- `time_step.py` (47 lines) — `compute_dt()` robust Δt estimation
- `least_squares.py` (105 lines) — Least-squares derivative estimation

### 📁 statistical/
EMA/Holt-based forecasting with online alpha adjustment (1 file)
- `engine.py` (481 lines) — `StatisticalPredictionEngine` double exponential smoothing + online alpha micro-adjustment (P4)

### 📁 lightgbm/
Gradient-boosting regressor for non-linear patterns (3 files)
- `engine.py` (226 lines) — `LightGBMPredictionEngine` with lazy lightgbm import + graceful fallback (P5)
- `feature_builder.py` (141 lines) — Stateless feature extraction (delta, rolling mean, lag features)
- `__init__.py` (5 lines) — Public exports

### 📁 adaptive_ensemble/
Regime-based meta-engine with fallback chain (2 files)
- `engine.py` (170 lines) — `AdaptiveEnsembleEngine` routes noisy→Statistical, trending→Taylor, stable→Baseline (P6)
- `__init__.py` (5 lines) — Public exports

### 📁 ensemble/
Weighted combination of multiple engines (1 file)
- `predictor.py` (291 lines) — `EnsembleWeightedPredictor` (implements `PredictionPort`, NOT `PredictionEngine`)

### 📁 kalman/
Kalman filter prediction engine (3 files)
- `engine.py` — `KalmanPredictionEngine` with adaptive Q/R
- `engine_helpers.py` — Kalman-specific helpers
- `kalman_cv_math.py` — Constant-velocity Kalman math

### 📁 multivariate/
Multi-sensor correlation engine (3 files)
- `engine.py` — `MultivariatePredictionEngine` with PCA online
- `correlation_tracker.py` — Cross-sensor correlation tracking
- `pca_online.py` — Online PCA for dimensionality reduction

### 📁 seasonal/
Seasonal pattern detection and prediction (4 files)
- `engine.py` — `SeasonalPredictionEngine` with cycle detection
- `cycle_detector.py` — Automatic cycle period detection
- `resampler.py` — Time-series resampling for irregular intervals
- `config.json` — Seasonal engine configuration

### 📁 statistical/
- `param_optimizer.py` — Online parameter optimization for EMA/Holt (added complement to engine.py)

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
│   ├── least_squares.py           ← Least-squares derivative estimation
│   └── engine_helpers.py          ← sanitize_inputs, classify_trend
├── statistical/                   ← Statistical engine
│   ├── __init__.py
│   ├── engine.py                  ← StatisticalPredictionEngine (EMA/Holt + online alpha)
│   └── param_optimizer.py         ← Online parameter optimization
├── lightgbm/                      ← LightGBM regressor (optional dependency)
│   ├── __init__.py
│   ├── engine.py                  ← LightGBMPredictionEngine
│   └── feature_builder.py       ← Stateless feature extraction
├── adaptive_ensemble/             ← Regime-based meta-engine
│   ├── __init__.py
│   └── engine.py                  ← AdaptiveEnsembleEngine
├── ensemble/                      ← Ensemble predictor
│   ├── __init__.py
│   └── predictor.py               ← EnsembleWeightedPredictor (PredictionPort, not PredictionEngine)
├── kalman/                        ← Kalman filter engine
│   ├── __init__.py
│   ├── engine.py                  ← KalmanPredictionEngine
│   ├── engine_helpers.py          ← Kalman helpers
│   └── kalman_cv_math.py         ← Constant-velocity Kalman math
├── multivariate/                  ← Multivariate correlation engine
│   ├── __init__.py
│   ├── engine.py                  ← MultivariatePredictionEngine
│   ├── correlation_tracker.py     ← Cross-sensor correlation
│   └── pca_online.py             ← Online PCA
└── seasonal/                      ← Seasonal prediction engine
    ├── __init__.py
    ├── engine.py                  ← SeasonalPredictionEngine
    ├── cycle_detector.py          ← Cycle detection
    ├── resampler.py              ← Interval resampling
    └── config.json               ← Engine configuration
```

## Confidence Floor

All engines share a unified confidence floor via `CONFIDENCE.MIN_CONFIDENCE` in `core/parameters/numerical_constants.py`:

| Engine | Antes (2026-05) | Después (2026-06) |
|--------|----------------|-------------------|
| All engines | 0.3 | **0.5** |

Razón: datos industriales ruidosos requieren un piso más alto para mantener credibilidad operativa frente a operadores. El floor de 0.3 generaba confianzas fusionadas de 0.29 en el dataset ALPLA; con 0.5 subió a 0.55.

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
    LightGBMPredictionEngine,      # P5 (optional dependency)
    AdaptiveEnsembleEngine,        # P6 (lightweight regime router)
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
from infrastructure.ml.engines.lightgbm import LightGBMPredictionEngine
from infrastructure.ml.engines.adaptive_ensemble import AdaptiveEnsembleEngine
from infrastructure.ml.engines.ensemble import EnsembleWeightedPredictor
from infrastructure.ml.engines.kalman import KalmanPredictionEngine
from infrastructure.ml.engines.multivariate import MultivariatePredictionEngine
from infrastructure.ml.engines.seasonal import SeasonalPredictionEngine
```

---

## Engine Registration

**⚠️ 2026-06-20:** Always use **relative imports** inside `engines/__init__.py` and engine subpackages. Absolute FQN imports (`from iot_machine_learning.infrastructure.ml.engines.core...`) create duplicate `EngineFactory` classes when both `/path/to/project` and `/path/to/project/..` are on `sys.path`. Use `.core`, `.taylor`, etc.

```python
# ✅ Correct — relative import (works regardless of sys.path entry)
from .core import EngineFactory

# ❌ Wrong — creates duplicate EngineFactory class
from iot_machine_learning.infrastructure.ml.engines.core import EngineFactory
```

**Note:** `Prediction` domain entity uses `confidence_score`, NOT `confidence`. When accessing prediction results in adapters:
```python
# ✅ Correct
confidence = prediction.confidence_score
# ❌ AttributeError
confidence = prediction.confidence
```

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

**Engine Summary:**

| Engine | File(s) | Strategy |
|--------|---------|----------|
| Taylor | `taylor/` (9 files) | Taylor series + Savitzky-Golay smoothing |
| Statistical | `statistical/` (2 files) | EMA/Holt double exponential smoothing |
| Baseline | `baseline/` (2 files) | Simple moving average, adaptive window |
| Kalman | `kalman/` (3 files) | Kalman filter, adaptive Q/R |
| LightGBM | `lightgbm/` (3 files) | Gradient-boosting regressor (opt dep) |
| Adaptive Ensemble | `adaptive_ensemble/` (1 file) | Regime-routing meta-engine |
| Multivariate | `multivariate/` (3 files) | PCA online, cross-sensor correlation |
| Seasonal | `seasonal/` (4 files) | Cycle detection + seasonal prediction |
