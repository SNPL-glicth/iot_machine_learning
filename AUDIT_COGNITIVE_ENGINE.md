# Audit: Temporal Cognitive Engine — Architectural Integrity

**Date:** 2026-02-12 (original) | **Última actualización:** 2026-02-12  
**Scope:** `iot_machine_learning/` — domain, application, infrastructure, ml_service  
**Tests:** 1096 passed, 6 skipped, 0 failures  
**Objective:** Verify and harden the system's integrity as a temporal cognitive engine.  
Original scope: audit, hardening, and architectural correction only.  
Subsequent phases added intelligence features (explainability, filters, cognitive orchestration).

---

## Phase 1: Hardcoding Elimination

**Goal:** All numeric thresholds, cutoffs, model parameters, and magic numbers must be
configurable via constructor injection or config dataclasses. Module-level constants
serve only as documented fallback defaults.

### Files Modified

| File | What Changed |
|------|-------------|
| `ml_service/config/ml_config.py` | `AnomalyConfig` extended: `lof_max_neighbors`, `min_training_points`, `z_vote_lower/upper`, `severity_*_max`. `EngineConfig` extended: `taylor_trend_threshold`, `taylor_min/max_confidence`. |
| `infrastructure/ml/engines/taylor_engine.py` | `_TREND_THRESHOLD`, `_MIN_CONFIDENCE`, `_MAX_CONFIDENCE`, `_CLAMP_MARGIN_PCT` → constructor kwargs with module-level fallbacks. |
| `infrastructure/ml/anomaly/voting_anomaly_detector.py` | `random_state=42`, `n_estimators=100`, `min(20, ...)`, `< 50` → constructor kwargs: `n_estimators`, `random_state`, `lof_max_neighbors`, `min_training_points`. |
| `infrastructure/ml/anomaly/statistical_methods.py` | `compute_z_vote` thresholds 2.0/3.0 → `lower`/`upper` params. |
| `domain/entities/anomaly.py` | `AnomalySeverity.from_score` cutoffs 0.3/0.5/0.7/0.9 → `none_max`, `low_max`, `medium_max`, `high_max` kwargs. |
| `domain/entities/prediction.py` | `PredictionConfidence.from_score` cutoffs 0.2/0.4/0.7/0.9 → `very_low_max`, `low_max`, `medium_max`, `high_max` kwargs. |
| `domain/services/anomaly_domain_service.py` | Hardcoded `0.5`/`0.7` confidence → `min_consensus_confidence`, `single_detector_confidence` constructor kwargs. |
| `infrastructure/ml/patterns/regime_detector.py` | `random_state=42`, `n_init=10` → constructor kwargs. |

### Design Decision
All changes are **backward-compatible**: existing callers that pass no kwargs get the
same behavior as before. New callers can override any threshold via DI or config.

---

## Phase 2: Time-Series Integrity

**Goal:** Validate temporal order, timestamp finiteness, gap detection, duplicate
handling, and out-of-order rejection at domain boundaries.

### Files Created

| File | Purpose |
|------|---------|
| `domain/validators/temporal.py` | `validate_timestamps()`, `diagnose_temporal_quality()`, `sort_and_deduplicate()`, `TemporalDiagnostic` dataclass. |
| `tests/unit/domain/test_temporal_validator.py` | 31 tests covering monotonicity, gaps, duplicates, out-of-order, edge cases. |

### Files Modified

| File | What Changed |
|------|-------------|
| `domain/entities/sensor_reading.py` | `SensorReading.__post_init__` now validates `timestamp` finiteness. `SensorWindow.temporal_diagnostic` property added for observability. |

### Key Invariants Enforced
- `SensorReading` rejects `NaN`/`Inf` timestamps at construction.
- `SensorWindow.temporal_diagnostic` provides non-blocking quality metrics (monotonicity, gap count, duplicate count, median Δt).
- `sort_and_deduplicate()` available as repair function at adapter boundaries.

---

## Phase 3: Context Definition Boundary

**Goal:** Verify the ML engine derives context **only** from the incoming data payload —
no hidden sensor-type assumptions, no ambient state, no implicit domain logic in inference.

### Findings

| Layer | Status | Detail |
|-------|--------|--------|
| `infrastructure/ml/engines/` | ✅ Clean | Zero references to `sensor_type` or `sensor_ranges`. |
| `infrastructure/ml/anomaly/` | ✅ Clean | Zero references to `sensor_type` or `sensor_ranges`. |
| `domain/services/prediction_domain_service.py` | ✅ Clean | Operates on `SensorWindow` values only. |
| `domain/services/anomaly_domain_service.py` | ✅ Clean | Operates on `SensorWindow` values only. |
| `domain/services/severity_rules.py` | ⚠️ Legacy | `classify_severity(sensor_type)` uses hardcoded `DEFAULT_SENSOR_RANGES`. |

### Action Taken
- `compute_risk_level()` and `classify_severity()` now emit `DeprecationWarning` at runtime.
- Agnostic path `classify_severity_agnostic(value, threshold=...)` already exists and uses `Threshold` from payload.
- `sensor_ranges.py` already marked `.. deprecated::` in docstring.

---

## Phase 4: ML Relational Persistence Integrity

**Goal:** No silent data loss. Every inference result must be fully traceable in the DB.

### Issues Found & Fixed

| Issue | Before | After |
|-------|--------|-------|
| `save_prediction` missing fields | Only `predicted_value`, `confidence` stored | Now stores `engine_name`, `trend` |
| `save_anomaly_event` missing fields | Only `event_type`, `message` stored | Now stores `anomaly_score`, `anomaly_confidence`, `method_votes` (JSON), `audit_trace_id` |
| `_get_or_create_model_id` wrong engine | Always used `BASELINE_MOVING_AVERAGE` metadata | Now uses actual `engine_name` parameter |
| `get_latest_prediction` data loss | Returned `engine_name="unknown"`, `trend="stable"` | Now reads `engine_name`, `trend` from DB |
| Hardcoded horizon multiplier | `horizon_steps * 10` | Configurable `horizon_minutes_per_step` param |
| Logging level | `logger.debug` on persistence | Upgraded to `logger.info` for observability |

### Transaction Boundaries
The adapter operates on a caller-provided `Connection`. Transaction commit/rollback
is the caller's responsibility — this is correct for the Adapter pattern. The adapter
itself never calls `commit()` or `rollback()`.

---

## Phase 5: Metrics & Validation Hardening

**Goal:** Input validation at use-case boundaries, persistence counters, anomaly counters.

### Files Created

| File | Purpose |
|------|---------|
| `domain/validators/input_guard.py` | `guard_sensor_id()`, `guard_series_id()`, `guard_window_size()`, `guard_no_future_timestamps()`, `guard_finite_value()`. |
| `tests/unit/domain/test_input_guard.py` | 28 tests for all guard functions. |

### Files Modified

| File | What Changed |
|------|-------------|
| `ml_service/metrics/performance_metrics.py` | `MetricsCollector` extended: `record_persistence_success/failure()`, `record_anomaly_result(is_anomaly)`. `MLMetrics` extended: `persistence_successes/failures`, `total_anomalies_detected/normal`. Module-level convenience functions added. |

---

## Phase 6: Cognitive Engine Alignment

| Principle | Status | Evidence |
|-----------|--------|----------|
| **Determinism** | ✅ | Same inputs + config → same outputs. `random_state` configurable. |
| **Observability** | ✅ | Structured logging, `trace_id` on every prediction/anomaly, `MetricsCollector` with 12 metric types. `PipelineTimer` per-phase latency. |
| **Configurability** | ✅ | All thresholds injectable. `GlobalMLConfig` + `FeatureFlags` as config sources. |
| **Inference ↔ Memory separation** | ✅ | Memory recall is enrichment-only (step 3.5). Never modifies `predicted_value`, `confidence_score`, `trend`. |
| **Domain ↔ Infra separation** | ✅ | `domain/` has zero infra imports. All engines/detectors behind ports. |
| **No hardcoded behavior** | ✅ | All magic numbers moved to config with documented defaults. Severity unified via `AnomalySeverity.from_score()`. |
| **Temporal integrity** | ✅ | `TemporalValidator`, timestamp validation on `SensorReading`, `SensorWindow.temporal_diagnostic`. Temporal anomaly votes. |
| **Context from payload** | ✅ | Core inference path clean. Legacy sensor_type path deprecated. |
| **Persistence integrity** | ✅ | Full inference state persisted: engine, trend, score, confidence, votes, trace_id. |
| **Input validation** | ✅ | `input_guard.py` at use-case boundaries. `safe_series_id_to_int()` for identity conversion. |
| **Fail-safe design** | ✅ | Detector/audit/memory/persistence failures don't break pipeline. Budget guard cuts to fallback. |
| **Explainability** | ✅ | `Explanation` (domain) → `ExplanationBuilder` (infra) → `ExplanationRenderer` (app). Full reasoning trace. |
| **Extensibility** | ✅ | `@register_engine`, `@register_detector`, `engine.as_port()`. No modification of existing code needed. |
| **Architectural guards** | ✅ | Meta-tests enforce orchestrator ≤300 lines, no numpy/scipy, delegation pattern. |

---

## Subsequent Phases (post-original audit)

The following phases were implemented after the original 6-phase audit:

### Taylor Modular Redesign
- `taylor/` package: types, derivatives (3 methods), polynomial, diagnostics, time_step, least_squares
- `TaylorPredictionEngine` accepts `DerivativeMethod` param
- `TaylorDiagnostic` with slope, curvature, stability, accel_variance
- **Tests:** 636 total (+60)

### Temporal Anomaly Detection
- `VotingAnomalyDetector` expanded to 8 votes: velocity z-score, acceleration z-score, IF 3D, LOF 3D
- `train(values, timestamps=...)` accepts timestamps
- `statistical_methods.py` extended with temporal statistics
- **Tests:** 769 total (+133)

### Structural Analysis (Shared)
- `StructuralAnalysis` frozen dataclass (domain): slope, curvature, stability, regime, noise_ratio
- `compute_structural_analysis(values, timestamps)` pure function (domain validator)
- `from_taylor_diagnostic()` bridge reuses Taylor diagnostics
- `SensorWindow.structural_analysis` and `TimeSeries.structural_analysis` properties
- **Tests:** 830 total (+61)

### Entity Reorganization
- `domain/entities/` split into `series/`, `patterns/`, `results/`, `iot/`, `explainability/`
- Root-level facades for 100% backward compatibility
- **Tests:** 845 total (zero regressions)

### Wiring & Deprecation
- Taylor → `StructuralAnalysis` wired via `from_taylor_diagnostic()`
- Pattern detection enriches results with structural metadata
- `train_all(values, timestamps=...)` forwards to all detectors
- Legacy `get_default_range()` and `select_engine_for_sensor()` emit DeprecationWarning
- **Tests:** 845 total (+15)

### Filter Infrastructure Expansion
- `EMASignalFilter` (fixed + adaptive), `MedianSignalFilter`, `FilterChain`, `FilterDiagnostic`
- `KalmanSignalFilter` extended with adaptive Q
- All implement `SignalFilter(series_id: str)`. Composable pipelines.
- **Tests:** 922 total (+77)

### Explainability Layer
- **Domain:** `Explanation`, `ReasoningTrace`, `ContributionBreakdown`, `SignalSnapshot` (pure value objects)
- **Infrastructure:** `ExplanationBuilder` (fluent API, dynamic phases)
- **Application:** `ExplanationRenderer` (summary, technical report, structured JSON)
- 5 metacognitive classifications: certainty, disagreement, cognitive stability, overfit risk, engine conflict
- **Tests:** 1014 total (+92)

### Cognitive Unification (COG-1, COG-2)
- `SignalAnalyzer` returns `StructuralAnalysis` (domain) instead of `SignalProfile` (infra)
- `_classify_regime()` single source of truth in domain
- `SignalProfile` deprecated
- **Tests:** 1014 total (zero regressions)

### Interface Consolidation (ARQ-1, ARQ-2)
- `PredictionEnginePortBridge`: wraps any `PredictionEngine` as `PredictionPort`
- `engine.as_port()` one-liner, `EngineFactory.create_as_port()`
- Manual adapters deprecated
- **Tests:** 1035 total (+21)

### Extensibility & DI (ROB-1, ROB-2, MOD-2)
- `@register_engine("name")` decorator + `discover_engines()` scanner
- `DetectorRegistry` + `@register_detector("name")`
- `VotingAnomalyDetector(sub_detectors=[...])` DI
- **Tests:** 1056 total (+21)

### Technical Debt Cleanup (DEBT-1, DEBT-4, COG-3, COG-4)
- `safe_series_id_to_int()` replaces 14 unsafe `int(series_id)` bridges in 7 files
- `dataclasses.replace()` in `PredictionDomainService`
- `MetaDiagnostic` deprecated → use `last_explanation`
- `template_generator` delegates severity to `AnomalySeverity.from_score()`
- **Tests:** 1075 total (+19)

### Architectural Hardening (⚠️1, ⚠️2, ⚠️3)
- `PipelineTimer` per-phase latency (perceive, predict, inhibit, adapt, fuse, explain)
- Budget guard: cuts to fallback if pipeline exceeds `budget_ms` (default 500ms)
- Meta-tests: orchestrator ≤300 lines, no numpy/scipy, delegates to 5 sub-modules
- `ARCHITECTURE.md` with 7 enforced architectural rules
- `MIGRATION_SCORECARD.md` with legacy sunset plan
- **Tests:** 1096 total (+21)

---

## Test Summary

| Phase | New Tests | Total After |
|-------|-----------|-------------|
| Phase 1 (Hardcoding) | 0 (backward-compatible) | 548 |
| Phase 2 (Temporal) | 31 | 548 |
| Phase 5 (Validation) | 28 | 576 |
| Taylor Modular | 60 | 636 |
| Temporal Anomaly | 133 | 769 |
| Structural Analysis | 61 | 830 |
| Entity Reorg + Wiring | 15 | 845 |
| Filter Expansion | 77 | 922 |
| Explainability | 92 | 1014 |
| Interface Consolidation | 21 | 1035 |
| Extensibility & DI | 21 | 1056 |
| Technical Debt | 19 | 1075 |
| Architectural Hardening | 21 | 1096 |
| **Final** | **579 new** | **1096 passed, 6 skipped** |

---

## Documentation

| Document | Purpose |
|----------|---------|
| `README.md` | Full architecture, APIs, test counts, decision log |
| `AUDIT_ARCHITECTURE_DEEP.md` | Original deep audit + resolution status for all findings |
| `AUDIT_COGNITIVE_ENGINE.md` | This file — cognitive engine integrity audit + all phases |
| `AUDIT_UTSAE_AGNOSTIC.md` | UTSAE agnostic migration audit |
| `ARCHITECTURE.md` | 7 enforced architectural rules with meta-tests |
| `MIGRATION_SCORECARD.md` | Legacy IoT → UTSAE agnostic migration inventory + sunset plan |
