# Changelog — iot_machine_learning

Todas las fases de desarrollo del motor ML cognitivo para **Sandevistan**.

---

## [Unreleased]

### Architectural Hardening (1096 tests)

**Respuesta a 3 advertencias arquitectónicas:**

- **⚠️1 Pipeline Latency:** `PipelineTimer` dataclass en `cognitive/types.py` — timing per-phase (perceive, predict, inhibit, adapt, fuse, explain). Budget guard: si perceive+predict excede `budget_ms` (default 500ms), corta a fallback. `last_pipeline_timing` property. `pipeline_timing` en `PredictionResult.metadata`.
- **⚠️2 Orchestrator Complexity Guard:** Meta-tests: `orchestrator.py` ≤ 300 líneas, sin numpy/scipy, delega a 5 sub-módulos. `ARCHITECTURE.md` con 7 reglas arquitectónicas enforced.
- **⚠️3 Legacy Sunset Plan:** `MIGRATION_SCORECARD.md` con inventario completo legacy vs agnóstico. 4 fases de sunset. Meta-test `TestMigrationScorecard` valida dual interface.

**Files:** 3 modified, 3 created | **Tests:** +21 (1075 → 1096)

---

## Phase 4: Technical Debt Cleanup (1075 tests)

**Hallazgos resueltos:** DEBT-1, DEBT-4, COG-3, COG-4

- **DEBT-1:** `safe_series_id_to_int(series_id, fallback=0)` en `input_guard.py`. Reemplaza 14 `int(series_id)` inseguros en 7 archivos. Nunca lanza excepciones, loguea debug para IDs no numéricos.
- **DEBT-4:** `dataclasses.replace()` en `PredictionDomainService` reemplaza reconstrucción manual de `Prediction` (12 campos × 2 sitios).
- **COG-3:** `MetaDiagnostic` deprecated con `DeprecationWarning`. `record_actual()` usa estado interno (`_last_regime`, `_last_perceptions`) en vez de leer de `MetaDiagnostic`.
- **COG-4:** `template_generator._determine_severity()` delega a `AnomalySeverity.from_score()` — fuente única de verdad.

**Files:** 12 modified, 2 created | **Tests:** +19 (1056 → 1075)

---

## Phase 3: Extensibility & DI (1056 tests)

**Hallazgos resueltos:** ROB-1, ROB-2, MOD-2

- **ROB-1:** `@register_engine("name")` decorator auto-registra engines en `EngineFactory`. `discover_engines("package")` escanea paquetes.
- **ROB-2:** `DetectorRegistry` + `@register_detector("name")` para sub-detectores de anomalía.
- **MOD-2:** `VotingAnomalyDetector(sub_detectors=[...])` — DI de detectores custom. `create_default_detectors()` como factory.

**Files:** 3 modified, 1 created | **Tests:** +21 (1035 → 1056)

---

## Phase 2: Interface Consolidation (1035 tests)

**Hallazgos resueltos:** ARQ-1, ARQ-2

- **ARQ-1:** `PredictionEnginePortBridge(PredictionPort)` — adapter genérico que wrappea cualquier `PredictionEngine` como `PredictionPort`. `engine.as_port()` one-liner.
- **ARQ-2:** `EngineFactory.create_as_port("name")` combina create + bridge. `get_engine_for_sensor()` deprecated sin import inverso.
- Adapters manuales (`TaylorPredictionAdapter`, `CognitivePredictionAdapter`) deprecated.

**Files:** 6 modified, 1 created | **Tests:** +21 (1014 → 1035)

---

## Phase 1-Cog: Cognitive Unification (1014 tests)

**Hallazgos resueltos:** COG-1, COG-2

- **COG-1:** `_classify_regime()` unificado en `domain/entities/series/structural_analysis.py`. `SignalAnalyzer` delega a domain.
- **COG-2:** `SignalAnalyzer` retorna `StructuralAnalysis` (domain) en vez de `SignalProfile` (infra). `SignalProfile` deprecated.

**Files:** 10 modified | **Tests:** 0 new (zero regressions)

---

## ExplanationRenderer (1014 tests)

- `ExplanationRenderer` en `application/explainability/` — summary, technical report, structured JSON.
- 5 clasificaciones metacognitivas: certainty, disagreement, cognitive stability, overfit risk, engine conflict.
- Regla: renderer solo transforma, nunca piensa. Sin imports de infrastructure.

**Files:** 2 created | **Tests:** +38 (976 → 1014)

---

## ExplanationBuilder (976 tests)

- `ExplanationBuilder` en `infrastructure/ml/cognitive/` — fluent API con fases dinámicas.
- Fases reflejan lo que realmente pasó (emergente), no un pipeline idealizado.
- Integrado en `MetaCognitiveOrchestrator.predict()`. `last_explanation` property.

**Files:** 1 created, 2 modified | **Tests:** +16 (960 → 976)

---

## Explainability Domain Layer (960 tests)

- `domain/entities/explainability/` — `Explanation`, `ReasoningTrace`, `ContributionBreakdown`, `SignalSnapshot`.
- Pure value objects, frozen, JSON-serializable. Zero imports de infrastructure.
- `Explanation.minimal(series_id)` factory para casos simples.

**Files:** 5 created | **Tests:** +38 (922 → 960)

---

## Filter Infrastructure Expansion (922 tests)

- `EMASignalFilter` (fijo + adaptativo por innovación)
- `MedianSignalFilter` (sliding window, robusto a spikes)
- `FilterChain` (pipeline composable: Median → Kalman, Median → EMA)
- `FilterDiagnostic` (noise_reduction, distortion, lag)
- `KalmanSignalFilter` extended con adaptive Q

**Files:** 4 created, 2 modified | **Tests:** +77 (845 → 922)

---

## Entity Reorganization + Wiring (845 tests)

- `domain/entities/` reorganizado en `series/`, `patterns/`, `results/`, `iot/`, `explainability/`.
- Root-level facades para 100% backward compatibility.
- Taylor → `StructuralAnalysis` wired vía `from_taylor_diagnostic()`.
- Pattern detection enriquece resultados con structural metadata.
- `train_all(values, timestamps=...)` forwards a todos los detectores.
- Legacy `get_default_range()` y `select_engine_for_sensor()` emiten `DeprecationWarning`.

**Files:** ~30 modified/created | **Tests:** +15 (830 → 845)

---

## Structural Analysis (830 tests)

- `StructuralAnalysis` frozen dataclass (domain): slope, curvature, stability, accel_variance, noise_ratio, regime.
- `compute_structural_analysis(values, timestamps)` — pure function, median Δt.
- `from_taylor_diagnostic()` bridge reutiliza diagnósticos Taylor existentes.
- `SensorWindow.structural_analysis` y `TimeSeries.structural_analysis` properties.

**Files:** 3 created, 3 modified | **Tests:** +61 (769 → 830)

---

## Temporal Anomaly Detection (769 tests)

- `VotingAnomalyDetector` expandido a 8 votos: velocity z-score, acceleration z-score, IF 3D, LOF 3D.
- `train(values, timestamps=...)` acepta timestamps.
- `statistical_methods.py` extended con estadísticas temporales.

**Files:** ~5 modified | **Tests:** +133 (636 → 769)

---

## Taylor Modular Redesign (636 tests)

- `taylor/` package: types, derivatives (backward, central, least_squares), polynomial, diagnostics, time_step, least_squares.
- `TaylorPredictionEngine` acepta `DerivativeMethod` param.
- `TaylorDiagnostic` con slope, curvature, stability, accel_variance, local_fit_error.
- `taylor_math.py` como facade backward-compatible.

**Files:** 7 created, 1 modified | **Tests:** +60 (576 → 636)

---

## Cognitive Engine Audit (576 tests)

**6 fases de auditoría y hardening:**

1. **Hardcoding Elimination:** Todos los umbrales/cutoffs → constructor kwargs con defaults documentados.
2. **Time-Series Integrity:** `TemporalValidator`, timestamp validation en `SensorReading`, `TemporalDiagnostic`.
3. **Context Boundary:** Core inference path limpio. Legacy `sensor_type` deprecated.
4. **Persistence Integrity:** `save_prediction` ahora persiste `engine_name`, `trend`. `save_anomaly_event` persiste `anomaly_score`, `method_votes`, `audit_trace_id`.
5. **Metrics & Validation:** `input_guard.py` (guards), `MetricsCollector` extended (persistence, anomaly counters).
6. **Alignment:** 11 principios cognitivos verificados.

**Files:** 12 modified, 4 created | **Tests:** +163 (413 → 576)

---

## UTSAE Agnostic Migration (413 tests)

**Migración completa de `sensor_id: int` → `series_id: str`:**

- `TimeSeries`, `TimePoint`, `SeriesProfile`, `SeriesContext`, `Threshold` creados.
- Entidades core migradas: `Prediction`, `AnomalyResult`, `PatternResult`.
- DTOs migrados: `PredictionDTO`, `AnomalyDTO`, `PatternDTO`.
- Ports: dual interface (`SensorWindow` + `TimeSeries`).
- `SignalFilter(series_id: str)`, `KalmanFilter(Dict[str, ...])`.
- `FeatureFlags`: dual interface (series_id + sensor_id).
- `AccessControl`: dual interface (series_id + sensor_id).
- `select_engine_for_series(profile, flags)` — selección por datos.
- `classify_severity_agnostic()` — sin lenguaje IoT.
- 27 archivos propagados. Zero regresiones.

**Files:** ~30 modified/created | **Tests:** 413 total

---

## Enterprise Phase (213 tests)

**Fase 3 de UTSAE — funcionalidades enterprise:**

- Pattern detection: `DeltaSpikeClassifier`, `CUSUMDetector`, `PELTDetector`, `RegimeDetector`.
- Anomaly detection: `VotingAnomalyDetector` (IF + Z-score + IQR + LOF).
- Engines: `EnsembleWeightedPredictor` (dynamic weight auto-tuning).
- Explainability: `TaylorFeatureImportance`, `CounterfactualExplainer`.
- Security: `FileAuditLogger` (ISO 27001), `AccessControlService` (RBAC).
- Adapters: `InMemoryPredictionCache` (LRU+TTL), `BatchPredictor` (ThreadPool+CircuitBreaker).
- Feature flags enterprise (all default false).

**Files:** ~25 created | **Tests:** +115 (98 → 213)

---

## Foundation (98 tests)

**Fase 1-2 de UTSAE — core ML:**

- Arquitectura hexagonal: domain (puro) → application (use cases) → infrastructure (implementaciones).
- `TaylorPredictionEngine` + `taylor_math.py`.
- `BaselineMovingAverageEngine`.
- `KalmanSignalFilter` + `kalman_math.py`.
- `PredictionDomainService`, `AnomalyDomainService`, `PatternDomainService`.
- `PredictSensorValueUseCase`, `DetectAnomaliesUseCase`, `AnalyzePatternsUseCase`.
- `EngineFactory` (registry + create).
- FastAPI endpoints.
- SQL Server adapter.

**Tests:** 98 total

---

## Test Progression

```
98 → 213 → 413 → 576 → 636 → 769 → 830 → 845 → 922 → 960 → 976 → 1014 → 1035 → 1056 → 1075 → 1096
```
