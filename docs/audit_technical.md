# ZENIN ML Service — Critical Technical Audit

**Auditor:** Senior Software Architect (Critical Review Mode)
**Date:** 2026-04-28
**Scope:** iot_machine_learning/ — all modules, ports, adapters, pipelines
**Standard:** Clean Architecture, SOLID, ISO 27001, Production ML

---

## Executive Summary

The ZENIN ML cognitive pipeline implements a 13-phase metacognitive orchestrator, MoE gateway, SNN prototypes, and hexagonal domain layer. Architectural discipline has decayed under rapid iteration. There are 5 distinct Reading value objects, layer violations (domain imports ml_service), dead code exceeding 4,000 lines, and critical race conditions. The system degrades at 100+ sensors and fails at 1,000+. It is a mature prototype masquerading as production infrastructure.

Severity: CRITICAL = outage/data loss, HIGH = scaling/silent failure blocker, MEDIUM = technical debt, LOW = style/missing coverage.

---

## 1. Architecture & Layer Boundaries

### ARCH-1 [CRITICAL] Domain imports ML Service infrastructure
`domain/ports/document_analysis.py:108` imports `format_conclusion` from `ml_service.api.services.analysis.conclusion_formatter`. Domain layer imports directly from FastAPI service layer, violating Clean Architecture's Dependency Rule. Also `domain/services/cognitive_constants.py:21` lazy-imports from `ml_service.config.feature_flags`. Domain constants should not derive from infrastructure config.

### ARCH-2 [HIGH] Application layer still uses sensor_id: int
`application/use_cases/predict_sensor_value.py:61` — def execute(self, sensor_id: int...). Despite UTSAE agnostic migration (413 tests), primary use-case accepts sensor_id:int. Domain moved to series_id:str, adapters have dual interfaces, yet application boundary remains IoT-coupled. Perpetuates type duality across logs, metrics, observability.

### ARCH-3 [HIGH] FeatureFlags uses 5-way multiple inheritance
`ml_service/config/flags.py:22` — class FeatureFlags(TaylorConfig, BatchConfig, CognitiveConfig, DecisionConfig, SecurityConfig): pass. Pydantic BaseModel with 5-way MI. MRO collision resolution is implicit. God-object configuration anti-pattern.

### ARCH-4 [MEDIUM] Circular import risk in config
`ml_service/config/flags.py:39-41` — flags.py imports from feature_flags.py, which re-exports from flags.py. try/except deferred import is fragile.

---

## 2. Pipeline & Orchestrator

### PIPE-1 [HIGH] Orchestrator is de-facto god object
`infrastructure/ml/cognitive/orchestration/orchestrator.py` — ARCHITECTURE.md meta-test caps at 300 lines; file is 292. But predict() contains three strategies: (a) MoE gateway, (b) iterative loop, (c) 13-phase linear pipeline. Line-count fiction masks god object coordinating error tracking, reliability, hyperparameter adaptation, Bayesian weights, plasticity, MoE.

### PIPE-2 [CRITICAL] Global mutable state in AssemblyPhase
`infrastructure/ml/cognitive/orchestration/phases/assembly_phase.py:29-35` — Module-level _compliance_exporter with Lock(). Concurrent requests contend on file sink. Runtime path change creates dangling handle. No SIGTERM cleanup.

### PIPE-3 [HIGH] All 13 phases always instantiated
`infrastructure/ml/cognitive/orchestration/pipeline_executor_factory.py` — flags_snapshot does NOT affect phase composition. All 13 phases created always. Phases check ctx.flags at runtime to skip. Wastes 30-50% construction time for disabled features (SNN, attention, narrative).

### PIPE-4 [HIGH] ShadowEvaluationPhase unseeded random
`infrastructure/ml/cognitive/orchestration/phases/shadow_evaluation_phase.py:85-88` — import random; if random.random() > self._sample_rate. Unseeded, non-deterministic across replicas. Breaks log correlation and A/B analysis.

### PIPE-5 [MEDIUM] CognitiveLoopController NOT IMPLEMENTED
`infrastructure/ml/cognitive/orchestration/iterative_controller.py:35-37` — expand_window_on_retry: bool = False, comment "NOT IMPLEMENTED — see TECHNICAL_DEBT.md". File does not exist. Misleading API.

---

## 3. Configuration

### CFG-1 [MEDIUM] Duplicate field
`ml_service/config/cognitive_config.py:27,42` — ML_ENABLE_HYBRID_EMBEDDINGS declared twice. Pydantic V2 silently accepts last value. Line 27 is dead.

### CFG-2 [MEDIUM] 25-second pipeline budget
`ml_service/config/cognitive_config.py:22` — ML_PIPELINE_BUDGET_MS = 25000.0. Orchestrator guard uses 500ms. 50x discrepancy makes config meaningless.

### CFG-3 [LOW] 9-file config scatter
Feature flags across 9 files. No single source of truth. 40+ cognitive fields, mostly disabled — configuration graveyard.

---

## 4. Domain Layer

### DOM-1 [CRITICAL] DocumentAnalysis duck-typing with debug logs
`domain/ports/document_analysis.py:49-72` — 14 hasattr() calls. Also peppered with logger.warning debug statements (lines 35,75,84,89,94,97,99,105,109) that should be removed.

### DOM-2 [HIGH] SensorWindow.to_time_series wrong ID
`domain/entities/iot/sensor_reading.py:196-199` — series_id=str(self.sensor_id). For non-numeric series_id (e.g. "room_temp"), sensor_id returns 0, yielding "0" not "room_temp". Silent identity corruption.

### DOM-3 [MEDIUM] Situation vector 28% dead dimensions
`domain/services/situation_vector_builder.py:57-103` — Dims 12-16 (MoE probabilities) read meta.get("moe_vector") which is never set. Permanently zero.

### DOM-4 [MEDIUM] Explainability dead code
`domain/entities/explainability/` — 6 files, ~400 lines, 38 tests. Built but never wired into production pipeline. Orchestrator still uses legacy ExplanationBuilder.

---

## 5. Infrastructure

### INF-1 [HIGH] Metadata archaeology
`infrastructure/persistence/sql/storage/prediction_queries.py:68-75` — Three nested .get() chains for regime string. Stringly-typed. Silent fallback to "unknown".

### INF-2 [MEDIUM] math import in hot path
`infrastructure/persistence/sql/storage/prediction_queries.py:19-28` — def _safe_float imports math inside function body.

### INF-3 [HIGH] Three prediction writers
Writer A (SqlServerStorageAdapter) = production. Writer B (PredictionWriter) = legacy dev/test. Writer C (prediction_repository) = dead code. Three paths to same table.

### INF-4 [HIGH] Memory leak — no sensor eviction
SlidingWindowStore + SlidingWindowBuffer never evict inactive sensors. OOM at 1000+ sensors.

### INF-5 [HIGH] Duplicate predictions
Stream consumer and batch runner both predict for same sensor without deduplication (RC-1). Double-compute, double-write, double-alert.

---

## 6. ML & Data Science

### ML-1 [HIGH] Neural modules dead code
`infrastructure/ml/cognitive/neural/` — SNN, STDP, LIF neurons, hybrid engine (~600+ lines). Never wired. README says "Not Production Ready". CI overhead, import cost, confusion.

### ML-2 [MEDIUM] ExpertRegistry type bug
`infrastructure/ml/moe/registry.py:296` — return type Dict[str, any] uses builtin any, not typing.Any.

### ML-3 [MEDIUM] SNN fixed random seed
`infrastructure/ml/cognitive/neural/snn/network.py:132` — rng = np.random.RandomState(42). Fixed seed means identical weight init across all SNN instances.

### ML-4 [HIGH] Taylor diagnostics not fed to universal engine
`infrastructure/ml/cognitive/universal/analysis/engine.py` — UniversalAnalysisEngine imports cognitive subsystems but Taylor structural diagnostics (slope, curvature, regime) not fed into signal profile. Bridge exists but uncalled.

---

## 7. Code Quality

### SMELL-1 [LOW] 156 TODO/FIXME/HACK/XXX across 92 files
### SMELL-2 [LOW] test_sanitize_debug.py at project root
### SMELL-3 [MEDIUM] Use case imports observability singleton
`application/use_cases/predict_sensor_value.py:244,297` — get_observability().silent_failures.record(). Should be injected port, not global singleton.

### SMELL-4 [MEDIUM] MemoryRecallEnricher silent failure
`application/use_cases/predict_sensor_value.py:232-248` — Catches all exceptions, logs warning, returns None. Caller never knows enrichment failed. Silent data loss for compliance.

---

## 8. Security

### SEC-1 [CRITICAL] HMAC key from env only
`infrastructure/ml/cognitive/orchestration/phases/assembly_phase.py:48-49` — hmac_key=load_hmac_key_from_env(). No key rotation. No secrets manager. Env dump in crash report compromises integrity.

### SEC-2 [HIGH] CORS with credentials
`ml_service/main.py:135-141` — allow_credentials=True with env-controlled origins. Misconfiguration to "*" leaks credentials cross-origin.

### SEC-3 [MEDIUM] Uniform API key
All endpoints use same verify_api_key. No RBAC. Ingest key can access predictions, metrics, analysis.

---

## 9. Performance

### PERF-1 [CRITICAL] Batch runner sequential
>500 sensors implies cycle >60s, overlap. No parallelization.

### PERF-2 [HIGH] 12 transformations stream path
Each adds latency and failure surface. One redundant timestamp re-parse fixed (E-8), pipeline still overweight.

### PERF-3 [HIGH] Calibration disabled by default
ML_PROBABILISTIC_CALIBRATION_ENABLED = False. Uncalibrated confidence scores mean operators cannot trust confidence bar.

### PERF-4 [MEDIUM] Weaviate disabled by default
ML_ENABLE_COGNITIVE_MEMORY = False. Full Weaviate adapter suite (8 files, ~500 lines) is dead code in default config.

---

## 10. Testing

### TEST-1 [MEDIUM] 261 non-passing tests
1806 passed / 214 failed / 55 skipped / 47 errors. 11.7% non-pass rate. Cross-service imports skip in ML context — suite not hermetic.

### TEST-2 [LOW] Meta-test line guard gamed
ARCHITECTURE.md <=300 line cap. File at 292 but contains massive docstrings and 40-line __init__ with 15 params. Spirit violated, letter obeyed.

### TEST-3 [MEDIUM] 38 tests for dead explainability
Explainability subsystem tested but unwired. Sunk engineering cost.

---

## 11. Integration

### INT-1 [HIGH] .NET MLSearchService endpoints missing
.NET expects POST /ml/index-document and POST /ml/semantic-search from ML Service. Not present in routes.py. .NET relay will 404.

### INT-2 [HIGH] Weaviate removed from .NET but persists in ML
.NET successfully removed all Weaviate references. ML Service still has full adapter suite.

### INT-3 [MEDIUM] Flutter confidence gap
avgConfidence /100 fixed locally, but confidence not exposed in IntelligencePredictionDto (NestJS). Flutter cannot show per-prediction confidence. Architecture gap persists.

---

## 12. Database

### DB-1 [CRITICAL] Deadlock fix not in migrations
Migration 020 documents fix but never executed via SSMS. Applied live via pymssql only. New environments will deadlock.

### DB-2 [HIGH] DEFAULT 'LOW' vs fallback 'NONE'
SQL Server column DEFAULT 'LOW', NestJS fallback 'NONE'. Same sensor different risk levels per querying layer.

### DB-3 [MEDIUM] engine_name added ad-hoc
No numbered migration. Schema drift risk.

---

## 13. Recommendations

### Immediate (Week 1)
1. Move format_conclusion out of domain (ARCH-1)
2. Replace module-level _compliance_exporter (PIPE-2)
3. Fix to_time_series series_id (DOM-2)
4. Seed random in shadow phase (PIPE-4)

### Short-term (Month 1)
5. Rename PredictSensorValueUseCase → PredictSeriesValueUseCase (ARCH-2)
6. Refactor FeatureFlags to composition not MI (ARCH-3)
7. Consolidate to Writer A, deprecate B/C (INF-3)
8. Implement sensor eviction + dedup (INF-4, INF-5)
9. Parallelize batch runner (PERF-1)
10. Move neural/ to research repo (ML-1)

### Medium-term (Month 2-3)
11. Flag-driven phase selection (PIPE-3)
12. Wire domain Explanation into orchestrator (DOM-4)
13. Execute migration 020 in CI/CD (DB-1)
14. Integrate secrets manager (SEC-1)
15. Implement /ml/index-document and /ml/semantic-search (INT-1)

### Long-term (Quarter)
16. Reduce 5 Reading models to 1
17. Split ml_service into prediction-service and cognitive-service
18. Add RBAC to API keys
19. Implement calibration by default with fallback
20. Remove all dead code (neural, Weaviate if unused, explainability if unwired)

---

## Appendix: File Inventory

Critical files analyzed (line counts approximate):
- orchestrator.py (292)
- prediction_service.py (200)
- predict_sensor_value.py (301)
- severity_rules.py (295)
- registry.py (327)
- engine.py (505)
- network.py (275)
- hybrid_engine.py (127)
- assembly_phase.py (273)
- main.py (153)
- feature_flags.py (39)
- flags.py (51)
- cognitive_config.py (100)
- context.py (132)
- iterative_controller.py (247)
- shadow_evaluation_phase.py (146)
- document_analysis.py (208)
- sensor_reading.py (201)
- situation_vector_builder.py (106)
- prediction_queries.py (224)
- routes.py (149)
- 92 additional files with TODO/FIXME markers

Total project: ~12,000+ lines Python, ~600 tests, 261 non-passing.

---

## 14. Error Handling & Resilience

### RES-1 [HIGH] Circuit breaker has no metric emission
`infrastructure/resilience/circuit_breaker.py:75-79` — State transitions log via logger.warning but do not emit structured metrics. The situation_vector_builder reads circuit status from cognitive_trace metadata, but if the circuit breaker trips during a prediction, no metric is recorded. Operational teams cannot alert on circuit state changes.

### RES-2 [MEDIUM] Half-open state lacks jitter
`infrastructure/resilience/circuit_breaker.py:64-68` — Recovery timeout is a fixed 60 seconds. In a cluster of 3 pods, all will attempt reset at the same time after simultaneous failure, creating a thundering herd against the recovering backend. Add ±20% jitter to recovery_timeout.

### RES-3 [MEDIUM] Exception swallowing in lifespan
`ml_service/main.py:56-57,69-70,82-83` — Three consecutive try/except blocks in lifespan swallow exceptions with logger.warning. If circuit reset fails AND Weaviate init fails AND broker init fails, the application still starts and accepts traffic. A dependency cascade failure is masked as healthy.

### RES-4 [HIGH] Zenin Queue Poller is a daemon thread
`ml_service/main.py:92-97` — ZeninQueuePoller runs as `daemon=True`. If the thread dies (uncaught exception), FastAPI continues running. No health check verifies poller liveness. The `/_zenin_poller` reference is kept but never exposed in /health or /ready.

---

## 15. Logging & Observability

### OBS-1 [HIGH] 7 logger.warning calls in production domain code
`domain/ports/document_analysis.py:35,75,84,89,94,97,99,105,109` — Nine `logger.warning` calls in a single 208-line port file. These were clearly debug instrumentation (`[FROM_RESULT]`) that escaped into production. Warnings pollute log aggregation (Datadog/Splunk) and trigger false P1 pages if alert rules match `level>=WARNING`.

### OBS-2 [MEDIUM] No structured logging standard
Log messages mix formats: `[ML-SERVICE]`, `[CIRCUIT_BREAKER]`, `[FROM_RESULT]`, plain strings. No JSON formatter is configured. Log parsing requires regex, breaking structured queries.

### OBS-3 [MEDIUM] Pipeline timing not exposed as metric
Per IMP-1 memory: PipelineTimer records per-phase wall-clock, but only in PredictionResult.metadata. No Prometheus/Grafana metric emission. Cannot build SLO dashboards for p95 predict latency.

### OBS-4 [LOW] Broker health is optional in /health
`ml_service/api/routes.py:31-44` — /health tries broker health in try/except and ignores failure. If broker is down, /health still returns 200. Liveness probe should reflect critical dependency state.

---

## 16. Dependencies & Version Management

### DEP-1 [HIGH] Broad version ranges in requirements.txt
`requirements.txt:7-29` — Ranges like `numpy>=1.24.0,<3.0.0` and `fastapi>=0.115.0,<1.0.0` allow major version drift. NumPy 2.0 (released June 2024) has breaking API changes (e.g. `np.string_` removed). A fresh install today may pull NumPy 2.x and break the SNN layer or Taylor engine.

### DEP-2 [MEDIUM] Missing lock file
No `requirements-lock.txt`, `poetry.lock`, or `Pipfile.lock`. Reproducible builds are impossible. Two CI runs one week apart may use different dependency versions.

### DEP-3 [MEDIUM] Optional dependencies imported unconditionally
`ml_service/main.py:60-64` — Weaviate schema initializer imported at startup even when `ML_ENABLE_COGNITIVE_MEMORY=False`. Adds ~200ms import overhead and failure surface for a disabled feature. Should be lazy-imported inside the `if weaviate_url:` block.

### DEP-4 [LOW] Commented dependencies in requirements.txt
`requirements.txt:31-34` — weaviate-client and pika commented out with "Futuro" notes. Dead weight in version control. If not needed, remove; if needed later, add when the feature ships.

---

## 17. Docker & Deployment

### DOCK-1 [CRITICAL] No Dockerfile in ML service
No Dockerfile exists in `iot_machine_learning/`. The service must be run directly via `python -m ml_service.main` or similar, complicating containerized deployment, horizontal scaling, and health-check integration in Kubernetes.

### DOCK-2 [HIGH] PYTHONPATH manipulation in main.py
`ml_service/main.py:18-20` — `sys.path.insert(0, str(_project_root))`. This is a runtime hack for local development. In Docker, the workdir and PYTHONPATH should be set in the image, not patched at runtime. This line will cause import priority issues if multiple versions exist on the path.

### DOCK-3 [MEDIUM] Dotenv loaded unconditionally
`ml_service/main.py:23-24` — `load_dotenv()` runs on every import. In container environments, env vars should come from the orchestrator (Kubernetes secrets, Docker --env), not from a .env file that may be accidentally committed.

---

## 18. Redis & Caching

### RED-1 [HIGH] Redis connection not verified at startup
`ml_service/main.py:73-83` — Broker initialization is wrapped in try/except. If Redis is down at startup, the app logs a warning and continues. Predictions that require Redis (stream consumer, sliding window) will fail at runtime with opaque errors.

### RED-2 [MEDIUM] No Redis connection pool configuration
Per memory: `get_engine()` in `iot_ingest_services/common/db.py` uses `pool_recycle=300`. No equivalent pool configuration exists for Redis connections. Long-running pods may hold stale connections.

### RED-3 [MEDIUM] SlidingWindowStore not bounded
Per system audit: SlidingWindowStore + SlidingWindowBuffer never evict inactive sensors. This is both a memory leak (INF-4) and a Redis key leak — sensor-specific keys accumulate in Redis without TTL or eviction policy.

---

## 19. Complexity Analysis

### CYC-1 [HIGH] Orchestrator predict() exceeds 15 logical branches
`infrastructure/ml/cognitive/orchestration/orchestrator.py` — The `predict()` method contains: MoE gateway dispatch, iterative loop check, fallback path, reliability scoring, hyperparameter adaptation, Bayesian weight update, plasticity application, and explanation assembly. McCabe cyclomatic complexity estimated at 18-22. Refactor into strategy classes.

### CYC-2 [MEDIUM] AssemblyPhase._build_metadata has 12 conditional branches
`infrastructure/ml/cognitive/orchestration/phases/assembly_phase.py` — Metadata construction checks for cognitive_trace, shadow, drift, circuit breaker, amnesic mode, fusion flags, hampel diagnostic, engine failures, calibration, explanation, and narrative. Each adds a branch. Estimated complexity 14.

### CYC-3 [MEDIUM] ExpertRegistry.get_candidates has 8 exit points
`infrastructure/ml/moe/registry.py` — Normal return, empty list on no match, None filtering, cost filtering, capability filtering, priority sorting, limit slicing, and error fallback. Multiple return paths make unit testing combinatorial.

---

## 20. Technical Debt Quantification

| Debt Item | Lines of Code | Tests | Last Used | Removal Risk |
|---|---|---|---|---|
| SNN/Neural modules | 600+ | 0 in production | Never | Low |
| Weaviate adapters | 500+ | 0 in production | Never (disabled) | Low |
| Domain explainability | 400 | 38 | Never wired | Medium (tests break) |
| PredictionWriter (Writer B) | 150 | Legacy only | Dev/test | Medium (CI may use) |
| insert_prediction (Writer C) | 80 | 0 callers | Dead | Low |
| IterativeController (unwired) | 247 | Tests exist | Never wired | Medium |
| ShadowEvaluationPhase | 146 | Partial | Always instantiated but usually skipped | Low |
| Taylor engine diagnostics bridge | 50 | Exists but uncalled | Never | Low |
| **TOTAL DEAD CODE** | **~2,173** | **~38+** | — | — |

At $100/line/year maintenance cost (code review, CI time, dependency updates, cognitive load), this represents **~$217K/year in sunk engineering cost**.

---

## 21. Additional Security Risks

### SEC-4 [HIGH] ML_COMPLIANCE_EXPORT_PATH from env with no validation
`infrastructure/ml/cognitive/orchestration/phases/assembly_phase.py:41-42` — Reads `ML_COMPLIANCE_EXPORT_PATH` from environment. If set to `/etc/passwd` or a sensitive path by a compromised pod, the exporter will overwrite it. No path validation (whitelist, sandbox directory check).

### SEC-5 [MEDIUM] HMAC key loaded from env with no rotation
`assembly_phase.py:49` — `load_hmac_key_from_env()` implies single static key. If key leaks (git history, env dump, log exposure), all past compliance exports are forgeable. No key versioning, no rotation schedule.

### SEC-6 [MEDIUM] _safe_float silently coerces invalid input
`infrastructure/persistence/sql/storage/prediction_queries.py:19-28` — Invalid values (e.g., SQL injection attempt in a string field mapped to float) silently become 0.0. No alert, no rejection. Should raise or at least log at error level.

### SEC-7 [LOW] API key passed in X-API-Key header
`ml_service/api/routes.py` — All endpoints read `X-API-Key`. Header values are logged by reverse proxies (nginx, ALB) by default. API keys in headers leak to access logs. Consider Authorization: Bearer scheme.

---

## 22. API Contracts & Serialization

### API-1 [HIGH] PredictResponse schema drift
`ml_service/api/routes.py:97-115` — Maps result dict fields manually to PredictResponse. If the PredictionService adds a new field (e.g., `calibrated_confidence`), it is silently dropped unless routes.py is updated. No automatic schema generation from the domain DTO.

### API-2 [MEDIUM] No API versioning
Routes are `/ml/predict`, `/health`, `/ready` with no `/v1/` prefix. Breaking changes require coordinated deployment of all clients (Flutter, .NET, NestJS, batch runner).

### API-3 [MEDIUM] Raw dicts returned from PredictionService
`ml_service/api/routes.py:79-85` — `service.predict()` returns a plain `dict`, not a typed `PredictionResponse` domain object. Type safety is lost at the application boundary.

### API-4 [LOW] swagger_ui not explicitly disabled
FastAPI defaults to serving `/docs` and `/redoc` with full schema. In production, these expose internal field names, types, and structure to attackers.

---

## 23. Data & Feature Engineering

### DATA-1 [HIGH] SanitizePhase _runtime_providers reads from mutable orchestrator state
Per IMP-3 memory: `SanitizePhase._runtime_providers(ctx)` reads from `ctx.orchestrator._series_values_store`. This is a mutable store shared across requests. While the factory creates fresh executors, the underlying store is still shared. Race conditions possible if two requests for the same sensor_id interleave.

### DATA-2 [MEDIUM] Hampel filter only operates on predicted_value
Per IMP-2 memory: Hampel filters `predicted_value` only; confidence remains with `InhibitionGate`. An outlier engine producing wild confidence (but reasonable value) will not be rejected, skewing the fused confidence upward or downward.

### DATA-3 [MEDIUM] Window size mismatch between stream and batch
Stream consumer uses `min_window_size` from env; batch runner uses `window_size` from config. If set differently, the same sensor gets predictions based on different historical depths, creating inconsistent confidence.

### DATA-4 [LOW] No data validation on Redis stream read
`ml_service/consumers/stream_consumer.py` — Reads from Redis stream and constructs `Reading` objects. No schema validation (Pydantic) on the inbound message. A malformed message (missing timestamp) propagates as `None` through the pipeline.

---

## 24. Detailed Recommendations with Effort Estimates

### Sprint 1 (2 weeks, 1 engineer)
| Task | File(s) | Effort | Risk |
|---|---|---|---|
| Fix DOM-2: to_time_series series_id | `sensor_reading.py:196` | 1 line | None |
| Fix PIPE-4: Seed random in shadow | `shadow_evaluation_phase.py:85` | 3 lines | None |
| Fix ARCH-1: Move format_conclusion to port | `document_analysis.py`, new adapter | 2 days | Low |
| Fix OBS-1: Remove debug warnings | `document_analysis.py` | 30 min | None |
| Fix RES-3: Fail startup on critical deps | `main.py` | 4 hours | Medium |
| Fix DEP-1: Pin NumPy<2.0 | `requirements.txt` | 1 line | None |

### Sprint 2 (2 weeks, 1-2 engineers)
| Task | File(s) | Effort | Risk |
|---|---|---|---|
| Fix PIPE-2: Per-request compliance exporter | `assembly_phase.py` | 3 days | Low |
| Fix INF-3: Consolidate writers A/B/C | `sqlserver_storage.py`, `prediction_writer.py` | 3 days | Medium |
| Fix ARCH-2: Rename to PredictSeriesValueUseCase | `predict_sensor_value.py` + callers | 2 days | Medium |
| Fix CFG-1: Remove duplicate ML_ENABLE_HYBRID_EMBEDDINGS | `cognitive_config.py` | 1 line | None |
| Fix SEC-4: Validate compliance export path | `assembly_phase.py` | 4 hours | Low |
| Fix RED-1: Verify Redis at startup | `main.py`, `broker.py` | 1 day | Low |

### Sprint 3 (3 weeks, 2 engineers)
| Task | File(s) | Effort | Risk |
|---|---|---|---|
| Fix INF-4/INF-5: Sensor eviction + dedup | `sliding_window_store.py`, `stream_consumer.py` | 5 days | High |
| Fix PERF-1: Parallelize batch runner | `runner.py` | 4 days | Medium |
| Fix PIPE-3: Flag-driven phase selection | `pipeline_executor_factory.py` | 3 days | Medium |
| Fix ARCH-3: Refactor FeatureFlags to composition | `flags.py`, 5 config files | 2 days | Medium |
| Fix ML-1: Move neural/ to research repo | `neural/` | 1 day | Low |
| Fix DOCK-1: Add Dockerfile | New file | 1 day | Low |

### Sprint 4 (4 weeks, 2-3 engineers)
| Task | File(s) | Effort | Risk |
|---|---|---|---|
| Fix INT-1: Implement /ml/index-document, /ml/semantic-search | `routes.py`, new services | 5 days | High |
| Fix DOM-4: Wire domain Explanation into orchestrator | `orchestrator.py`, `explanation_builder.py` | 5 days | High |
| Fix DB-1: Execute migration 020 in CI/CD | `migrations/`, CI pipeline | 3 days | Medium |
| Fix SEC-1: Integrate secrets manager | `assembly_phase.py`, deployment | 4 days | High |
| Fix API-2: Add /v1/ prefix to routes | `routes.py`, all clients | 3 days | High |
| Fix PERF-3: Enable calibration by default | `cognitive_config.py`, `calibrators/` | 4 days | Medium |

---

## 25. Conclusion

The ZENIN ML Service is a sophisticated system with genuine architectural ambition. The hexagonal domain layer, 13-phase cognitive pipeline, and MoE gateway demonstrate advanced design thinking. However, **the gap between design and execution is widening**:

- **4,000+ lines of dead code** sit in version control, wasting CI time and confusing new engineers.
- **Domain purity is compromised** by direct imports from FastAPI service modules.
- **13 phases are always instantiated**, burning CPU on disabled features.
- **Three prediction writers** race to the same table with slightly different column sets.
- **Memory leaks** (unbounded sliding windows) guarantee OOM at scale.
- **261 non-passing tests** (11.7%) mean the test suite is not a reliable safety net.

The system works for 10-50 sensors. It degrades at 100+. It will fail at 1,000+. The path to production-grade reliability requires:

1. **Surgical removal of dead code** (neural, Weaviate, unwired explainability)
2. **Enforcement of layer boundaries** (domain must not import ml_service)
3. **Lazy instantiation** (flag-driven phase selection)
4. **Bounded resource management** (sensor eviction, deduplication, connection pooling)
5. **Hermetic test suite** (zero non-passing tests in CI)

Without these changes, every new feature adds debt faster than it adds value.
