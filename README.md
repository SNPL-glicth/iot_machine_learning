# iot_machine_learning

Motor de Machine Learning cognitivo y agnóstico para el sistema IoT **Sandevistan**.

**Tests:** 1096 passed, 6 skipped | **Arquitectura:** Hexagonal + UTSAE | **Identidad:** `series_id: str` (agnóstico)

---

## Arquitectura

El proyecto sigue **arquitectura hexagonal** con principios **UTSAE** (Universal Time Series Analysis Engine). Cada capa tiene una responsabilidad clara y las dependencias apuntan hacia adentro. El sistema es **agnóstico al dominio**: opera sobre cualquier serie temporal (IoT, finanzas, red, salud) usando `series_id: str` como identidad universal.

```
iot_machine_learning/
├── domain/                              # Capa de dominio (pura, sin I/O)
│   ├── entities/
│   │   ├── results/                     # Value objects de resultados
│   │   │   ├── prediction.py            # Prediction, PredictionConfidence (frozen)
│   │   │   ├── anomaly.py               # AnomalyResult, AnomalySeverity (frozen)
│   │   │   └── memory_search_result.py  # MemorySearchResult
│   │   ├── series/                      # Entidades agnósticas de serie
│   │   │   ├── structural_analysis.py   # StructuralAnalysis, RegimeType (frozen)
│   │   │   ├── series_profile.py        # SeriesProfile, VolatilityLevel
│   │   │   └── series_context.py        # SeriesContext, Threshold
│   │   ├── explainability/              # Value objects de explicabilidad
│   │   │   ├── explanation.py           # Explanation (root aggregate), Outcome
│   │   │   ├── reasoning_trace.py       # ReasoningTrace, ReasoningPhase, PhaseKind
│   │   │   ├── contribution_breakdown.py # ContributionBreakdown, EngineContribution
│   │   │   └── signal_snapshot.py       # SignalSnapshot, FilterSnapshot
│   │   ├── patterns/                    # Entidades de patrones
│   │   │   ├── pattern_result.py        # PatternResult
│   │   │   ├── change_point.py          # ChangePoint
│   │   │   ├── delta_spike.py           # DeltaSpike
│   │   │   └── operational_regime.py    # OperationalRegime
│   │   └── iot/                         # Entidades legacy IoT
│   │       ├── sensor_reading.py        # SensorReading, SensorWindow
│   │       └── sensor_ranges.py         # DEFAULT_SENSOR_RANGES (deprecated)
│   ├── ports/                           # Interfaces abstractas (contratos)
│   │   ├── prediction_port.py           # PredictionPort (dual: SensorWindow + TimeSeries)
│   │   ├── anomaly_detection_port.py    # AnomalyDetectionPort (dual)
│   │   ├── pattern_detection_port.py    # PatternDetectionPort (dual)
│   │   ├── storage_port.py              # StoragePort (dual: sensor_id + series_id)
│   │   └── audit_port.py               # AuditPort (dual: sensor_id + series_id)
│   ├── services/                        # Servicios de dominio (lógica pura)
│   │   ├── prediction_domain_service.py # Orquesta predicción + auditoría
│   │   ├── anomaly_domain_service.py    # Voting + consenso + auditoría
│   │   ├── pattern_domain_service.py    # Patrones + enriquecimiento structural
│   │   ├── severity_rules.py            # classify_severity_agnostic()
│   │   └── memory_recall_enricher.py    # Enriquecimiento con memoria cognitiva
│   └── validators/                      # Validaciones reutilizables
│       ├── numeric.py                   # validate_window, clamp_prediction, safe_float
│       ├── temporal.py                  # validate_timestamps, TemporalDiagnostic
│       ├── input_guard.py              # guard_series_id, safe_series_id_to_int
│       └── structural_analysis.py       # compute_structural_analysis()
│
├── application/                         # Capa de aplicación (casos de uso)
│   ├── use_cases/
│   │   ├── predict_sensor_value.py      # PredictSensorValueUseCase + memory recall
│   │   ├── detect_anomalies.py          # DetectAnomaliesUseCase
│   │   ├── analyze_patterns.py          # AnalyzePatternsUseCase
│   │   └── select_engine.py             # select_engine_for_series() (agnóstico)
│   ├── dto/
│   │   └── prediction_dto.py            # PredictionDTO, AnomalyDTO, PatternDTO (series_id: str)
│   └── explainability/
│       └── explanation_renderer.py      # ExplanationRenderer (summary, report, JSON)
│
├── infrastructure/                      # Capa de infraestructura (implementaciones)
│   ├── ml/
│   │   ├── interfaces.py               # PredictionEngine, PredictionResult, SignalFilter
│   │   │                                # + PredictionEnginePortBridge (engine.as_port())
│   │   ├── engines/
│   │   │   ├── engine_factory.py        # EngineFactory + @register_engine + discover_engines
│   │   │   ├── baseline_engine.py       # BaselineMovingAverageEngine (fallback)
│   │   │   ├── taylor_engine.py         # TaylorPredictionEngine (orden 1-3)
│   │   │   ├── statistical_engine.py    # StatisticalPredictionEngine (Holt-Winters)
│   │   │   ├── taylor/                  # Paquete modular Taylor
│   │   │   │   ├── types.py             # TaylorCoefficients, TaylorDiagnostic, DerivativeMethod
│   │   │   │   ├── derivatives.py       # backward, central, least_squares
│   │   │   │   ├── polynomial.py        # project(), compute_local_fit_error()
│   │   │   │   ├── diagnostics.py       # compute_diagnostic()
│   │   │   │   ├── time_step.py         # compute_dt()
│   │   │   │   └── least_squares.py     # Gaussian elimination solver
│   │   │   ├── taylor_adapter.py        # Adapter (deprecated → use engine.as_port())
│   │   │   └── baseline_adapter.py      # Adapter (deprecated → use engine.as_port())
│   │   ├── cognitive/                   # Orquestador meta-cognitivo
│   │   │   ├── orchestrator.py          # MetaCognitiveOrchestrator (Perceive→Fuse→Explain)
│   │   │   ├── signal_analyzer.py       # SignalAnalyzer → StructuralAnalysis
│   │   │   ├── explanation_builder.py   # ExplanationBuilder → Explanation
│   │   │   ├── engine_selector.py       # WeightedFusion
│   │   │   ├── inhibition.py            # InhibitionGate (supresión de engines inestables)
│   │   │   ├── plasticity.py            # PlasticityTracker (pesos adaptativos)
│   │   │   └── types.py                 # EnginePerception, InhibitionState, MetaDiagnostic
│   │   ├── filters/
│   │   │   ├── kalman_filter.py         # KalmanSignalFilter (adaptive Q opcional)
│   │   │   ├── kalman_math.py           # KalmanState, calibración, update
│   │   │   ├── ema_filter.py            # EMASignalFilter, AdaptiveEMASignalFilter
│   │   │   ├── median_filter.py         # MedianSignalFilter (spike removal)
│   │   │   ├── filter_chain.py          # FilterChain (composable pipeline)
│   │   │   └── filter_diagnostic.py     # FilterDiagnostic, compute_filter_diagnostic()
│   │   ├── anomaly/
│   │   │   ├── voting_anomaly_detector.py  # VotingAnomalyDetector (8 votos, DI)
│   │   │   ├── statistical_methods.py      # Z-score, IQR, temporal stats
│   │   │   ├── anomaly_narrator.py         # Narrativa de anomalías
│   │   │   ├── detector_protocol.py        # SubDetector protocol + DetectorRegistry
│   │   │   └── detectors/                  # Sub-detectores individuales
│   │   ├── patterns/
│   │   │   ├── delta_spike_classifier.py   # DeltaSpikeClassifier
│   │   │   ├── change_point_detector.py    # CUSUM + PELT
│   │   │   └── regime_detector.py          # RegimeDetector (KMeans)
│   │   └── explainability/
│   │       ├── taylor_importance.py        # Feature importance via Taylor
│   │       └── counterfactual.py           # Explicaciones contrafactuales
│   ├── adapters/
│   │   ├── sqlserver_storage.py            # StoragePort → SQL Server
│   │   ├── cognitive_storage_decorator.py  # Decorator con memoria cognitiva
│   │   ├── prediction_cache.py             # LRU + TTL cache
│   │   └── batch_predictor.py              # ThreadPool + CircuitBreaker
│   └── security/
│       ├── file_audit_logger.py            # AuditPort → archivos (ISO 27001)
│       ├── null_audit_logger.py            # AuditPort no-op
│       └── access_control.py              # RBAC (series_id + sensor_id dual)
│
├── ml_service/                          # Servicio HTTP + runners (orquestación IoT)
│   ├── main.py                          # FastAPI app
│   ├── config/
│   │   ├── ml_config.py                 # GlobalMLConfig, AnomalyConfig, EngineConfig
│   │   └── feature_flags.py             # FeatureFlags (dual: series_id + sensor_id)
│   ├── api/services/
│   │   └── prediction_service.py        # Servicio HTTP de predicción
│   ├── runners/                         # Pipeline batch + stream
│   ├── explain/                         # AI Explainer + TemplateExplanationGenerator
│   ├── metrics/                         # MetricsCollector, A/B testing
│   ├── memory/                          # Weaviate cognitive memory
│   └── correlation/                     # Correlación entre sensores
│
└── tests/                               # 1096 tests
    ├── unit/
    │   ├── domain/                      # Entidades, servicios, validadores
    │   ├── infrastructure/              # Motores, filtros, cognitivo, DI
    │   ├── application/                 # Use cases, renderer
    │   └── ml_service/                  # Servicio HTTP, runners
    └── integration/                     # A/B, enterprise flow, COG-4
```

---

## Motores de predicción

| Motor | Ubicación | Descripción |
|-------|-----------|-------------|
| **Baseline** | `engines/baseline_engine.py` | Media móvil simple. Fallback por defecto. |
| **Taylor** | `engines/taylor_engine.py` + `taylor/` | Series de Taylor con 3 métodos de derivadas (backward, central, least_squares). Orden 1-3. |
| **Statistical** | `engines/statistical_engine.py` | Holt-Winters exponential smoothing (α, β). |
| **Meta-Cognitive** | `cognitive/orchestrator.py` | Orquesta múltiples engines con inhibición, plasticidad y fusión ponderada. |

### Selección de motor

La selección se controla vía **feature flags** (`select_engine.py`):

1. `ML_ROLLBACK_TO_BASELINE` = true → baseline (panic button)
2. Override agnóstico por `series_id` en `ML_ENGINE_SERIES_OVERRIDES`
3. Override legacy por `sensor_id` en `ML_ENGINE_OVERRIDES`
4. Serie en whitelist de Taylor → taylor
5. `ML_DEFAULT_ENGINE` global
6. Fallback a baseline

La selección agnóstica `select_engine_for_series(profile, flags)` elige motor por **características del dato** (volatilidad, estacionaridad, n_points), no por identidad del sensor.

### Extensibilidad — Agregar un nuevo engine

Agregar un engine requiere **un solo archivo** gracias al sistema de auto-registro:

```python
# infrastructure/ml/engines/my_engine.py
from iot_machine_learning.infrastructure.ml.engines.engine_factory import register_engine
from iot_machine_learning.infrastructure.ml.interfaces import PredictionEngine

@register_engine("my_engine")
class MyEngine(PredictionEngine):
    @property
    def name(self) -> str: return "my_engine"
    def can_handle(self, n_points: int) -> bool: return n_points >= 5
    def predict(self, values, timestamps=None): ...
```

APIs disponibles:
- **`@register_engine("name")`** — Decorator que auto-registra un engine en `EngineFactory`
- **`discover_engines("package.path")`** — Escanea un paquete e importa módulos para trigger decorators
- **`EngineFactory.create_as_port("name")`** — Crea engine + bridge a `PredictionPort` en una línea
- **`engine.as_port()`** — Convierte cualquier `PredictionEngine` a `PredictionPort`

---

## Orquestador Meta-Cognitivo

Pipeline: **Perceive → Predict → Inhibit → Adapt → Fuse → Explain**

```python
orchestrator = MetaCognitiveOrchestrator(
    engines=[taylor, statistical, baseline],
    enable_plasticity=True,
)
result = orchestrator.predict(values, timestamps, series_id="temp_room_1")
explanation = orchestrator.last_explanation  # Explanation (domain value object)
```

Capacidades:
- **Inhibición**: engines inestables (alta stability/fit_error) se suprimen automáticamente
- **Plasticidad**: pesos se adaptan por régimen tras `record_actual(value)`
- **Fusión ponderada**: weighted average con trend por voto mayoritario
- **Explicabilidad**: genera `Explanation` completo con traza de razonamiento

---

## Filtros de señal

| Filtro | Ubicación | Descripción |
|--------|-----------|-------------|
| **Kalman 1D** | `filters/kalman_filter.py` | Auto-calibración de R. Adaptive Q opcional. Thread-safe. |
| **EMA** | `filters/ema_filter.py` | Exponential Moving Average (fijo + adaptativo por innovación). |
| **Median** | `filters/median_filter.py` | Sliding window median. Robusto a spikes. |
| **FilterChain** | `filters/filter_chain.py` | Pipeline composable: `Median → Kalman`, `Median → EMA`, etc. |
| **Identity** | `interfaces.py` | No-op (fallback). |

Todos implementan `SignalFilter(series_id: str)`. `FilterDiagnostic` mide noise_reduction, distortion, lag.

---

## Detección de anomalías

**Voting ensemble temporal** (`anomaly/voting_anomaly_detector.py`) con **8 votos**:

| Voto | Peso | Descripción |
|------|------|-------------|
| **Isolation Forest 1D** | 0.30 | Anomalía por magnitud |
| **Z-score** | 0.20 | Desviación estadística |
| **Velocity Z-score** | 0.15 | Tasa de cambio anormal (dv/dt) |
| **LOF 1D** | 0.15 | Local Outlier Factor por magnitud |
| **IQR** | 0.10 | Rango intercuartílico |
| **Acceleration Z-score** | 0.10 | Aceleración anormal (d²v/dt²) |
| **IF 3D** | — | Isolation Forest [value, velocity, acceleration] (temporal) |
| **LOF 3D** | — | LOF [value, velocity, acceleration] (temporal) |

Consenso por voto ponderado con umbral configurable. Entrenamiento temporal: `train(values, timestamps=...)`.

### Extensibilidad — Agregar un sub-detector

```python
from iot_machine_learning.infrastructure.ml.anomaly.detector_protocol import register_detector

@register_detector("my_detector")
def create_my_detector(config):
    return MySubDetector(config)
```

APIs: `DetectorRegistry.register()`, `DetectorRegistry.create_all(config)`, `@register_detector("name")`

### Inyección de dependencias

```python
detector = VotingAnomalyDetector(config, sub_detectors=[custom1, custom2])  # DI
detector = VotingAnomalyDetector(config)  # Default: 8 sub-detectores
```

---

## Detección de patrones

| Detector | Descripción |
|----------|-------------|
| **DeltaSpikeClassifier** | Clasifica cambios bruscos (spikes, drops) |
| **CUSUMDetector** | Cambios de media acumulativos |
| **PELTDetector** | Change points por penalización |
| **RegimeDetector** | Regímenes operacionales (KMeans) |

`PatternDomainService` enriquece cada `PatternResult` con `StructuralAnalysis` en metadata.

---

## Análisis estructural

`StructuralAnalysis` es el **análisis compartido** que todos los subsistemas consumen:

```python
from iot_machine_learning.domain.validators.structural_analysis import compute_structural_analysis

sa = compute_structural_analysis(values, timestamps)
# sa.slope, sa.curvature, sa.stability, sa.noise_ratio, sa.regime, sa.dt
```

- **Taylor engine** → produce `StructuralAnalysis` vía `from_taylor_diagnostic()` bridge
- **Anomaly detection** → consume velocidad/aceleración para votos temporales
- **Pattern detection** → enriquece resultados con structural metadata
- **Cognitive orchestrator** → consume perfil completo para selección de pesos

Régimen clasificado como: `STABLE`, `TRENDING`, `VOLATILE`, `TRANSITIONAL`, `NOISY`.

---

## Explicabilidad

Tres capas independientes:

```
domain/entities/explainability/     ← Value objects puros (Explanation, ReasoningTrace, etc.)
infrastructure/ml/cognitive/        ← ExplanationBuilder (traduce infra → dominio)
application/explainability/         ← ExplanationRenderer (transforma dominio → humano)
```

### Explanation (domain)

```python
explanation = Explanation(
    series_id="temp_room_1",
    signal=SignalSnapshot(...),          # Perfil de señal
    contributions=ContributionBreakdown(...),  # Contribución por engine
    trace=ReasoningTrace(phases=[...]),  # Fases del razonamiento
    outcome=Outcome(predicted_value=25.0, confidence=0.85, trend="up"),
)
```

### ExplanationRenderer (application)

```python
renderer = ExplanationRenderer()
renderer.render_summary(explanation)           # 1-3 líneas para dashboards
renderer.render_technical_report(explanation)   # Reporte multi-sección
renderer.render_structured_json(explanation)    # JSON + clasificaciones metacognitivas
```

Clasificaciones metacognitivas (solo lectura de propiedades del dominio):
- **Certeza**: high / moderate / low / very_low
- **Desacuerdo**: consensus / mild / significant / severe
- **Estabilidad cognitiva**: stable / adapting / stressed / degraded
- **Riesgo de sobreajuste**: low / moderate / high
- **Conflicto entre engines**: aligned / mild_divergence / directional_conflict

---

## Conversión segura series_id → sensor_id

Toda conversión de `series_id: str` a `sensor_id: int` (para BD legacy) usa:

```python
from iot_machine_learning.domain.validators.input_guard import safe_series_id_to_int

sensor_id = safe_series_id_to_int("42")          # → 42
sensor_id = safe_series_id_to_int("room_temp")   # → 0 (fallback, logged)
sensor_id = safe_series_id_to_int("abc", fallback=-1)  # → -1
```

**Nunca** usar `int(series_id)` directamente. `safe_series_id_to_int` está aplicado en todos los ports y adapters.

---

## Pipelines de ejecución

### Predicción puntual (HTTP)

```
POST /ml/predict → FastAPI → PredictionService → EngineFactory → Motor → dbo.predictions
```

### Pipeline batch

```
ml_batch_runner → sensores activos → SensorProcessor → predicción + anomalía + severidad + narrativa → dbo.predictions + dbo.ml_events
```

### Pipeline online (stream)

```
ml_stream_runner → ReadingBroker → SlidingWindowBuffer → patrones (1s/5s/10s) → dbo.ml_events + dbo.alert_notifications
```

### Pipeline cognitivo

```
MetaCognitiveOrchestrator → Perceive(signal) → Predict(engines) → Inhibit(unstable) → Adapt(plasticity) → Fuse(weights) → Explain(builder) → Explanation
```

---

## Comunicación con otros servicios

| Servicio | Dirección | Detalle |
|----------|-----------|---------|
| **SQL Server** (`iot_database`) | Lee/Escribe | `sensor_readings`, `predictions`, `ml_models`, `ml_events`, `alert_thresholds`, `alert_notifications` |
| **Ingesta** (`iot_ingest_services`) | Lee | `ReadingBroker` (in-memory), conexión BD compartida |
| **AI Explainer** (`ai-explainer`) | HTTP | `/explain/anomaly` — fallback a templates si no disponible |
| **Weaviate** | HTTP | Memoria cognitiva: `recall_similar_explanations`, `recall_similar_anomalies` |
| **Backend** (`iot_monitor_backend`) | Indirecto | Consume `predictions`, `ml_events`, `notifications` vía BD |

---

## Reglas de dominio

- **Umbrales del usuario tienen prioridad**: si el valor está dentro del rango WARNING configurado, no se genera evento ML.
- **Estados operacionales bloquean ML**: sensores en `INITIALIZING` o `STALE` no generan eventos.
- **Fallbacks claros**: si AI Explainer no responde, se usan templates determinísticos.
- **Severidad unificada**: `AnomalySeverity.from_score()` es la fuente única de verdad para clasificación de severidad.
- **Inmutabilidad**: todas las entidades de dominio son `frozen=True`. Modificaciones vía `dataclasses.replace()`.

---

## Tests

```bash
# Ejecutar suite completa (1096 tests)
python -m pytest iot_machine_learning/tests/ -v

# Solo tests unitarios
python -m pytest iot_machine_learning/tests/unit/ -v

# Solo tests de integración
python -m pytest iot_machine_learning/tests/integration/ -v
```

| Capa | Tests | Cobertura |
|------|-------|-----------|
| Domain (entidades, servicios, validadores, structural analysis) | ~200 | Lógica pura, sin mocks |
| Infrastructure (motores, filtros, cognitivo, anomalía, DI) | ~400 | Math puro + adapters + extensibilidad |
| Application (use cases, renderer, select_engine) | ~80 | Mocks de ports |
| ML Service (prediction, narrator, severity, memory recall) | ~100 | Mocks de BD |
| Integration (A/B, enterprise flow, COG-4, wiring) | ~295 | End-to-end sin BD real |

### Progresión de tests

```
413 → 576 → 636 → 769 → 789 → 830 → 845 → 922 → 960 → 1014 → 1035 → 1056 → 1075 → 1096
```

---

## Decisiones técnicas

- **Arquitectura hexagonal**: domain puro (sin I/O) → application (use cases) → infrastructure (implementaciones). Dependencias apuntan hacia adentro.
- **UTSAE agnóstico**: `series_id: str` como identidad universal. `ml_service/` mantiene `sensor_id: int` como bridge IoT.
- **Feature flags**: todo motor nuevo se activa gradualmente vía flags, con rollback instantáneo a baseline.
- **Archivos < 180 líneas**: cada módulo tiene responsabilidad única. Math puro separado de orquestación.
- **Plugin architecture**: `@register_engine` y `@register_detector` permiten agregar engines/detectores sin modificar código existente.
- **Dual interface en ports**: métodos legacy (`SensorWindow`) coexisten con agnósticos (`TimeSeries`).
- **`dataclasses.replace()`**: para modificar entidades frozen sin reconstrucción manual.
- **`safe_series_id_to_int()`**: conversión centralizada y segura para bridges legacy.
- **Explicabilidad como capa**: domain (value objects) → infra (builder) → application (renderer). Sin acoplamiento.
- **MetaDiagnostic deprecated**: usar `last_explanation` (domain `Explanation`) en vez de `last_diagnostic`.
- **Persistencia en BD**: `predictions`, `ml_events`, `alert_notifications` con `engine_name`, `trend`, `method_votes`, `audit_trace_id` para trazabilidad completa.

---

## Qué NO hace este módulo

- No captura lecturas desde hardware (eso es `iot_ingest_services`).
- No gestiona usuarios/auth (eso es `iot_monitor_backend`).
- No define esquemas SQL ni ejecuta migraciones (eso es `iot_database`).
- No garantiza tiempo real end-to-end (broker in-memory, no distribuido).

---

## Hallazgos arquitectónicos resueltos

| ID | Hallazgo | Resolución | Fase |
|----|----------|------------|------|
| ARQ-1 | Dual interface PredictionEngine vs PredictionPort | `PredictionEnginePortBridge` + `engine.as_port()` | Phase 2 |
| ARQ-2 | Reverse import application → infra en EngineFactory | `get_engine_for_sensor()` deprecated, sin import inverso | Phase 2 |
| MOD-2 | VotingAnomalyDetector hardcodea 8 detectores | DI vía `sub_detectors` param + `create_default_detectors()` | Phase 3 |
| ROB-1 | Agregar engine requiere tocar 4+ archivos | `@register_engine` — 1 archivo nuevo | Phase 3 |
| ROB-2 | No se pueden inyectar detectores custom | `DetectorRegistry` + `@register_detector` | Phase 3 |
| DEBT-1 | Bridges `int(series_id)` inseguros en 7 archivos | `safe_series_id_to_int()` centralizado | Phase 4 |
| DEBT-4 | Reconstrucción manual de Prediction (12 campos) | `dataclasses.replace()` | Phase 4 |
| COG-1 | `_classify_regime()` duplicado en 2 archivos | Unificado en `structural_analysis.py` (domain) | Phase 1-Cog |
| COG-2 | `SignalProfile` ≈ `StructuralAnalysis` duplicados | `StructuralAnalysis` como tipo único | Phase 1-Cog |
| COG-3 | `MetaDiagnostic` redundante con `Explanation` | Deprecated con DeprecationWarning | Phase 4 |
| COG-4 | Umbrales de severidad duplicados renderer vs domain | `template_generator` delega a `AnomalySeverity.from_score()` | Phase 4 |
