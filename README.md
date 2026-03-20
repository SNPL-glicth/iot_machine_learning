# iot_machine_learning

Motor de Machine Learning cognitivo y agnóstico para el sistema IoT **Sandevistan**.

**Tests:** ~1260 passed, 39 skipped | **Arquitectura:** Hexagonal + UTSAE | **Identidad:** `series_id: str` (agnóstico)

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
│   │   │   ├── types.py                 # EnginePerception, InhibitionState, MetaDiagnostic
│   │   │   └── text/                    # TextCognitiveEngine (análisis profundo de texto)
│   │   │       ├── engine.py            # TextCognitiveEngine (orquestador principal)
│   │   │       ├── types.py             # TextAnalysisContext, TextAnalysisInput, TextCognitiveResult
│   │   │       ├── signal_profiler.py   # TextSignalProfiler → SignalSnapshot
│   │   │       ├── perception_collector.py # Sub-analyzers → EnginePerception[]
│   │   │       ├── severity_mapper.py   # classify_text_severity() → SeverityResult
│   │   │       ├── memory_enricher.py   # TextMemoryEnricher → TextRecallContext
│   │   │       └── explanation_assembler.py # Builds Explanation domain object
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
├── ml_service/                          # Servicio HTTP + runners + poller Zenin
│   ├── main.py                          # FastAPI app (puerto 8002) + Zenin poller daemon
│   ├── config/
│   │   ├── ml_config.py                 # GlobalMLConfig, AnomalyConfig, EngineConfig
│   │   └── feature_flags.py             # FeatureFlags (dual: series_id + sensor_id)
│   ├── api/
│   │   └── services/
│   │       └── prediction_service.py    # PredictionService (HTTP)
│   ├── runners/                         # Pipeline batch + stream (ver runners/README.md)
│   │   ├── ml_batch_runner.py           # Orquestador batch (sensor_id loop)
│   │   ├── ml_stream_runner.py          # Consumer stream (ReadingBroker → SlidingWindow)
│   │   ├── adapters/                    # Adapters de entrada (SensorWindow → TimeSeries)
│   │   ├── bridge_config/               # BatchEnterpriseContainer, FeatureFlags bridge
│   │   ├── common/                      # SensorProcessor, utilidades compartidas
│   │   ├── models/                      # RunnerConfig, SensorBatchResult
│   │   ├── monitoring/                  # MetricsCollector, HealthChecker
│   │   ├── services/                    # SensorProcessingService
│   │   └── wiring/                      # DI: crea engines, detectores, orquestador
│   ├── metrics/                         # Métricas y A/B testing (ver metrics/README.md)
│   │   ├── ab_testing.py                # ABTester (thread-safe, por sensor)
│   │   └── ab_metrics.py                # ABTestResult, winner, improvement
│   ├── broker/                          # ReadingBroker (in-memory pub/sub)
│   ├── consumers/                       # StreamConsumer (suscriptor del broker)
│   ├── workers/                         # Workers asíncronos
│   │   └── zenin_queue_poller.py        # Daemon: lee ingestion_queue → DocumentAnalyzer → analysis_results
│   ├── explain/                         # AI Explainer + TemplateExplanationGenerator
│   │   ├── models/                      # ExplanationRequest, ExplanationResponse
│   │   └── services/                    # AIExplainerClient, TemplateGenerator
│   ├── memory/                          # Weaviate cognitive memory
│   │   ├── models/                      # MemoryEntry, RecallResult
│   │   └── services/                    # WeaviateMemoryService
│   ├── context/                         # Contexto de series (perfil, umbrales)
│   │   ├── models/
│   │   └── services/
│   ├── features/                        # Feature engineering para ML
│   │   ├── models/
│   │   ├── persistence/
│   │   └── services/
│   ├── orchestrator/                    # Orquestador de alto nivel
│   │   ├── models/
│   │   └── services/
│   ├── correlation/                     # Correlación entre sensores
│   ├── trainers/                        # Entrenamiento de modelos
│   ├── repository/                      # Repositorio ML (predicciones, modelos)
│   ├── logging/                         # Logging estructurado ML
│   ├── models/                          # Modelos Pydantic del servicio
│   └── utils/                           # Utilidades: numeric_precision, safe_float
│
├── ml_api/                              # Facade HTTP (create_app → FastAPI + /health)
├── ml_batch/                            # Facade batch (run_batch_cycle)
├── ml_stream/                           # Facade stream (start_consumer)
│
├── infrastructure/persistence/sql/
│   └── zenin_db_connection.py           # SQLAlchemy connection separada a zenin_db
│
└── tests/                               # ~1207 tests
    ├── unit/
    │   ├── domain/                      # Entidades, servicios, validadores
    │   ├── infrastructure/              # Motores, filtros, cognitivo, DI
    │   ├── application/                 # Use cases, renderer
    │   └── ml_service/                  # Servicio HTTP, runners, métricas
    └── integration/                     # A/B, enterprise flow, cognitivo
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

## TextCognitiveEngine — Análisis profundo de texto

Motor cognitivo reutilizable para análisis de texto, al mismo nivel que `MetaCognitiveOrchestrator`. Vive en `infrastructure/ml/cognitive/text/` — **no** en `ml_service/` — porque es un engine de infraestructura, no lógica de servicio HTTP.

Pipeline: **Perceive → Analyze → Remember → Reason → Explain**

```python
from iot_machine_learning.infrastructure.ml.cognitive.text import (
    TextCognitiveEngine, TextAnalysisContext, TextAnalysisInput,
)

engine = TextCognitiveEngine()
result = engine.analyze(inp, ctx)

result.explanation   # Explanation (domain value object — mismo tipo que MetaCognitiveOrchestrator)
result.severity      # SeverityResult (critical / warning / info)
result.confidence    # float [0, 1]
result.domain        # Auto-detectado: infrastructure, security, operations, business, general
result.analysis      # Dict backward-compatible con formato legacy
result.to_dict()     # Serialización completa
result.to_legacy_dict()  # Formato legacy para callers existentes
```

### Arquitectura por capas

```
ml_service/api/services/analyzers/           ← Capa de servicio: ejecuta sub-analyzers
    text_sentiment.py, text_urgency.py, ...      (produce scores primitivos)
    text_analyzer.py                             (llama engine, enriquece conclusión)
            ↓ TextAnalysisInput (primitivos)
infrastructure/ml/cognitive/text/            ← Capa de infraestructura: razonamiento cognitivo
    engine.py                                    (orquesta 5 fases)
    perception_collector.py                      (scores → EnginePerception[])
    signal_profiler.py                           (métricas → SignalSnapshot)
    severity_mapper.py                           (urgency+sentiment → SeverityResult)
    memory_enricher.py                           (CognitiveMemoryPort → TextRecallContext)
    explanation_assembler.py                     (→ Explanation domain object)
```

Dependencias apuntan hacia adentro: `ml_service → infrastructure → domain`. El engine **no importa nada de ml_service** — recibe scores pre-computados via `TextAnalysisInput` (primitivos puros).

### Subcomponentes

| Componente | Archivo | Responsabilidad |
|---|---|---|
| **TextSignalProfiler** | `signal_profiler.py` | Mapea métricas de texto → `SignalSnapshot` (word_count→n_points, sentiment→slope, urgency→curvature) |
| **TextPerceptionCollector** | `perception_collector.py` | Convierte 5 sub-analyzers a `EnginePerception[]` (reutiliza `InhibitionGate`, `WeightedFusion`) |
| **classify_text_severity** | `severity_mapper.py` | Urgency + sentiment → `SeverityResult` (critical/warning/info) |
| **TextMemoryEnricher** | `memory_enricher.py` | Recall semántico de documentos similares via `CognitiveMemoryPort`. Graceful-fail si Weaviate cae. |
| **TextExplanationAssembler** | `explanation_assembler.py` | Construye `Explanation` domain object con contribuciones, traza, outcome |

### Degradación graciosa

- **Sin Weaviate**: memory recall se desactiva, pipeline continúa sin enrichment
- **Sin PlasticityTracker**: usa pesos por defecto (`text_urgency=0.30, text_sentiment=0.20, text_pattern=0.20, text_readability=0.15, text_structural=0.15`)
- **Sub-analyzer no disponible**: retorna `EnginePerception` con `confidence=0.3`, `predicted_value=0.5`

---

## HybridNeuralEngine — Red neuronal biológicamente realista

Motor cognitivo híbrido que combina **Spiking Neural Network (SNN)** + **red feedforward clásica** con plasticidad avanzada. Implementa neuronas Leaky-Integrate-Fire con parámetros de neurona piramidal cortical, STDP (Spike-Timing Dependent Plasticity), y mecanismos de metaplasticidad.

**Ubicación:** `infrastructure/ml/cognitive/neural/`

### Arquitectura modular

Pipeline de 5 etapas independientes y testeables:

```python
from iot_machine_learning.infrastructure.ml.cognitive.neural import HybridNeuralEngine

engine = HybridNeuralEngine(
    n_input=10,              # Número de analizadores de entrada
    n_hidden_snn=20,         # Neuronas ocultas en SNN
    n_hidden_classical=16,   # Neuronas ocultas en red clásica
    n_output=5,              # Clases de severidad (info → critical)
    snn_weight=0.5,          # Peso de SNN en fusión híbrida
    enable_online_learning=True,  # Aprendizaje por documento
)

result = engine.analyze(
    analysis_scores={"text_urgency": 0.8, "text_sentiment": 0.3, ...},
    input_type=InputType.TEXT,
    domain="security",
)

# result.severity         → "critical" / "high" / "medium" / "low" / "info"
# result.confidence       → [0, 1]
# result.energy_consumed  → Julios (1 picojoule/spike)
# result.active_neurons   → Neuronas que dispararon
# result.spike_patterns   → Patrones de disparo por neurona
```

### Pipeline de etapas

```
neural/pipeline/
├── encoder_stage.py      # Scores → spike trains (rate + temporal coding)
├── snn_stage.py          # Spike trains → red de neuronas LIF
├── classical_stage.py    # Scores → feedforward clásica
├── fusion_stage.py       # SNN + clásica → salida híbrida
└── decoder_stage.py      # Salida híbrida → severidad + confianza
```

Cada etapa es **stateless** y testeable de forma independiente. El orquestador (`HybridNeuralEngine`) delega a las etapas en secuencia.

### Componentes SNN biológicamente realistas

#### 1. Leaky-Integrate-Fire Neuron (LIF)

```python
from iot_machine_learning.infrastructure.ml.cognitive.neural.snn import (
    LeakyIntegrateFireNeuron, NeuronParameters
)

# Parámetros de neurona piramidal cortical
params = NeuronParameters(
    V_rest=-70.0,           # mV — potencial de reposo
    V_threshold=-55.0,      # mV — umbral de disparo
    V_reset=-80.0,          # mV — hiperpolarización post-spike
    tau_m=20.0,             # ms — constante de tiempo de membrana
    tau_ref=2.0,            # ms — período refractario absoluto
    tau_rel=5.0,            # ms — período refractario relativo
    C_m=281.0,              # pF — capacitancia de membrana
    R_m=70.0,               # MΩ — resistencia de membrana
    noise_std=0.5,          # mV — ruido de membrana
    adaptation_tau=144.0,   # ms — adaptación de frecuencia de disparo
    adaptation_increment=4.0,  # pA — incremento de corriente adaptativa
)

neuron = LeakyIntegrateFireNeuron("neuron_1", params)
spike_event = neuron.integrate(I_syn=50.0, dt=0.1, current_time=10.0)
```

**Dinámica de membrana:**
```
C_m * dV/dt = -(V - V_rest)/R_m + I_syn(t) + I_adapt(t) + I_noise(t)

Equivalente: dV/dt = -(V - V_rest)/τ_m + I_syn/C_m - I_adapt/C_m + noise
```

**Características biológicas:**
- Período refractario absoluto (2ms): neurona no puede disparar
- Período refractario relativo (5ms): umbral elevado exponencialmente
- Adaptación de frecuencia: previene disparo continuo (runaway firing)
- Ruido de membrana: dinámica estocástica (σ = 0.5mV)
- Energía: 1 picojoule por spike

#### 2. Synaptic Kernels — Corrientes sinápticas

```python
from iot_machine_learning.infrastructure.ml.cognitive.neural.snn import SynapticKernel

kernel = SynapticKernel(
    tau_syn_exc=5.0,   # ms — decaimiento excitatorio
    tau_syn_inh=10.0,  # ms — decaimiento inhibitorio
)

I_syn = kernel.compute_current(
    spike_times=[5.0, 10.0, 15.0],
    weights=[0.8, 0.6, 0.9],
    current_time=20.0,
    synapse_type="excitatory",
)

# I_syn(t) = Σ w_i * I_0 * exp(-(t - t_spike)/τ_syn)
```

Las corrientes sinápticas decaen exponencialmente después del spike (no son instantáneas). Sinapsis inhibitorias tienen decaimiento más lento (10ms vs 5ms).

#### 3. Spike Encoding — Rate + Temporal Coding

```python
from iot_machine_learning.infrastructure.ml.cognitive.neural.snn import SpikeEncoder

encoder = SpikeEncoder(
    max_rate=100.0,           # Hz — límite biológico
    min_rate=10.0,            # Hz
    use_temporal_coding=True,  # Latencia del primer spike
)

spike_trains = encoder.encode(
    analysis_scores={"urgency": 0.8, "sentiment": 0.3},
    input_type=InputType.TEXT,
    duration_ms=100.0,
)

# spike_trains = {
#   "urgency": [5.2, 15.8, 24.3, ...],     # Primer spike temprano (alta urgencia)
#   "sentiment": [45.1, 62.4, 78.9, ...],  # Primer spike tardío (baja sentiment)
# }
```

**Estrategias de codificación:**
- **Rate coding**: Tren de spikes Poisson (biológicamente realista)
  - `rate = score * max_rate` (Hz)
  - Intervalos inter-spike: distribución exponencial
- **Temporal coding**: Latencia del primer spike
  - Score alto → primer spike temprano
  - Score bajo → primer spike tardío
- **Hybrid coding**: Combina ambos para representación más rica

#### 4. STDP Learning — Plasticidad hebbiana

```python
from iot_machine_learning.infrastructure.ml.cognitive.neural.snn import STDPLearning

stdp = STDPLearning(
    A_plus=0.01,      # Amplitud de potenciación
    A_minus=0.012,    # Amplitud de depresión
    tau_plus=20.0,    # ms — constante de tiempo potenciación
    tau_minus=20.0,   # ms — constante de tiempo depresión
)

# Regla STDP:
# Si pre antes que post → ΔW = A+ * exp(-Δt/τ+)  (potenciación)
# Si post antes que pre → ΔW = -A- * exp(Δt/τ-)  (depresión)

new_weight = stdp.update_weight(
    weight=0.5,
    pre_spike_times=[10.0, 25.0],
    post_spike_times=[12.0, 28.0],
)
```

**"Neurons that fire together wire together"** — aprendizaje hebbiano para redes de spikes. Se aplica automáticamente después de cada `forward()` si `enable_stdp=True`.

### Plasticidad avanzada

#### 1. Metaplasticidad — BCM Theory

```python
from iot_machine_learning.infrastructure.ml.cognitive.neural.plasticity import (
    MetaplasticityController
)

meta = MetaplasticityController(
    tau_BCM=1000.0,        # ms — constante de tiempo para umbral deslizante
    initial_threshold=0.5,
)

# Actualizar umbral de modificación
theta = meta.update_threshold(
    domain="security",
    current_activity=0.7,
    dt=1.0,
)

# BCM: θ_M(t) = θ_M(t-1) + (activity² - θ_M) / τ_BCM
# El umbral se desliza hacia activity² para prevenir potenciación descontrolada
```

**Metaplasticidad = plasticidad de la plasticidad**. El umbral de modificación sináptica se ajusta según el historial de actividad, implementando la teoría BCM (Bienenstock-Cooper-Munro).

#### 2. Neuromodulación — Señal tipo dopamina

```python
from iot_machine_learning.infrastructure.ml.cognitive.neural.plasticity import (
    NeuromodulationSignal
)

neuromod = NeuromodulationSignal(
    baseline_modulation=1.0,
    surprise_amplification=3.0,
    consolidation_damping=0.1,
)

modulation = neuromod.compute_modulation(
    predicted_severity="low",
    actual_severity="critical",
    confidence=0.9,
)

# Correcto + alta confianza → 0.1× learning rate (consolidar)
# Incorrecto + alta confianza → 3.0× learning rate (sorpresa!)
# Incorrecto + baja confianza → 1.0× learning rate (error esperado)
```

**Señal de error de predicción** que modula la tasa de aprendizaje según la sorpresa. Inspirado en neuronas dopaminérgicas en cerebros biológicos.

#### 3. Regulación homeostática — Escalado sináptico

```python
from iot_machine_learning.infrastructure.ml.cognitive.neural.plasticity import (
    HomeostaticRegulator
)

homeostatic = HomeostaticRegulator(
    target_activity=0.1,      # 10% de neuronas activas (objetivo)
    tau_homeostatic=100000.0,  # ms — muy lento (100 segundos)
)

scaled_weights = homeostatic.regulate(
    weights=current_weights,
    current_activity=0.25,  # 25% activas → escalar hacia abajo
    dt=1.0,
)

# Escalado: w → w * (target_activity / current_activity)
# Previene estados "todo encendido" o "todo apagado"
```

**Plasticidad homeostática** mantiene una tasa de disparo objetivo mediante escalado multiplicativo de pesos. Opera en escalas de tiempo lentas (segundos a minutos) para estabilizar la red.

### Integración con DocumentAnalyzer

```python
# ml_service/api/services/document_analyzer.py

from iot_machine_learning.infrastructure.ml.cognitive.neural import HybridNeuralEngine
from iot_machine_learning.infrastructure.ml.cognitive.neural.competition import NeuralArbiter

class DocumentAnalyzer:
    def __init__(self):
        # Motor neural (opcional, con graceful-fail)
        self._neural_engine = HybridNeuralEngine() if NEURAL_AVAILABLE else None
        self._neural_arbiter = NeuralArbiter() if NEURAL_AVAILABLE else None
    
    def analyze(self, document_id, content_type, payload, tenant_id):
        # 1. Análisis universal (engine cognitivo existente)
        universal_result = analyze_with_universal(...)
        
        # 2. Análisis neural (si está disponible)
        if self._neural_engine:
            analysis_scores = extract_analysis_scores(universal_result)
            neural_result = analyze_with_neural(
                analysis_scores, content_type, domain, self._neural_engine
            )
            
            # 3. Arbitraje entre neural y universal
            winner_result, winner_engine, reason = arbitrate_results(
                neural_result, universal_result, domain, self._neural_arbiter
            )
            
            return {
                "severity": winner_result.severity,
                "confidence": winner_result.confidence,
                "engine_used": winner_engine,  # "neural" o "universal"
                "arbitration_reason": reason,
                "neural_metrics": {
                    "energy_consumed": neural_result.energy_consumed,
                    "active_neurons": neural_result.active_neurons,
                },
            }
```

**Arbitraje** entre motor neural y universal basado en:
1. **Confianza**: comparación de scores con margen
2. **Monte Carlo**: consistencia con simulación MC (si disponible)
3. **Historial por dominio**: qué motor ha ganado más en este dominio

### Métricas de rendimiento

| Métrica | Valor típico |
|---------|--------------|
| **Forward pass SNN** | 50-100ms (dt=0.1ms, 1000 pasos) |
| **Energía por spike** | 1 picojoule (1e-12 J) |
| **Neuronas activas** | 10-30% (regulado homeostáticamente) |
| **Precisión** | Similar a UniversalAnalysisEngine |
| **Confianza** | 0.6-0.9 (mayor cuando SNN y clásica concuerdan) |

### Archivos clave

```
infrastructure/ml/cognitive/neural/
├── hybrid_engine.py              # Orquestador (125 líneas)
├── types.py                      # NeuralResult, SpikePattern, NeuronState
├── pipeline/                     # Etapas modulares (5 archivos)
│   ├── encoder_stage.py
│   ├── snn_stage.py
│   ├── classical_stage.py
│   ├── fusion_stage.py
│   └── decoder_stage.py
├── snn/                          # Red neuronal de spikes
│   ├── neuron.py                 # LIF neuron (220 líneas)
│   ├── membrane.py               # Kernels sinápticos (216 líneas)
│   ├── spike_encoder.py          # Rate + temporal coding (204 líneas)
│   ├── spike_decoder.py          # Spike patterns → severidad
│   └── network.py                # SNNLayer + STDP (274 líneas)
├── classical/                    # Red feedforward clásica
│   ├── feedforward.py            # Capa densa con activaciones
│   ├── activations.py            # relu, sigmoid, softmax, tanh
│   └── online_learner.py         # Aprendizaje por documento
├── competition/                  # Arbitraje neural vs universal
│   ├── arbiter.py                # NeuralArbiter
│   ├── confidence_comparator.py  # Comparación de confianza
│   └── outcome_tracker.py        # Historial por dominio
└── plasticity/                   # Plasticidad avanzada
    ├── metaplasticity.py         # BCM sliding threshold (140 líneas)
    ├── neuromodulation.py        # Señal tipo dopamina (150 líneas)
    └── homeostatic.py            # Regulación homeostática (122 líneas)
```

**Total:** 15 archivos, ~1,835 líneas. Todos los archivos ≤ 280 líneas.

### Principios de diseño

✅ **Realismo biológico**: Parámetros de neurona piramidal cortical  
✅ **Modularidad**: Pipeline de 5 etapas independientes  
✅ **Testabilidad**: Cada componente testeable de forma aislada  
✅ **Degradación graciosa**: Fallback a universal engine si neural falla  
✅ **Zero dependencias nuevas**: Solo numpy (ya existente)  
✅ **Plasticidad avanzada**: BCM + neuromodulación + homeostática  
✅ **Energía realista**: 1 picojoule/spike (tracking completo)  

### Referencias

- Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982). Theory for the development of neuron selectivity: orientation specificity and binocular interaction in visual cortex. *Journal of Neuroscience*, 2(1), 32-48.
- Gerstner, W., & Kistler, W. M. (2002). *Spiking neuron models: Single neurons, populations, plasticity*. Cambridge University Press.
- Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464-10472.

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

Nunca usar `int(series_id)` directamente.

---

## Pipelines de ejecución

### Predicción puntual (HTTP)
```
POST /ml/predict → FastAPI → PredictionService → EngineFactory → Motor → dbo.predictions
```

### Pipeline batch
```
ml_batch_runner → sensores activos → SensorProcessor → predicción + anomalía + severidad → dbo.predictions + dbo.ml_events
```

### Pipeline online (stream)
```
ml_stream_runner → ReadingBroker → SlidingWindowBuffer → patrones (1s/5s/10s) → dbo.ml_events
```

### Pipeline cognitivo (series temporales)
```
MetaCognitiveOrchestrator
  → Perceive  (SignalAnalyzer → StructuralAnalysis)
  → Predict   (engines paralelos)
  → Inhibit   (InhibitionGate → supresión por fallos)
  → Adapt     (PlasticityTracker + AdvancedPlasticityCoordinator)
  → Fuse      (WeightedFusion → peso adaptativo)
  → Explain   (ExplanationBuilder → domain Explanation)
```

### Pipeline cognitivo (texto — TextCognitiveEngine)
```
text_analyzer.py (ml_service)
  → compute_sentiment, compute_urgency, compute_readability, etc.
  → TextAnalysisInput (primitivos)
  → TextCognitiveEngine.analyze(inp, ctx)
      → Perceive  (TextSignalProfiler → SignalSnapshot)
      → Analyze   (TextPerceptionCollector → EnginePerception[] × 5)
      → Remember  (TextMemoryEnricher → TextRecallContext)
      → Reason    (InhibitionGate → WeightedFusion → classify_text_severity)
      → Explain   (TextExplanationAssembler → domain Explanation)
  → build_semantic_conclusion() (ml_service — enriquecimiento humano)
  → Dict legacy (backward-compatible)
```

### Pipeline Zenin — Poller de ingesta de texto
```
zenin_queue_poller.py (daemon thread, arranca con ML Service)
  → SELECT ingestion_queue WHERE Status='pending'
  → UPDATE Status='processing' (optimistic lock)
  → DocumentAnalyzer._analyze_text(content)
  → UPDATE analysis_results SET Status='analyzed', MlResult, Conclusion, TextSummary
  → HTTP POST Weaviate /v1/objects (clase MLExplanation, domainName=zenin_docs)
  → UPDATE ingestion_queue SET Status='completed'
```

**Env vars para activar:**
```bash
ZENIN_QUEUE_POLLER_ENABLED=true
ZENIN_DB_HOST=localhost
ZENIN_DB_PORT=1434
ZENIN_DB_NAME=zenin_db
ZENIN_DB_USER=sa
ZENIN_DB_PASSWORD=<password>
ZENIN_QUEUE_POLL_INTERVAL=5       # segundos entre polls
ZENIN_QUEUE_BATCH_SIZE=10          # items por poll
```

**Principios clave:**
- ML Service es el **único** escritor de resultados ML en `analysis_results`
- .NET Backend **NO** llama HTTP al ML Service durante ingesta
- Comunicación .NET ↔ ML es **exclusivamente vía BD** (queue + results)
- El poller usa `ZeninDbConnection` (conexión separada a `zenin_db`)

---

## Comunicación con otros servicios

| Servicio | Dirección | Detalle |
|---|---|---|
| **SQL Server** (`iot_database`) | Lee/Escribe | `sensor_readings`, `predictions`, `ml_models`, `ml_events`, `alert_thresholds` |
| **SQL Server** (`zenin_db`) | Lee/Escribe | `zenin_docs.ingestion_queue` (lee), `zenin_docs.analysis_results` (escribe) |
| **Ingesta** (`iot_ingest_services`) | Lee | `ReadingBroker` (in-memory), conexión BD compartida |
| **AI Explainer** (`ai-explainer`) | HTTP | `/explain/anomaly` — fallback a templates si no disponible |
| **Weaviate** | HTTP | Memoria cognitiva + indexación de documentos Zenin (`MLExplanation` class) |
| **Backend IoT** (`iot_monitor_backend`) | Indirecto | Consume `predictions`, `ml_events` vía BD |
| **Backend Zenin** (`ZENIN/backend`) | Solo vía BD | Lee `ingestion_queue`, escribe `analysis_results` |

---

## Reglas de dominio

- **Umbrales del usuario tienen prioridad**: si el valor está dentro del rango WARNING configurado, no se genera evento ML.
- **Estados operacionales bloquean ML**: sensores en `INITIALIZING` o `STALE` no generan eventos.
- **Severidad unificada**: `AnomalySeverity.from_score()` es la fuente única de verdad.
- **Inmutabilidad**: todas las entidades de dominio son `frozen=True`. Modificaciones vía `dataclasses.replace()`.
- **Fallbacks claros**: si AI Explainer no responde, se usan templates determinísticos.

---

## Tests

```bash
# Suite completa
python -m pytest tests/ -v

# Solo unitarios
python -m pytest tests/unit/ -v

# Solo integración
python -m pytest tests/integration/ -v
```

| Capa | Tests |
|---|---|
| Domain (entidades, servicios, validadores) | ~200 |
| Infrastructure (motores, filtros, cognitivo, anomalía, text engine) | ~500 |
| Application (use cases, renderer) | ~80 |
| ML Service (runners, metrics, narrator, poller) | ~100 |
| Integration (A/B, enterprise, cognitivo) | ~370 |
| **Total** | **~1260** |

---

## Decisiones técnicas

- **Arquitectura hexagonal**: domain puro → application (use cases) → infrastructure. Dependencias apuntan hacia adentro.
- **UTSAE agnóstico**: `series_id: str` como identidad universal. `ml_service/` mantiene `sensor_id: int` como bridge IoT.
- **Archivos ≤ 300 líneas**: cada módulo tiene responsabilidad única. Math puro separado de orquestación.
- **Plugin architecture**: `@register_engine` y `@register_detector` permiten agregar engines/detectores sin modificar código existente.
- **Dual interface en ports**: métodos legacy (`SensorWindow`) coexisten con agnósticos (`TimeSeries`).
- **Explicabilidad como capa**: domain (value objects) → infra (builder) → application (renderer). Sin acoplamiento.
- **Feature flags**: todo motor nuevo se activa gradualmente vía `FeatureFlags`, con rollback instantáneo a baseline.

---

## Qué NO hace este módulo

- No captura lecturas desde hardware (eso es `iot_ingest_services`).
- No gestiona usuarios/auth (eso es `iot_monitor_backend` / `ZENIN/backend`).
- No define esquemas SQL ni ejecuta migraciones (eso es `iot_database` / `database/migrations`).
- No garantiza tiempo real end-to-end (broker in-memory, no distribuido).
- No parsea archivos subidos (eso lo hace .NET `IngestFileCommandHandler`). ML solo analiza el texto extraído.
