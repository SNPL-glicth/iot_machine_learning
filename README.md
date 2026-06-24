<!-- ZENIN ML BANNER -->
<p align="center">
  <img src="https://img.shields.io/badge/ZENIN%20ML-Cognitive%20Engine-00D4C8?style=for-the-badge&logo=brain&logoColor=white" alt="ZENIN ML"/>
</p>

<p align="center">
  <a href="#pipeline"><img src="https://img.shields.io/badge/15%20Fases%20ML-00D4C8?style=flat-square"/></a>
  <a href="#moe"><img src="https://img.shields.io/badge/MoE%20Bayesiano-00D4C8?style=flat-square"/></a>
  <a href="#audit"><img src="https://img.shields.io/badge/HMAC--SHA256-00D4C8?style=flat-square"/></a>
  <a href="#anomaly"><img src="https://img.shields.io/badge/7%20Detectores%20v2.0-00D4C8?style=flat-square"/></a>
  <a href="#governance"><img src="https://img.shields.io/badge/Governance-00D4C8?style=flat-square"/></a>
  <a href="#inference"><img src="https://img.shields.io/badge/Inferencia%20Bayesiana-00D4C8?style=flat-square"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/SQL%20Server-CC2927?style=flat-square&logo=microsoft-sql-server&logoColor=white"/>
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Weaviate-67C8E0?style=flat-square&logo=weaviate&logoColor=white"/>
  <img src="https://img.shields.io/badge/Prometheus-E6522C?style=flat-square&logo=prometheus&logoColor=white"/>
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white"/>
</p>

<h1 align="center">
  <pre style="background:none;border:none;margin:0;padding:0;font-family:'JetBrains Mono',monospace;color:#00D4C8;">
  ███████╗███████╗███╗   ██╗██╗███╗   ██╗      ███╗   ███╗██╗     
  ╚══███╔╝██╔════╝████╗  ██║██║████╗  ██║      ████╗ ████║██║     
    ███╔╝ █████╗  ██╔██╗ ██║██║██╔██╗ ██║█████╗ ██╔████╔██║██║     
   ███╔╝  ██╔══╝  ██║╚██╗██║██║██║╚██╗██║╚════╝ ██║╚██╔╝██║██║     
  ███████╗███████╗██║ ╚████║██║██║ ╚████║      ██║ ╚═╝ ██║███████╗
  ╚══════╝╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝      ╚═╝     ╚═╝╚══════╝
  </pre>
</h1>

<p align="center"><strong>Motor Cognitivo para Analisis y Prediccion de Datos IoT.</strong></p>
<p align="center">Pipeline de prediccion con Mixture of Experts, fusion bayesiana adaptativa y trazabilidad auditavel. Disenado para multiples dominios: industria, agricultura, energia, salud, ciudades inteligentes y mas.</p>

---

## Vision

ZENIN ML es el nucleo cognitivo de la plataforma ZENIN. Ingiere series temporales de cualquier sensor, ejecuta un <strong>pipeline de 25+ fases especializadas</strong> (evolucion de las 15 fases originales), y entrega una decision clara con confianza calibrada, independientemente del dominio de aplicacion:

<p align="center">
  <img src="https://img.shields.io/badge/ESCALATE-FF3D3D?style=for-the-badge&logo=alert-circle&logoColor=white"/>
  <img src="https://img.shields.io/badge/INVESTIGATE-FFB300?style=for-the-badge&logo=alert-triangle&logoColor=white"/>
  <img src="https://img.shields.io/badge/MONITOR-00E676?style=for-the-badge&logo=activity&logoColor=white"/>
  <img src="https://img.shields.io/badge/LOG__ONLY-4A7A8A?style=for-the-badge&logo=file-text&logoColor=white"/>
</p>

Cada prediccion queda registrada con firma <strong>HMAC-SHA256</strong> en NDJSON append-only para auditoria forense.

---

## <a name="pipeline"></a> Pipeline Cognitivo — 25+ Fases

```
Sensor (MQTT / HTTP)
      |
      ▼
┌──────────────────────────────────────────────────────────────────┐
|  [1]  Sanitize           → NaN/Inf → imputacion; clamp ±6sigma   |
|  [2]  BoundaryCheck      → validacion de rango operativo         |
|  [3]  SeasonalDecomp     → FFT / STL (periodo 24h, min 48pts)   |
|  [4]  Context            → contexto operacional del sensor       |
|  [5]  PredictionReadiness→ verifica disponibilidad de datos      |
|  [6]  Perceive           → regimen: STABLE/TRENDING/VOLATILE/    |
|                            NOISY con analisis estructural        |
|  [7]  DriftDetect        → Page-Hinkley (delta=0.005, lambda=50)|
|  [8]  DriftResponse      → accion correctiva ante deriva         |
|  [9]  Predict            → MoE: Taylor + Statistical + Kalman +  |
|                            Baseline + LightGBM en paralelo       |
|                            (timeout 400ms)                       |
|  [10] Causal             → correlacion causal entre series       |
|  [11] Adapt              → pesos bayesianos por (serie, regimen)  |
|  [12] Inhibit            → suprime engines con error alto        |
|  [13] Fuse               → Hampel k*MAD + consenso ponderado     |
|  [14] DecisionArb        → 8 amplificadores + 3 atenuadores      |
|  [15] CoherenceCheck     → validacion de consistencia temporal    |
|  [16] Memory             → memoria cognitiva (Weaviate)          |
|  [17] ConfCalib          → temperatura por regimen               |
|  [18] Explain            → ReasoningTrace por fase               |
|  [19] ActionGuard        → guardrail antes de emitir accion       |
|  [20] Narrative          → unificacion de explicacion final       |
|  [21] ShadowEvaluation   → evaluacion en segundo plano            |
|  [22] Observability      → metricas y trazas                     |
|  + fases adicionales de ensamble, shadow, y extension            |
└──────────────────────────────────────────────────────────────────┘
      |
      ▼
Prediccion + Confianza + Anomalia + Decision + Narrativa + Audit NDJSON
```

---

## <a name="moe"></a> Mixture of Experts (MoE)

El gating no es global — cada tipo de dispositivo o sensor tiene su propia distribucion de pesos, adaptada a su comportamiento natural:

```python
# Equipo estable (ej. temperatura controlada) -> Taylor domina
STABLE_DEVICE / stable  ->  taylor: 0.65  kalman: 0.15  statistical: 0.10  baseline: 0.10

# Equipo ciclico (ej. riego, ciclos de frio) -> Taylor excluido estructuralmente
CYCLIC_DEVICE / stable       ->  statistical: 0.70  kalman: 0.20  baseline: 0.10  taylor: 0.00
CYCLIC_DEVICE / volatile     ->  statistical: 0.70  kalman: 0.25  baseline: 0.05  taylor: 0.00

# Sensor de vibracion o movimiento -> Kalman domina
VIBRATION_SENSOR / noisy      ->  kalman: 0.75  statistical: 0.15  baseline: 0.05  taylor: 0.05

# Almacenamiento con nivel variable -> Taylor domina (detecta tendencias)
LEVEL_SENSOR / trending       ->  taylor: 0.75  statistical: 0.10  kalman: 0.10  baseline: 0.05
```

**Perfiles de dispositivo extensibles:** ZENIN permite definir nuevas clases de sensores sin modificar el motor central. Perfiles de ejemplo incluidos: `TEMPERATURE · HUMIDITY · VIBRATION · LEVEL · CYCLIC · GENERIC`

---

## Aprendizaje Bayesiano por Sensor

Los pesos NO son globales. Cada sensor aprende de forma independiente:

```
sensor_01 (pasteurizador) -> clave: default:sensor_01:stable -> taylor: 0.71
sensor_02 (pasteurizador) -> clave: default:sensor_02:stable -> taylor: 0.58
```

Estrategia de fallback:

```
< 50 lecturas  ->  cold start: pesos de aptitud por equipment_class (blend gradual)
>= 50 lecturas  ->  pesos per-sensor aprendidos por Bayesian update
sin historial  ->  pesos globales por regimen (comportamiento legacy)
```

---

## <a name="anomaly"></a> Hampel Filter Adaptativo

```python
# k universal antes
result = hampel_filter(perceptions, k=3.0)

# k por tipo de sensor ahora
# TEMPERATURA    -> k=2.5, window=10  (senal suave, estricto)
# CICLO_AGUA     -> k=5.0, window=5   (ciclos abruptos, permisivo)
# CONSUMO_ENERGIA-> k=4.0, window=20  (periodica, ventana larga)
# VIBRACION      -> k=4.5, window=15  (vibracion, permisivo)
result = hampel_filter_with_profile(perceptions, sensor_profile=profile)

# Durante eventos activos (ARRANQUE, CICLO, CAMBIO_DE_PRODUCTO):
# k se amplia automaticamente x 1.5
result = hampel_filter_with_profile(perceptions, sensor_profile=profile, event_context=event)
```

---

## Deteccion de Eventos Contextuales

```python
# Eventos detectados automaticamente desde la senal:
SensorEvent.STARTUP            # rampa abrupta -> arranque de equipo
SensorEvent.CYCLE              # ciclo periodico activo
SensorEvent.PRODUCT_CHANGEOVER # cambio de configuracion o producto
SensorEvent.FAULT_TRANSIENT    # >30% de valores clampeados
SensorEvent.SHUTDOWN           # caida de senal

# Efecto en el pipeline:
# -> InhibitPhase: supresion reducida al 50% (PostEventStabilizationGate)
# -> HampelFilter: k x 1.5 (mas permisivo durante transitorio)
# -> Regimen: reclasificado con contexto temporal (hour_of_day)
```

---

## <a name="audit"></a> Auditoria y Compliance

Cada prediccion genera un registro NDJSON firmado:

```jsonc
{
  "ts": "2024-01-15T06:23:41Z",
  "series_id": "PAST-01",
  "predicted_value": 71.84,
  "confidence": 0.91,
  "regime": "STABLE",
  "equipment_class": "PASTEURIZER",
  "top_expert": "taylor",
  "decision": "MONITOR",
  "reasoning_phases": ["sanitize", "perceive", "predict", "fuse", "explain"],
  "hmac": "sha256:a3f9c2..."   // verificacion independiente via verify_record()
}
```

---

## Capacidades Verificadas en Codigo

| Capacidad | Detalle |
|-----------|---------|
| **Pipeline 25+ fases** | Orden fijo, fases desacoplables por flags, phase files en `orchestration/phases/` |
| **MoE equipment-aware** | Gating por `(equipment_class, regimen)` — 7+ equipos x 4 regimenes |
| **Bayesian per-sensor** | Clave `namespace:series_id:regimen`, fallback a global, cold start blend |
| **Hampel adaptativo** | `k` y `window` desde `SensorProfile`, amplificacion x 1.5 en eventos |
| **Drift detection** | Page-Hinkley online + ADWIN; reset de regimen afectado |
| **Anomaly Ensemble v2.0** | 7 detectores con votacion ponderada, F1=0.2857 (NAB machine temp) |
| **Confidence floor 0.5** | Piso unificado via `CONFIDENCE.MIN_CONFIDENCE` en constantes |
| **Inferencia Bayesiana** | `infrastructure/ml/inference/` — priors, likelihood, posterior, Naive Bayes, MLE |
| **Optimizacion** | `infrastructure/ml/optimization/` — convexa (gradiente, L-BFGS), no convexa (genetico, PSO) |
| **Governance** | 9 componentes: ParameterRegistry, BoundsEnforcer, DynamicTuner, TemperatureScaler, etc. |
| **Warmup** | Precarga de modelos y cache al iniciar (ml_service/warmup.py) |
| **Compliance HMAC** | NDJSON append-only, SHA-256, verificacion constant-time, fsync atomico |
| **Circuit breaker** | CLOSED/OPEN/HALF_OPEN, backoff exponencial, decorador `@protected` |
| **Rate limiter** | Proteccion DoS por serie, configurable |
| **Rollout progresivo** | `FeatureActivator` con whitelist `ML_BATCH_ENTERPRISE_SENSORS` |
| **RUL Estimator** | Estimacion de vida util residual con modelos de regresion |
| **Cognitive Memory** | Weaviate para memoria episodica y semantica (feature flag) |

---

## Capacidades en Desarrollo

| Capacidad | Estado |
|-----------|--------|
| `TextCognitiveEngine` | Verificado en codigo, no integrado en pipeline numerico |
| `HybridNeuralEngine` | Verificado en codigo, no integrado en pipeline numerico |
| `CognitiveMemory / Weaviate` | `ML_ENABLE_COGNITIVE_MEMORY=false` por defecto |
| `SNNLayer con STDP` | Implementado en `cognitive/neural/snn/`, no integrado en pipeline activo |
| Benchmarks NAB/Yahoo S5 | NAB machine temp completado (F1=0.2857 v2.0); Yahoo S5 pendiente |
| Escalabilidad >1000 sensores | Verificado — tests de carga y estres |
| `SeasonalDecomposition` | FFT y STL descomposicion estacional (periodo 24h, min 48pts) |
| `MultivariateEngine` | Correlacion entre sensores via PCA online |
| `CausalAnalysis` | Grafo de dependencia operacional entre series |

---

## Stack

```
API          ->  FastAPI · Uvicorn · Python 3.10+
ML           ->  NumPy · SciPy · scikit-learn · XGBoost · LightGBM (opt)
Inferencia   ->  Bayesiana (priors, likelihood, posterior, Naive Bayes) · MLE
Optimizacion ->  Gradiente · L-BFGS · Newton · Genetico · PSO · Simulated Annealing
Estado       ->  Redis 7 (streams, sliding windows, plasticity, TSDB)
Persistencia ->  SQL Server (pyodbc) · Weaviate (vector DB) · NDJSON append-only
Arquitectura ->  Hexagonal (Ports & Adapters)
Contenedores ->  Docker · docker-compose (4 servicios: ml, redis, weaviate, t2v)
Metricas     ->  Prometheus · MLflow (opt)
Mensajeria   ->  MQTT (paho-mqtt) · RabbitMQ (opt)
Seguridad    ->  HMAC-SHA256 · API Key · Rate Limiter · Circuit Breaker
```

---

## Inicio Rapido

```bash
# 1. Infraestructura
docker run -d --name redis-zenin -p 6379:6379 redis:7-alpine
docker run -d --name sql-zenin -p 1433:1433 \
  -e SA_PASSWORD=YourPassword123 \
  -e ACCEPT_EULA=Y \
  mcr.microsoft.com/mssql/server:2022-latest

# 2. Configuracion
cp .env.example .env
# Editar .env con credenciales reales

# 3. Levantar
uvicorn ml_service.main:app --host 0.0.0.0 --port 8002 --reload

# 4. Verificar
curl http://localhost:8002/
# {"service": "iot-ml-service", "status": "ok"}

# 5. Prediccion de ejemplo
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "series_id": "PAST-01",
    "values": [71.2, 71.4, 71.3, 71.5, 71.4],
    "timestamps": [1700000000, 1700000300, 1700000600, 1700000900, 1700001200]
  }'
```

---

## Validacion con Datos Industriales Reales (ALPLA + NAB)

**Junio 2026** — Pipeline MoE cognitivo ejecutado sobre dataset industrial de 47 parámetros (Chiller + Air Compressor, Ene 2025 – May 2026).
**Anomaly Ensemble v2.0** validado sobre NAB `machine_temperature_system_failure`.

### Resultados — Pipeline Cognitivo (ALPLA)

| Métrica | Valor |
|---------|-------|
| Parámetros analizados | 47 (18 Chiller + 29 CA) |
| Confianza promedio fused | **0.55** (moderada, con floor 0.5) |
| Matching expert→régimen | **100%** (64% volatile→taylor, 26% stable→baseline, 11% noisy→kalman) |
| Tiempo total | ~1.1s (23ms por parámetro) |

### Resultados — Anomaly Ensemble v2.0 (NAB)

| Métrica | v1.0 | v2.0 |
|---------|:----:|:----:|
| F1 | 0.164 | **0.2857** |
| FP | 73 | **24** |
| Recall | 0.167 | 0.2143 |
| Precision | 0.161 | — |
| Cliff's delta | — | 0.7261 |
| Validación | — | Grid search 243 combinaciones |

### Anomalías Detectadas

| Equipo | Parámetro | Evento | Fecha |
|--------|-----------|--------|-------|
| Chiller | Consumo RTAE 5 | 9.7M → **97.9M** (10×) | 2025-11-27 |
| Chiller | Cto.2 N° arranques | 4,520 → **591M** (130,000×) | 2025-12-02 |
| Chiller | Tiempo operación compresor | 63 → **63,061** (1,000×) | 2025-12-26 |
| CA | Horas de carga | 90,037 → **900,052** (10×) | 2025-08-16 |
| CA | Punto de rocío secador | 3.3°C → **27°C** | 2025-08-23 |

### Bugs Corregidos en Validación

1. **Registro duplicado de `EngineFactory`** — imports FQN vs relativos creaban dos clases en memoria
2. **`confidence` vs `confidence_score`** — `AttributeError` silencioso en MoE fusion
3. **Doble penalización** — MoE + runner aplicaban penalizaciones por separado
4. **Taylor floor 0.30 → 0.50** — elevado para datos industriales
5. **Anomaly v1.0 adaptativo** — pesos recalculados silenciosamente causaban inestabilidad (v2.0 eliminó pesos adaptativos)
6. **Drift coupling en anomalía** — sobreescribía pesos configurados sin advertencia (eliminado en v2.0)

---

## Comparacion de Mercado

| Capacidad | ZENIN | AWS Lookout | Azure AD | Palantir |
|-----------|:-----:|:-----------:|:--------:|:--------:|
| Pesos bayesianos online (sin retraining) | :white_check_mark: | :x: | :x: | :x: |
| Equipment awareness por tipo de equipo | :white_check_mark: | :x: | :x: | :x: |
| MoE routing contextual | :white_check_mark: | :x: | :x: | :x: |
| Decision con guardrails auditables | :white_check_mark: | :x: | :warning: | :white_check_mark: |
| Explicacion trazable por fase | :white_check_mark: | :warning: | :warning: | :white_check_mark: |
| Compliance HMAC-SHA256 nativo | :white_check_mark: | :x: | :x: | :x: |
| Deploy on-premise sin cloud | :white_check_mark: | :x: | :x: | :warning: |
| velocity_z + acceleration_z | :white_check_mark: | :x: | :x: | :x: |
| Escalabilidad >1000 sensores | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Benchmarks publicos validados | :x: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

---

## Estructura del Repositorio

```
iot_machine_learning/
├── domain/                       # Entidades, puertos, politicas — zero infra deps
│   ├── entities/                 # Value objects frozen (Prediction, Anomaly, Explanation...)
│   ├── ports/                    # 23 interfaces (StoragePort, AuditPort, CognitivePort...)
│   ├── services/                 # 33 servicios de dominio (anomaly, actions, cognitive, prediction...)
│   ├── policies/                 # ActionBuilders, ContextPolicy, ThresholdPolicy
│   ├── tools/                    # Tool registry, executors, guards
│   ├── validators/               # DataSanitizer, InputGuard, validadores numericos
│   └── value_objects/            # EquipmentClass, IndustrialEvent, SensorProfile
├── application/
│   ├── use_cases/                # 7 casos de uso (predict, detect anomalies, analyze patterns...)
│   ├── services/                 # DecisionService
│   ├── evaluation/               # Dataset evaluation, metrics, quality scoring
│   ├── explainability/           # ExplanationRenderer
│   └── semantic_extraction/      # Entity prioritizer
├── infrastructure/
│   ├── ml/
│   │   ├── cognitive/            # Pipeline 25+ fases + BayesianWeightTracker
│   │   │   ├── orchestration/    # MetaCognitiveOrchestrator + 25+ phase files
│   │   │   ├── bayesian_weight_tracker/  # 33 archivos — per-sensor learning
│   │   │   ├── fusion/           # WeightedFusion, HampelFilter adaptativo
│   │   │   ├── inhibition/       # InhibitionGate + AdaptiveConfig
│   │   │   ├── drift/            # Page-Hinkley, ADWIN, ErrorDriftDetector
│   │   │   ├── plasticity/       # PlasticityTracker, AdvancedPlasticityCoordinator
│   │   │   ├── narrative/        # Generacion de narrativa (embedding network)
│   │   │   ├── compliance/       # HMAC-SHA256, ComplianceExporter
│   │   │   ├── neural/           # HybridEngine, SNN, attention
│   │   │   └── ... +20 subdirs
│   │   ├── engines/              # Taylor, Statistical, Baseline, Kalman, LightGBM
│   │   ├── anomaly/              # VotingAnomalyDetector v2.0 + 7 detectores
│   │   ├── moe/                  # MoE: registry, gating tree, experts, fusion, rollout
│   │   ├── inference/            # Bayesiana (prior, likelihood, posterior, NaiveBayes) + MLE
│   │   ├── optimization/         # Convexa (gradient, L-BFGS) + No convexa (genetic, PSO)
│   │   ├── filters/              # Kalman, EMA, Median, FilterChain
│   │   ├── patterns/             # ChangePoint, DeltaSpike, RegimeDetector
│   │   ├── calibration/          # ConfidenceCalibrator
│   │   ├── explainability/       # FeatureImportance
│   │   └── benchmark/            # BenchmarkRunner, DatasetLoader, Metrics
│   ├── adapters/                 # SQL, Weaviate, MLflow, CognitiveMemory, IOT
│   ├── persistence/              # SQL Server, Redis, Weaviate (vector)
│   ├── resilience/               # CircuitBreaker
│   ├── security/                 # AccessControl, AuditLogger, AuthProvider, RateLimiter
│   └── config/                   # MoE factory
├── ml_service/                   # FastAPI app, runners, metrics, workers, governance
│   ├── api/                      # 9 route files, schemas, dependencies
│   ├── runners/                  # Batch, Stream, CLI, Worker
│   ├── metrics/                  # A/B testing, prometheus, observability
│   ├── config/                   # Feature flags (40+), ML config
│   ├── workers/                  # Queue poller, job processor
│   ├── orchestrator/             # PredictionOrchestrator
│   ├── warmup.py                 # Precarga de modelos al iniciar
│   └── governance_initializer.py # 9-componentes de governance
├── core/                         # Ensemble, drift, parameters, tuning, statistical
├── benchmarks/                   # NAB, memory, explainability, forensic
└── tests/                        # 3,600+ tests (unit, integration, load, stress, property)
```

---

## Documentacion

| Tema | Archivo |
|------|---------|
| Arquitectura hexagonal | `docs/architecture.md` |
| Pipeline ML | `docs/ml_pipeline.md` |
| Motores de prediccion | `docs/ENGINES.md` |
| Deteccion de anomalias | `docs/anomaly_detection.md` |
| Drift y adaptacion | `docs/drift_and_adaptation.md` |
| Compliance y auditoria | `docs/compliance_and_audit.md` |
| Plasticidad y aprendizaje | `docs/plasticity.md` |
| Feature flags | `docs/configuration.md` |
| Inicio rapido | `docs/QUICKSTART.md` |
| Operaciones | `docs/operations.md` |
| Desarrollo | `docs/development.md` |
| ROI y casos de uso | `docs/roi_and_business_case.md` |
| Backlog de rendimiento | `docs/BACKLOG_PERFORMANCE.md` |

---

## Metricas

| Metrica | Valor |
|---------|-------|
| Fases del pipeline | 25+ (en `orchestration/phases/`) |
| Detectores de anomalias | 7 (v2.0, votacion ponderada) |
| Experts MoE | 5 (Taylor, Statistical, Kalman, Baseline, LightGBM) |
| Regimenes operativos | 4 (STABLE, TRENDING, VOLATILE, NOISY) |
| Equipment classes | 7+ (TEMPERATURE, HUMIDITY, VIBRATION, LEVEL, CYCLIC, GENERIC +) |
| Decision levels | 4 (ESCALATE, INVESTIGATE, MONITOR, LOG_ONLY) |
| Tests automatizados | 3,600+ |
| Componentes governance | 9 |
| Anomaly F1 (NAB v2.0) | 0.2857 |
| Confidence floor | 0.5 (para datos industriales) |
| Arquitectura | Hexagonal (Ports & Adapters) |
| Contenedores | 4 servicios (ml, redis, weaviate, t2v-transformers) |

---

> Licencia: Propietaria — Todos los derechos reservados.
