# iot_machine_learning

Motor de Machine Learning y enriquecimiento de contexto para el sistema IoT **Sandevistan**.

## Arquitectura

El proyecto sigue **arquitectura hexagonal** con principios **UTSAE** (Universal Time Series Analysis Engine). Cada capa tiene una responsabilidad clara y las dependencias apuntan hacia adentro.

```
iot_machine_learning/
├── domain/                          # Capa de dominio (pura, sin I/O)
│   ├── entities/                    # Entidades inmutables
│   │   ├── sensor_reading.py        # SensorReading, SensorWindow
│   │   ├── prediction.py            # Prediction, PredictionConfidence
│   │   ├── anomaly.py               # AnomalyResult
│   │   └── pattern.py               # PatternResult, ChangePoint, DeltaSpike, OperationalRegime
│   ├── ports/                       # Interfaces abstractas (contratos)
│   │   ├── prediction_port.py       # PredictionPort
│   │   ├── anomaly_detection_port.py
│   │   ├── pattern_detection_port.py
│   │   ├── storage_port.py          # StoragePort (persistencia)
│   │   └── audit_port.py            # AuditPort (ISO 27001)
│   ├── services/                    # Servicios de dominio (lógica pura)
│   │   ├── prediction_domain_service.py
│   │   ├── anomaly_domain_service.py
│   │   └── pattern_domain_service.py
│   └── validators/                  # Validaciones numéricas reutilizables
│       └── numeric.py               # validate_window, clamp_prediction, safe_float
│
├── application/                     # Capa de aplicación (casos de uso)
│   ├── use_cases/
│   │   ├── predict_sensor_value.py  # PredictSensorValueUseCase
│   │   ├── detect_anomalies.py      # DetectAnomaliesUseCase
│   │   ├── analyze_patterns.py      # AnalyzePatternsUseCase
│   │   └── select_engine.py         # Selección de motor por feature flags
│   └── dto/                         # Data Transfer Objects
│       ├── prediction_dto.py
│       ├── anomaly_dto.py
│       └── pattern_dto.py
│
├── infrastructure/                  # Capa de infraestructura (implementaciones concretas)
│   ├── ml/
│   │   ├── interfaces.py            # PredictionEngine, PredictionResult, SignalFilter (ABCs)
│   │   ├── engines/
│   │   │   ├── engine_factory.py    # EngineFactory (registry + create)
│   │   │   ├── baseline_engine.py   # Media movil simple (fallback)
│   │   │   ├── taylor_engine.py     # Series de Taylor con diferencias finitas
│   │   │   ├── taylor_math.py       # Funciones matematicas puras de Taylor
│   │   │   ├── taylor_adapter.py    # Adapter: PredictionEngine -> PredictionPort
│   │   │   ├── baseline_adapter.py  # Adapter: Baseline -> PredictionPort
│   │   │   └── ensemble_weighted.py # Predictor con pesos dinamicos
│   │   ├── filters/
│   │   │   ├── kalman_filter.py     # Kalman 1D con warmup y auto-calibracion
│   │   │   └── kalman_math.py       # Estado, calibracion y update puros
│   │   ├── anomaly/
│   │   │   └── voting_anomaly_detector.py  # Z-score + IQR + IF + LOF (voting)
│   │   ├── patterns/
│   │   │   ├── delta_spike.py       # Clasificador de delta/spikes
│   │   │   ├── cusum_detector.py    # CUSUM para cambios de media
│   │   │   ├── pelt_detector.py     # PELT para change points
│   │   │   └── regime_detector.py   # Detector de regimenes operacionales
│   │   └── explainability/
│   │       ├── taylor_importance.py # Feature importance via Taylor
│   │       └── counterfactual.py    # Explicaciones contrafactuales
│   ├── adapters/
│   │   ├── sqlserver_storage.py     # StoragePort -> SQL Server
│   │   ├── prediction_cache.py      # LRU + TTL cache
│   │   └── batch_predictor.py       # ThreadPool + CircuitBreaker
│   ├── security/
│   │   ├── file_audit_logger.py     # AuditPort -> archivos (ISO 27001)
│   │   ├── null_audit_logger.py     # AuditPort no-op
│   │   └── access_control.py        # RBAC
│   └── repositories/
│
├── ml_service/                      # Servicio HTTP + runners (orquestacion)
│   ├── main.py                      # FastAPI app
│   ├── config/
│   │   ├── ml_config.py             # Configuracion global
│   │   └── feature_flags.py         # Feature flags (rollback, whitelist, etc.)
│   ├── api/services/
│   │   └── prediction_service.py    # Servicio HTTP de prediccion
│   ├── runners/
│   │   ├── ml_batch_runner.py       # Pipeline batch (cron)
│   │   ├── ml_stream_runner.py      # Pipeline online (stream)
│   │   └── common/
│   │       ├── sensor_processor.py  # Procesamiento por sensor
│   │       ├── severity_classifier.py
│   │       ├── prediction_narrator.py
│   │       ├── prediction_writer.py
│   │       └── event_writer.py
│   ├── orchestrator/                # Enriquecimiento + explicacion
│   ├── explain/                     # AI Explainer client + templates
│   ├── trainers/                    # Entrenamiento de modelos
│   ├── metrics/                     # A/B testing
│   ├── memory/                      # Memoria de decisiones
│   ├── correlation/                 # Correlacion entre sensores
│   └── repository/                  # Repositorios SQL
│
└── tests/                           # 413 tests
    ├── unit/
    │   ├── domain/                  # Tests de entidades y servicios puros
    │   ├── infrastructure/          # Tests de motores, filtros, math
    │   ├── application/             # Tests de use cases
    │   └── ml_service/              # Tests de servicio HTTP y runners
    └── integration/                 # Tests de integracion (A/B, pipelines)
```

## Motores de prediccion

| Motor | Ubicacion | Descripcion |
|-------|-----------|-------------|
| **Baseline** | `engines/baseline_engine.py` | Media movil simple. Fallback por defecto. |
| **Taylor** | `engines/taylor_engine.py` + `taylor_math.py` | Series de Taylor con diferencias finitas (orden 1-3). Clamp obligatorio. |
| **Ensemble** | `engines/ensemble_weighted.py` | Pesos dinamicos con auto-tuning. |

La seleccion de motor se controla via **feature flags** (`select_engine.py`):

1. `ML_ROLLBACK_TO_BASELINE` = true -> baseline (panic button)
2. Override por sensor en `ML_ENGINE_OVERRIDES`
3. Sensor en whitelist de Taylor -> taylor
4. `ML_DEFAULT_ENGINE` global
5. Fallback a baseline

## Filtros de senal

| Filtro | Ubicacion | Descripcion |
|--------|-----------|-------------|
| **Kalman 1D** | `filters/kalman_filter.py` + `kalman_math.py` | Auto-calibracion de R durante warmup. Thread-safe. |
| **Identity** | `interfaces.py` | No-op (fallback cuando Kalman esta desactivado). |

## Deteccion de anomalias

**Voting ensemble** (`anomaly/voting_anomaly_detector.py`) con 4 metodos:

- **Z-score** (peso 0.25) — desviacion estadistica
- **IQR** (peso 0.15) — rango intercuartilico
- **Isolation Forest** (peso 0.30) — sklearn
- **Local Outlier Factor** (peso 0.30) — sklearn

Consenso por voto ponderado con umbral configurable.

## Deteccion de patrones

| Detector | Descripcion |
|----------|-------------|
| **DeltaSpikeClassifier** | Clasifica cambios bruscos (spikes, drops) |
| **CUSUMDetector** | Cambios de media acumulativos |
| **PELTDetector** | Change points por penalizacion |
| **RegimeDetector** | Regimenes operacionales |

## Pipelines de ejecucion

### Prediccion puntual (HTTP)

```
POST /ml/predict -> FastAPI -> PredictionService -> EngineFactory -> Motor -> dbo.predictions
```

### Pipeline batch

```
ml_batch_runner -> sensores activos -> SensorProcessor -> prediccion + anomalia + severidad + narrativa -> dbo.predictions + dbo.ml_events
```

### Pipeline online (stream)

```
ml_stream_runner -> ReadingBroker -> SlidingWindowBuffer -> patrones (1s/5s/10s) -> dbo.ml_events + dbo.alert_notifications
```

## Comunicacion con otros servicios

| Servicio | Direccion | Detalle |
|----------|-----------|---------|
| **SQL Server** (`iot_database`) | Lee/Escribe | `sensor_readings`, `predictions`, `ml_models`, `ml_events`, `alert_thresholds`, `alert_notifications` |
| **Ingesta** (`iot_ingest_services`) | Lee | `ReadingBroker` (in-memory), conexion BD compartida |
| **AI Explainer** (`ai-explainer`) | HTTP | `/explain/anomaly` — fallback a templates si no disponible |
| **Backend** (`iot_monitor_backend`) | Indirecto | Consume `predictions`, `ml_events`, `notifications` via BD |

## Reglas de dominio

- **Umbrales del usuario tienen prioridad**: si el valor esta dentro del rango WARNING configurado, no se genera evento ML.
- **Estados operacionales bloquean ML**: sensores en `INITIALIZING` o `STALE` no generan eventos.
- **Fallbacks claros**: si AI Explainer no responde, se usan templates deterministicos.

## Tests

```bash
# Ejecutar suite completa (413 tests)
python -m pytest iot_machine_learning/tests/ -v

# Solo tests unitarios
python -m pytest iot_machine_learning/tests/unit/ -v

# Solo tests de integracion
python -m pytest iot_machine_learning/tests/integration/ -v
```

| Capa | Tests | Cobertura |
|------|-------|-----------|
| Domain (entidades, servicios, validadores) | ~80 | Logica pura, sin mocks |
| Infrastructure (motores, filtros, anomalia) | ~120 | Math puro + adapters |
| Application (use cases, select_engine) | ~30 | Mocks de ports |
| ML Service (prediction, narrator, severity) | ~50 | Mocks de BD |
| Integration (A/B testing, pipelines) | ~130 | End-to-end sin BD real |

## Decisiones tecnicas

- **Arquitectura hexagonal**: domain puro (sin I/O) -> application (use cases) -> infrastructure (implementaciones). Dependencias apuntan hacia adentro.
- **UTSAE**: Sensing, Modeling, Reasoning, Narrative, Adaptation, Orchestration como fases del pipeline.
- **Feature flags**: todo motor nuevo se activa gradualmente via flags, con rollback instantaneo a baseline.
- **Archivos < 180 lineas**: cada modulo tiene responsabilidad unica. Math puro separado de orquestacion.
- **FastAPI** para prediccion puntual: endpoints simples, bajo overhead.
- **Persistencia en BD**: `predictions`, `ml_events`, `alert_notifications` para que backend y UI consuman snapshots sin recalcular.

## Que NO hace este modulo

- No captura lecturas desde hardware (eso es `iot_ingest_services`).
- No gestiona usuarios/auth (eso es `iot_monitor_backend`).
- No define esquemas SQL ni ejecuta migraciones (eso es `iot_database`).
- No garantiza tiempo real end-to-end (broker in-memory, no distribuido).
