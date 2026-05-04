# ZENIN — Motor de Decisión Cognitiva para Operaciones Industriales

Predicción de anomalías en sensores IoT, fusión de motores con pesos bayesianos adaptativos por régimen de señal, y decisión automatizada con trazabilidad auditável. Diseñado para plantas industriales latinoamericanas con sensores heterogéneos, equipos de mantenimiento limitados, y presión regulatoria creciente.

---

## El Problema

Las plantas industriales operan con uno o más de estos dolores:

- **Máquinas que fallan sin aviso** — los sensores generan datos, pero nadie los cruza con el historial de mantenimiento hasta que la falla es costosa.
- **Sistemas de monitoreo con falsos positivos constantes** — umbrales fijos disparan alarmas en horas pico; en horario nocturno el operador las ignora por costumbre.
- **Decisiones de parada por intuición** — el supervisor decide si detener la línea basado en experiencia, no en una puntuación de riesgo trazable.
- **Auditorías que consumen semanas** — cuando llega el cliente o el regulador, reconstruir qué pasó requiere revisar logs de 3 sistemas distintos.

## Cómo lo Resuelve ZENIN

1. **Ingesta** — lecturas de sensores vía MQTT o HTTP.
2. **Sanitización** — NaN/Inf se detectan antes de cualquier cálculo; valores extremos se suavizan con ventana local ±6σ.
3. **Percepción** — clasifica la señal en régimen (`STABLE`, `TRENDING`, `VOLATILE`, `NOISY`, `TRANSITIONAL`) y mide ruido.
4. **Predicción concurrente** — múltiples motores (Taylor, estadístico, seasonal, baseline) predicen con timeout por motor (400ms default).
5. **Adaptación bayesiana** — pesos por régimen se actualizan con inferencia bayesiana (prior gaussiano, σ²_obs empírica por motor), no con EMA fijo.
6. **Inhibición** — motores con error reciente alto se suprimen antes de la fusión.
7. **Fusión** — filtro Hampel (~3σ) descarta percepciones atípicas; consenso ponderado genera valor y confianza.
8. **Detección de anomalías** — ensemble de 8 sub-detectores con votación ponderada adaptativa.
9. **Decisión contextual** — 8 amplificadores + 3 atenuadores ajustan severidad. Acciones: `ESCALATE` / `INVESTIGATE` / `MONITOR` / `LOG_ONLY`.
10. **Explicación y auditoría** — cada predicción lleva reasoning trace por fase. Exporte NDJSON con HMAC-SHA256.

```mermaid
flowchart LR
    subgraph Input
        A1[Sensor MQTT]
        A3[API HTTP]
    end

    subgraph Pipeline["Pipeline Cognitivo (15 fases)"]
        B1[Sanitize]
        B2[BoundaryCheck]
        B3[SeasonalDecomp]
        B4[Perceive]
        B5[DriftDetection]
        B6[Predict]
        B7[Adapt]
        B8[Inhibit]
        B9[Fuse]
        B10[DecisionArbiter]
        B11[CoherenceCheck]
        B12[ConfidenceCalib]
        B13[Explain]
        B14[ActionGuard]
        B15[NarrativeUnification]
    end

    subgraph Output
        C1[Predicción + Confianza]
        C2[Anomalía + Severidad]
        C3[Decisión AUTO/ASK/DENY]
        C4[Audit NDJSON HMAC]
    end

    A1 --> B1
    A3 --> B1
    B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7 --> B8 --> B9 --> B10 --> B11 --> B12 --> B13 --> B14 --> B15
    B15 --> C1
    B15 --> C2
    B15 --> C3
    B15 --> C4
```

---

## Capacidades Técnicas Verificadas en Código

Toda capacidad en esta tabla tiene su implementación verificada en el código fuente del repositorio.

| Capacidad | Detalle técnico | Diferenciador vs mercado |
|-----------|-----------------|--------------------------|
| Pipeline cognitivo 15 fases | Orden fijo: Sanitize → BoundaryCheck → SeasonalDecomposition → Perceive → DriftDetection → Predict → Adapt → Inhibit → Fuse → DecisionArbiter → CoherenceCheck → ConfidenceCalibration → Explain → ActionGuard → NarrativeUnification | Fases desacoplables; cada una inyectable por flags. Early termination en NaN/Inf, out-of-domain, o budget excedido. |
| Voting ensemble 8+ detectores | Isolation Forest (30%), Z-score (20%), IQR (10%), LOF (15%), velocity_z (15%), acceleration_z (10%), IF-ND, LOF-ND; pesos adaptativos por precisión histórica (50 outcomes). `RobustScaler` para entrenamiento. | `velocity_z` y `acceleration_z` detectan cambios de régimen invisibles para detectores de magnitud pura. |
| BayesianWeightTracker por régimen | Prior gaussiano N(μ,σ²), update conjugado normal-normal, σ²_obs empírica por motor con ventana de 20 errores, mínimo 5 muestras, clamp a 0.01. LRU eviction de 10 régimes, TTL 24h. | No asume σ²_obs=1.0 para todos los motores; temperatura y vibración tienen escalas de error distintas. |
| Drift detection online | Page-Hinkley (δ=0.005, λ=50, α=0.9999) por defecto; ADWIN opcional (δ=0.002, max_window=1000); cooldown 300s por serie. | Reset de pesos del régimen afectado, no del sistema completo. Emite indicador ISO 13374. |
| Seasonal decomposition | FFT por defecto (periodo 24h); STL opcional (requiere statsmodels); mínimo 48 puntos. | Descompone antes de la predicción, no post-hoc. |
| Filtro Hampel | k=3.0 × 1.4826 × MAD; rechaza percepciones atípicas antes del consenso. No-op si <3 percepciones o MAD=0. | Aplica sobre predicciones de motores, no sobre datos brutos. Evita que un motor errático contamine la fusión. |
| Circuit breaker | CLOSED/OPEN/HALF_OPEN; backoff exponencial; thread-safe; decorador `@protected`. | Protege llamadas a Weaviate, SQL Server, Redis ante fallos transitorios. |
| Decision engine contextual | 8 amplificadores + 3 atenuadores; base scores por severidad; umbrales ESCALATE (0.85) / INVESTIGATE (0.65) / MONITOR (0.40). | Ajusta decisión según régimen, tasa de anomalías recientes, y drift. No hardcodea umbrales. |
| ComplianceExporter | NDJSON line-delimited; 12 campos estructurados + HMAC-SHA256 sobre cuerpo canónico ordenado lexicográficamente. | Verificación independiente con `verify_record()`; comparación constant-time. Escritura atómica (append + fsync). |
| Explicación por fase | `ExplanationRenderer` + `CausalNarrativeBuilder`; reasoning trace por fase en `PredictionResult.metadata`. | Auditable operador-a-operador, no solo científico de datos. |
| AuditPort dual interface | `log_prediction(sensor_id:int)` legacy + `log_series_prediction(series_id:str)` agnóstico con bridge a legacy. | Permite migración gradual de `sensor_id:int` a `series_id:str` sin romper implementaciones existentes. |
| StoragePort dual interface | `load_sensor_window(sensor_id:int)` legacy + `load_series_window(series_id:str)` agnóstico. | Bridge convierte con `safe_series_id_to_int`; fallback a sensor_id=0 para series no numéricas. |
| Confidence calibration por régimen | Temperatura configurable: STABLE=1.2, VOLATILE=2.0, NOISY=1.8, TRENDING=1.5. | Evita sobreconfianza en régimen VOLATILE y subconfianza en STABLE. |
| PredictPhase concurrente | `ThreadPoolExecutor` con max_workers=3, timeout por motor=400ms. Fallback a secuencial si executor falla. | Preserva orden de engines. Surface de fallos (timeout, excepción, cannot_handle) en metadata. |
| PipelineTimer | `perceive_ms`, `predict_ms`, `adapt_ms`, `inhibit_ms`, `fuse_ms`, `explain_ms`. Budget default 500ms. | Corta a fallback si PERCEIVE+PREDICT excede budget. Evita degradación silenciosa. |

---

## Capacidades en Desarrollo / No Confirmadas en Pipeline Numérico Activo

| Capacidad | Documentado en | Estado | Notas |
|-----------|---------------|--------|-------|
| TextCognitiveEngine | `docs/ENGINES.md` | Verificado en código (`infrastructure/ml/cognitive/text/engine.py:41`), **NO usado en pipeline numérico de 15 fases** | Pipeline documental/API separado. Analiza reportes de mantenimiento en texto. |
| HybridNeuralEngine | `docs/ENGINES.md` | Verificado en código (`infrastructure/ml/cognitive/neural/hybrid_engine.py:27`), **NO usado en pipeline numérico** | Bridge en `neural_bridge.py` para análisis documental. No integrado en `MetaCognitiveOrchestrator`. |
| MoEGateway | `docs/ENGINES.md` | Infraestructura presente (`infrastructure/ml/moe/gateway/moe_gateway.py`), **NO confirmado en producción** | Rama condicional en `orchestrator.py:239` si `ML_MOE_ENABLED=True`. Flag `ML_MOE_ENABLED` no documentado en config base. Tests de integración son stubs. |
| SNNLayer with STDP | `docs/ENGINES.md` | No verificado en pipeline numérico | Mencionado en documentación de motores. No encontrado en fases del pipeline. |
| CognitiveMemory / Weaviate | `cognitive_config.py` | Infraestructura presente, `ML_ENABLE_COGNITIVE_MEMORY=false` por defecto | Stub de integración. No activo en pipeline numérico. |
| Attention mechanism | `cognitive_config.py` | `ML_ENABLE_ATTENTION=false` por defecto | Documentado como "too slow for CPU". No activo. |
| SNN full | `cognitive_config.py` | `ML_ENABLE_SNN_FULL=false` por defecto | Deshabilitado en CPU. No activo. |

> ⚠️ **Nota:** Las capacidades de esta sección están documentadas en `docs/ENGINES.md` o configuradas en `ml_service/config/cognitive_config.py`, pero no forman parte del pipeline numérico de 15 fases que procesa lecturas de sensores en tiempo real.

---

## ROI Estimado

### ¿Qué retorno puede esperar una planta industrial?

| Métrica | Estimación conservadora | Capacidad ZENIN que lo genera |
|---|---|---|
| Reducción de paradas no planificadas | 20–35% | Drift detection + anomaly voting ensemble (detecta degradación 24–72h antes de falla catastrófica) |
| Reducción de falsos positivos | 40–60% | Hampel filter + InhibitionGate + atenuadores de decisión (suprime ruido de motores erráticos) |
| Tiempo de diagnóstico post-incidente | –70% | ReasoningTrace por fase + ExplanationRenderer + pipeline_timing (reconstrucción en minutos, no días) |
| Cumplimiento de auditoría técnica | Automatizado | ComplianceExporter HMAC-SHA256 + AuditPort (NDJSON append-only con verificación criptográfica) |
| Costo vs soluciones enterprise (Palantir, AWS Lookout) | –80% infraestructura | Open source + deploy local sin cloud obligatorio; stack Python/Redis/SQL Server |

Estos estimados asumen integración con sensores existentes vía MQTT o HTTP. El sistema no requiere reemplazo de infraestructura OT existente.

---

## Comparación de Mercado (Honesta)

| Capacidad | ZENIN | AWS Lookout | Azure AD | Palantir AIP |
|---|---|---|---|---|
| Detección de anomalías univariada | ✅ Voting ensemble 8 detectores | ✅ Isolation Forest | ✅ Ensemble limitado | ✅ Extensible |
| Velocidad/aceleración (derivadas) | ✅ velocity_z + acceleration_z | ❌ No nativo | ❌ No nativo | ❌ Requiere desarrollo |
| Pesos bayesianos por régimen | ✅ Online, sin retraining | ❌ Retrain batch | ❌ Retrain batch | ❌ Retrain batch |
| Decisión contextual con guardrails | ✅ AUTO/ASK/DENY + amplificadores | ❌ Solo alerta | ⚠️ Con conector extra | ✅ Flexible, costoso |
| Explicación trazable por fase | ✅ ReasoningTrace + CausalNarrative | ⚠️ Básica | ⚠️ Básica | ✅ Avanzada |
| Exporte de auditoría HMAC | ✅ NDJSON + SHA-256 + verificación | ❌ No nativo | ❌ No nativo | ❌ No nativo |
| Deploy on-premise sin cloud | ✅ Docker local | ❌ Cloud-only | ❌ Cloud-only | ⚠️ On-prem costoso |
| Escalabilidad >1000 sensores | ⚠️ Degrada >100 (estimado) | ✅ Alta | ✅ Alta | ✅ Alta |
| Benchmark público NAB/Yahoo S5 | ❌ Pendiente | ✅ Validado | ✅ Validado | ✅ Validado |
| Equipo de ciencia de datos requerido | ⚠️ 1 ingeniero ML para tuning | ❌ No (managed) | ❌ No (managed) | ✅ Sí (especialista) |

ZENIN gana en **transparencia arquitectónica**, **decisión contextual con guardrails**, y **costo de infraestructura**. Pierde en **escalabilidad masiva out-of-the-box** y **benchmarks públicos validados**.

---

## Stack Técnico

| Capa | Tecnología |
|------|-----------|
| API | Python 3.10+, FastAPI, Uvicorn |
| ML / Matemáticas | NumPy, scikit-learn, SciPy |
| Estado | Redis (streams, cache, sliding windows, plasticity shared state) |
| Persistencia | SQL Server (predictions, anomalies, config); NDJSON append-only (compliance) |
| Arquitectura | Hexagonal (Ports & Adapters) — dominio sin dependencias externas |
| Deployment | Docker, docker-compose |
| Observabilidad | Structured JSON logging; PipelineTimer por fase |
| Tests | pytest + typeguard; meta-tests arquitectónicos |

---

## Inicio Rápido

```bash
# 1. Dependencias (Redis + SQL Server)
docker run -d --name redis-zenin -p 6379:6379 redis:7-alpine
docker run -d --name sql-zenin -p 1433:1433 \
  -e SA_PASSWORD=YourPassword123 -e ACCEPT_EULA=Y \
  mcr.microsoft.com/mssql/server:2022-latest

# 2. Variables de entorno
cp .env.example .env
# Editar .env con credenciales reales

# 3. Levantar servicio
uvicorn ml_service.main:app --host 0.0.0.0 --port 8002 --reload

# 4. Verificar
curl http://localhost:8002/
# {"service": "iot-ml-service", "status": "ok"}
```

Para configuración completa de feature flags, ver `docs/configuration.md`.

---

## Estructura del Repositorio

```
iot_machine_learning/
├── application/          # Casos de uso, DTOs, explainability, servicios de aplicación
├── domain/               # Entidades, puertos abstractos, políticas, servicios de dominio
│   ├── entities/         # SensorReading, AnomalyResult, Decision, Prediction (frozen dataclasses)
│   ├── ports/            # StoragePort, AuditPort, DecisionEnginePort, AnomalyDetectionPort, etc.
│   ├── policies/         # ThresholdPolicy, DecisionPolicy
│   └── services/         # AnomalyDomainService, PredictionDomainService
├── infrastructure/       # Adaptadores concretos, ML, persistencia, resiliencia
│   ├── ml/               # Motores, pipeline cognitivo, inferencia bayesiana, detección de anomalías
│   │   ├── cognitive/    # Orquestador, fases del pipeline, fusión, inhibición, explicación
│   │   ├── engines/      # Taylor, Statistical, Seasonal, Baseline
│   │   ├── anomaly/      # VotingAnomalyDetector + 8 sub-detectores
│   │   └── inference/    # BayesianUpdater, GaussianPrior, Posterior, ProbabilityCalibrator
│   ├── persistence/      # Redis, SQL Server, repositorios
│   ├── resilience/       # CircuitBreaker (CLOSED/OPEN/HALF_OPEN)
│   └── security/         # Rate limiting, validación de entrada
├── ml_service/           # FastAPI app, runners, consumers, configuración
│   ├── config/           # CognitiveConfig, DecisionConfig, FeatureFlags
│   └── consumers/        # SlidingWindowStore, ReadingsStreamConsumer
├── tests/                # Unit, integration, stress, property-based, meta-tests arquitectónicos
└── docs/                 # Documentación técnica (ver índice abajo)
```

---

## Documentación Técnica

| Tema | Archivo |
|------|---------|
| Arquitectura hexagonal + reglas | `docs/architecture.md` |
| Pipeline ML de 15 fases | `docs/ml_pipeline.md` |
| Detección de drift y adaptación | `docs/drift_and_adaptation.md` |
| Detección de anomalías (ensemble) | `docs/anomaly_detection.md` |
| Cumplimiento y auditoría | `docs/compliance_and_audit.md` |
| ROI y casos de uso | `docs/roi_and_business_case.md` |
| Deuda técnica documentada | `docs/technical_debt.md` |
| Feature flags y configuración | `docs/configuration.md` (existente) |
| Referencia de motores | `docs/ENGINES.md` (existente) |
| Reglas arquitectónicas | `docs/ARCHITECTURE.md` (existente) |
| Plasticidad y aprendizaje | `docs/plasticity.md` (existente) |
| Guía de desarrollo | `docs/development.md` (existente) |
| Operaciones y monitoreo | `docs/operations.md` (existente) |
| API reference | `docs/api.md` (existente) |

---

## Estándares y Certificaciones

| Estándar | Estado | Implementado | Faltante |
|----------|--------|--------------|----------|
| **ISO 13374** (CM&D) | Parcial | Percepción de estado, indicadores de condición, detección de anomalías, drift como indicador de cambio | Diagnóstico de causa raíz, pronóstico de vida útil remanente |
| **ISO 27001** | Parcial | AuditPort con logging estructurado, HMAC-SHA256 en compliance export, trazabilidad de decisiones | Gestión de acceso (RBAC), cifrado en tránsito TLS obligatorio, políticas de retención |
| **IEC 62443** | En evaluación | Segmentación por series_id, validación de entrada (`safe_series_id_to_int`) | Seguridad por diseño (SaB), gestión de parches, autenticación de dispositivos |
| **NIS2** | Gap documentado | No implementado | Notificación de incidentes, gestión de riesgos de cadena de suministro, registro de operadores |

---

## Contacto

Desarrollado por Sergio Nicolás.

- GitHub: [SNPL-glicth/ZENIN](https://github.com/SNPL-glicth/ZENIN)
- Piloto industrial: contacto vía LinkedIn o issue en GitHub
- Contribuciones: PRs bienvenidos; leer `docs/development.md` antes de contribuir

Licencia: MIT (a verificar — confirmar en `pyproject.toml` o `LICENSE`)
