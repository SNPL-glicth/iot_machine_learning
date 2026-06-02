<!-- ZENIN ML BANNER -->
<p align="center">
  <img src="https://img.shields.io/badge/ZENIN%20ML-Cognitive%20Engine-00D4C8?style=for-the-badge&logo=brain&logoColor=white" alt="ZENIN ML"/>
</p>

<p align="center">
  <a href="#pipeline"><img src="https://img.shields.io/badge/15%20Fases%20ML-00D4C8?style=flat-square"/></a>
  <a href="#moe"><img src="https://img.shields.io/badge/MoE%20Bayesiano-00D4C8?style=flat-square"/></a>
  <a href="#audit"><img src="https://img.shields.io/badge/HMAC--SHA256-00D4C8?style=flat-square"/></a>
  <a href="#anomaly"><img src="https://img.shields.io/badge/8%20Detectores-00D4C8?style=flat-square"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/SQL%20Server-CC2927?style=flat-square&logo=microsoft-sql-server&logoColor=white"/>
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

<p align="center"><strong>Motor de Decision Cognitiva para Operaciones Industriales.</strong></p>
<p align="center">Pipeline de prediccion IoT con Mixture of Experts, fusion bayesiana adaptativa y trazabilidad auditavel.</p>

---

## Vision

ZENIN ML es el nucleo cognitivo de la plataforma ZENIN. Ingiere series temporales de cualquier sensor, ejecuta un <strong>pipeline de 15 fases especializadas</strong>, y entrega al operador una decision clara con confianza calibrada:

<p align="center">
  <img src="https://img.shields.io/badge/ESCALATE-FF3D3D?style=for-the-badge&logo=alert-circle&logoColor=white"/>
  <img src="https://img.shields.io/badge/INVESTIGATE-FFB300?style=for-the-badge&logo=alert-triangle&logoColor=white"/>
  <img src="https://img.shields.io/badge/MONITOR-00E676?style=for-the-badge&logo=activity&logoColor=white"/>
  <img src="https://img.shields.io/badge/LOG__ONLY-4A7A8A?style=for-the-badge&logo=file-text&logoColor=white"/>
</p>

Cada prediccion queda registrada con firma <strong>HMAC-SHA256</strong> en NDJSON append-only para auditoria forense.

---

## <a name="pipeline"></a> Pipeline Cognitivo — 15 Fases

```
Sensor (MQTT / HTTP)
      |
      ▼
┌─────────────────────────────────────────────────────────────┐
|  [1] Sanitize      → NaN/Inf → imputacion; clamp ±6sigma   |
|  [2] BoundaryCheck → validacion de rango operativo          |
|  [3] SeasonalDecomp→ FFT / STL (periodo 24h, min 48pts)    |
|  [4] Perceive      → regimen: STABLE/TRENDING/VOLATILE/NOISY|
|  [5] DriftDetect   → Page-Hinkley (delta=0.005, lambda=50) |
|  [6] Predict       → MoE: Taylor + Statistical + Kalman +  |
|                      Baseline en paralelo (timeout 400ms)  |
|  [7] Adapt         → pesos bayesianos por (serie, regimen)  |
|  [8] Inhibit       → suprime engines con error reciente alto |
|  [9] Fuse          → Hampel k*MAD + consenso ponderado     |
|  [10] DecisionArb  → 8 amplificadores + 3 atenuadores      |
|  [11] Coherence    → validacion de consistencia temporal     |
|  [12] ConfCalib    → temperatura por regimen                |
|  [13] Explain      → ReasoningTrace por fase               |
|  [14] ActionGuard  → guardrail antes de emitir accion      |
|  [15] Narrative    → unificacion de explicacion final       |
└─────────────────────────────────────────────────────────────┘
      |
      ▼
Prediccion + Confianza + Anomalia + Decision + Audit NDJSON
```

---

## <a name="moe"></a> Mixture of Experts (MoE)

El gating no es global — cada tipo de equipo tiene su propia distribucion de pesos:

```python
# Pasteurizador en regimen stable -> Taylor domina
PASTEURIZER / stable  ->  taylor: 0.65  kalman: 0.15  statistical: 0.10  baseline: 0.10

# Llenadora en cualquier regimen -> Taylor excluido estructuralmente
FILLER / stable       ->  statistical: 0.70  kalman: 0.20  baseline: 0.10  taylor: 0.00
FILLER / volatile     ->  statistical: 0.70  kalman: 0.25  baseline: 0.05  taylor: 0.00

# Transportador (vibracion) -> Kalman domina
CONVEYOR / noisy      ->  kalman: 0.75  statistical: 0.15  baseline: 0.05  taylor: 0.05

# Silo (nivel) -> Taylor domina (detecta tendencias de vaciado)
SILO / trending       ->  taylor: 0.75  statistical: 0.10  kalman: 0.10  baseline: 0.05
```

**Equipment classes soportados:** `PASTEURIZER · CIP · FILLER · PET_BLOWER · CONVEYOR · SILO · GENERIC`

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

# k por tipo de equipo ahora
# PASTEURIZER -> k=2.5, window=10  (senal suave, estricto)
# CIP         -> k=5.0, window=5   (ciclos abruptos, permisivo)
# FILLER      -> k=4.0, window=20  (periodica, ventana larga)
# CONVEYOR    -> k=4.5, window=15  (vibracion, permisivo)
result = hampel_filter_with_profile(perceptions, sensor_profile=profile)

# Durante eventos industriales activos (STARTUP, CIP_CYCLE):
# k se amplia automaticamente x 1.5
result = hampel_filter_with_profile(perceptions, sensor_profile=profile, event_context=event)
```

---

## Deteccion de Eventos Industriales

```python
# Eventos detectados automaticamente desde la senal:
IndustrialEvent.STARTUP            # rampa abrupta -> arranque de linea
IndustrialEvent.CIP_CYCLE          # ciclo de limpieza activo
IndustrialEvent.PRODUCT_CHANGEOVER # cambio de producto en silo
IndustrialEvent.FAULT_TRANSIENT    # >30% de valores clampeados
IndustrialEvent.SHUTDOWN           # caida de senal

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
| **Pipeline 15 fases** | Orden fijo, fases desacoplables por flags, early termination en NaN/budget |
| **MoE equipment-aware** | Gating por `(equipment_class, regimen)` — 6 equipos x 4 regimenes |
| **Bayesian per-sensor** | Clave `namespace:series_id:regimen`, fallback a global, cold start blend |
| **Hampel adaptativo** | `k` y `window` desde `SensorProfile`, amplificacion x 1.5 en eventos |
| **Drift detection** | Page-Hinkley online; reset de regimen afectado, no del sistema completo |
| **Ensemble 8 detectores** | Z-score, IQR, IsolationForest, LOF, velocity_z, acceleration_z + 2 ND |
| **Compliance HMAC** | NDJSON append-only, SHA-256, verificacion constant-time, fsync atomico |
| **Circuit breaker** | CLOSED/OPEN/HALF_OPEN, backoff exponencial, decorador `@protected` |
| **Rate limiter** | Proteccion DoS por serie, configurable |
| **Rollout progresivo** | `FeatureActivator` con whitelist `ML_BATCH_ENTERPRISE_SENSORS` |

---

## Capacidades en Desarrollo

| Capacidad | Estado |
|-----------|--------|
| `TextCognitiveEngine` | Verificado en codigo, no integrado en pipeline numerico |
| `HybridNeuralEngine` | Verificado en codigo, no integrado en pipeline numerico |
| `CognitiveMemory / Weaviate` | `ML_ENABLE_COGNITIVE_MEMORY=false` por defecto |
| `SNNLayer con STDP` | Documentado, no encontrado en pipeline activo |
| Benchmarks NAB/Yahoo S5 | Pendiente |
| Escalabilidad >1000 sensores | Verificado — tests de carga y estres |

---

## Stack

```
API          ->  FastAPI · Uvicorn · Python 3.10+
ML           ->  NumPy · scikit-learn · SciPy · XGBoost
Estado       ->  Redis (streams, sliding windows, plasticity)
Persistencia ->  SQL Server · NDJSON append-only
Arquitectura ->  Hexagonal (Ports & Adapters)
Deploy       ->  Docker · docker-compose
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
├── domain/                    # Entidades, puertos, politicas — zero infra deps
│   ├── entities/              # Value objects frozen (Prediction, Anomaly, Explanation...)
│   ├── ports/                 # StoragePort, AuditPort, DecisionEnginePort...
│   └── value_objects/         # EquipmentClass, IndustrialEvent, SensorProfile
├── infrastructure/
│   ├── ml/
│   │   ├── cognitive/         # Pipeline 15 fases + BayesianWeightTracker + Hampel
│   │   │   ├── orchestration/ # MetaCognitiveOrchestrator + phases/
│   │   │   ├── bayesian_weight_tracker/  # 28 archivos — per-sensor learning
│   │   │   ├── fusion/        # WeightedFusion, HampelFilter adaptativo
│   │   │   └── inhibition/    # InhibitionGate + EngineReliabilityTracker
│   │   ├── engines/           # Taylor, Statistical, Baseline, Kalman
│   │   ├── anomaly/           # VotingAnomalyDetector + 8 sub-detectores
│   │   └── moe/               # MoE: registry, gating, experts, events, drift
│   ├── repositories/          # SqlSensorProfileRepository (equipment-aware)
│   ├── persistence/           # Redis, SQL Server
│   └── resilience/            # CircuitBreaker, RateLimiter
├── ml_service/                # FastAPI app, feature flags, governance
├── core/                      # Ensemble, drift, statistical, tuning
└── tests/                     # Unit, integration, meta-tests arquitectonicos
```

---

## Documentacion

| Tema | Archivo |
|------|---------|
| Arquitectura hexagonal | `docs/architecture.md` |
| Pipeline 15 fases | `docs/ml_pipeline.md` |
| Deteccion de anomalias | `docs/anomaly_detection.md` |
| Drift y adaptacion | `docs/drift_and_adaptation.md` |
| Compliance y auditoria | `docs/compliance_and_audit.md` |
| Feature flags | `docs/configuration.md` |
| ROI y casos de uso | `docs/roi_and_business_case.md` |

---

## Metricas

| Metrica | Valor |
|---------|-------|
| Fases del pipeline | 15 |
| Detectores de anomalias | 8 |
| Experts MoE | 4 |
| Regimenes operativos | 4 (STABLE, TRENDING, VOLATILE, NOISY) |
| Equipment classes | 7 |
| Decision levels | 4 (ESCALATE, INVESTIGATE, MONITOR, LOG_ONLY) |
| Tests automatizados | 3,600+ |
| Arquitectura | Hexagonal (Ports & Adapters) |

---

> Licencia: Propietaria — Todos los derechos reservados.
