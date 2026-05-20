# ZENIN
### Motor de Decisión Cognitiva para Operaciones Industriales
 
> Pipeline de predicción IoT con Mixture of Experts, fusión bayesiana adaptativa y trazabilidad auditável — diseñado para plantas industriales con sensores heterogéneos.
 
---
 
## ¿Qué problema resuelve?
 
| Dolor | Cómo lo resuelve ZENIN |
|-------|------------------------|
| Máquinas que fallan sin aviso | Predicción continua con 4 engines especializados + detección de drift por sensor |
| Falsos positivos constantes | Hampel filter adaptativo por equipo + InhibitionGate + Ensemble de 8 detectores |
| Decisiones de parada por intuición | Score de riesgo trazable con `ESCALATE / INVESTIGATE / MONITOR / LOG_ONLY` |
| Auditorías que consumen semanas | NDJSON append-only con HMAC-SHA256 por cada predicción |
 
---
 
## Pipeline Cognitivo — 15 Fases
 
```
Sensor (MQTT / HTTP)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  [1] Sanitize      → NaN/Inf → imputación; clamp ±6σ       │
│  [2] BoundaryCheck → validación de rango operativo          │
│  [3] SeasonalDecomp→ FFT / STL (periodo 24h, mín 48pts)    │
│  [4] Perceive      → régimen: STABLE/TRENDING/VOLATILE/NOISY│
│  [5] DriftDetect   → Page-Hinkley (δ=0.005, λ=50)          │
│  [6] Predict       → MoE: Taylor + Statistical + Kalman +   │
│                      Baseline en paralelo (timeout 400ms)   │
│  [7] Adapt         → pesos bayesianos por (série, régimen)  │
│  [8] Inhibit       → suprime engines con error reciente alto │
│  [9] Fuse          → Hampel k·MAD + consenso ponderado      │
│  [10] DecisionArb  → 8 amplificadores + 3 atenuadores       │
│  [11] Coherence    → validación de consistencia temporal     │
│  [12] ConfCalib    → temperatura por régimen                 │
│  [13] Explain      → ReasoningTrace por fase                 │
│  [14] ActionGuard  → guardrail antes de emitir acción       │
│  [15] Narrative    → unificación de explicación final        │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
Predicción + Confianza + Anomalía + Decisión + Audit NDJSON
```
 
---
 
## Mixture of Experts (MoE)
 
El gating no es global — cada tipo de equipo tiene su propia distribución de pesos:
 
```python
# Pasteurizador en régimen stable → Taylor domina
PASTEURIZER / stable  →  taylor: 0.65  kalman: 0.15  statistical: 0.10  baseline: 0.10
 
# Llenadora en cualquier régimen → Taylor excluido estructuralmente
FILLER / stable       →  statistical: 0.70  kalman: 0.20  baseline: 0.10  taylor: 0.00
FILLER / volatile     →  statistical: 0.70  kalman: 0.25  baseline: 0.05  taylor: 0.00
 
# Transportador (vibración) → Kalman domina
CONVEYOR / noisy      →  kalman: 0.75  statistical: 0.15  baseline: 0.05  taylor: 0.05
 
# Silo (nivel) → Taylor domina (detecta tendencias de vaciado)
SILO / trending       →  taylor: 0.75  statistical: 0.10  kalman: 0.10  baseline: 0.05
```
 
**Equipment classes soportados:** `PASTEURIZER · CIP · FILLER · PET_BLOWER · CONVEYOR · SILO · GENERIC`
 
---
 
## Aprendizaje Bayesiano por Sensor
 
Los pesos NO son globales. Cada sensor aprende de forma independiente:
 
```
sensor_01 (pasteurizador) → clave: default:sensor_01:stable → taylor: 0.71
sensor_02 (pasteurizador) → clave: default:sensor_02:stable → taylor: 0.58
```
 
Estrategia de fallback:
 
```
< 50 lecturas  →  cold start: pesos de aptitud por equipment_class (blend gradual)
≥ 50 lecturas  →  pesos per-sensor aprendidos por Bayesian update
sin historial  →  pesos globales por régimen (comportamiento legacy)
```
 
---
 
## Hampel Filter Adaptativo
 
```python
# k universal antes
result = hampel_filter(perceptions, k=3.0)
 
# k por tipo de equipo ahora
# PASTEURIZER → k=2.5, window=10  (señal suave, estricto)
# CIP         → k=5.0, window=5   (ciclos abruptos, permisivo)
# FILLER      → k=4.0, window=20  (periódica, ventana larga)
# CONVEYOR    → k=4.5, window=15  (vibración, permisivo)
result = hampel_filter_with_profile(perceptions, sensor_profile=profile)
 
# Durante eventos industriales activos (STARTUP, CIP_CYCLE):
# k se amplía automáticamente × 1.5
result = hampel_filter_with_profile(perceptions, sensor_profile=profile, event_context=event)
```
 
---
 
## Detección de Eventos Industriales
 
```python
# Eventos detectados automáticamente desde la señal:
IndustrialEvent.STARTUP            # rampa abrupta → arranque de línea
IndustrialEvent.CIP_CYCLE          # ciclo de limpieza activo
IndustrialEvent.PRODUCT_CHANGEOVER # cambio de producto en silo
IndustrialEvent.FAULT_TRANSIENT    # >30% de valores clampeados
IndustrialEvent.SHUTDOWN           # caída de señal
 
# Efecto en el pipeline:
# → InhibitPhase: supresión reducida al 50% (PostEventStabilizationGate)
# → HampelFilter: k × 1.5 (más permisivo durante transitorio)
# → Régimen: reclasificado con contexto temporal (hour_of_day)
```
 
---
 
## Auditoría y Compliance
 
Cada predicción genera un registro NDJSON firmado:
 
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
  "hmac": "sha256:a3f9c2..."   // verificación independiente vía verify_record()
}
```
 
---
 
## Capacidades Verificadas en Código
 
| Capacidad | Detalle |
|-----------|---------|
| **Pipeline 15 fases** | Orden fijo, fases desacoplables por flags, early termination en NaN/budget |
| **MoE equipment-aware** | Gating por `(equipment_class, regime)` — 6 equipos × 4 regímenes |
| **Bayesian per-sensor** | Clave `namespace:series_id:regime`, fallback a global, cold start blend |
| **Hampel adaptativo** | `k` y `window` desde `SensorProfile`, amplificación × 1.5 en eventos |
| **Drift detection** | Page-Hinkley online; reset de régimen afectado, no del sistema completo |
| **Ensemble 8 detectores** | Z-score, IQR, IsolationForest, LOF, velocity\_z, acceleration\_z + 2 ND |
| **Compliance HMAC** | NDJSON append-only, SHA-256, verificación constant-time, fsync atómico |
| **Circuit breaker** | CLOSED/OPEN/HALF\_OPEN, backoff exponencial, decorador `@protected` |
| **Rate limiter** | Protección DoS por serie, configurable |
| **Rollout progresivo** | `FeatureActivator` con whitelist `ML_BATCH_ENTERPRISE_SENSORS` |
 
---
 
## Capacidades en Desarrollo
 
| Capacidad | Estado |
|-----------|--------|
| `TextCognitiveEngine` | Verificado en código, no integrado en pipeline numérico |
| `HybridNeuralEngine` | Verificado en código, no integrado en pipeline numérico |
| `CognitiveMemory / Weaviate` | `ML_ENABLE_COGNITIVE_MEMORY=false` por defecto |
| `SNNLayer con STDP` | Documentado, no encontrado en pipeline activo |
| Benchmarks NAB/Yahoo S5 | Pendiente |
| Escalabilidad >1000 sensores | Verificado — tests de carga y estrés |
 
---
 
## Stack
 
```
API          →  FastAPI · Uvicorn · Python 3.10+
ML           →  NumPy · scikit-learn · SciPy · XGBoost
Estado       →  Redis (streams, sliding windows, plasticity)
Persistencia →  SQL Server · NDJSON append-only
Arquitectura →  Hexagonal (Ports & Adapters)
Deploy       →  Docker · docker-compose
```
 
---
 
## Inicio Rápido
 
```bash
# 1. Infraestructura
docker run -d --name redis-zenin -p 6379:6379 redis:7-alpine
docker run -d --name sql-zenin -p 1433:1433 \
  -e SA_PASSWORD=YourPassword123 \
  -e ACCEPT_EULA=Y \
  mcr.microsoft.com/mssql/server:2022-latest
 
# 2. Configuración
cp .env.example .env
# Editar .env con credenciales reales
 
# 3. Levantar
uvicorn ml_service.main:app --host 0.0.0.0 --port 8002 --reload
 
# 4. Verificar
curl http://localhost:8002/
# {"service": "iot-ml-service", "status": "ok"}
 
# 5. Predicción de ejemplo
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "series_id": "PAST-01",
    "values": [71.2, 71.4, 71.3, 71.5, 71.4],
    "timestamps": [1700000000, 1700000300, 1700000600, 1700000900, 1700001200]
  }'
```
 
---
 
## Comparación de Mercado
 
| Capacidad | ZENIN | AWS Lookout | Azure AD | Palantir |
|-----------|:-----:|:-----------:|:--------:|:--------:|
| Pesos bayesianos online (sin retraining) | ✅ | ❌ | ❌ | ❌ |
| Equipment awareness por tipo de equipo | ✅ | ❌ | ❌ | ❌ |
| MoE routing contextual | ✅ | ❌ | ❌ | ❌ |
| Decisión con guardrails auditables | ✅ | ❌ | ⚠️ | ✅ |
| Explicación trazable por fase | ✅ | ⚠️ | ⚠️ | ✅ |
| Compliance HMAC-SHA256 nativo | ✅ | ❌ | ❌ | ❌ |
| Deploy on-premise sin cloud | ✅ | ❌ | ❌ | ⚠️ |
| velocity\_z + acceleration\_z | ✅ | ❌ | ❌ | ❌ |
| Escalabilidad >1000 sensores | ✅ | ✅ | ✅ | ✅ |
| Benchmarks públicos validados | ❌ | ✅ | ✅ | ✅ |
 
---
 
## Estructura del Repositorio
 
```
iot_machine_learning/
├── domain/                    # Entidades, puertos, políticas — zero infra deps
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
└── tests/                     # Unit, integration, meta-tests arquitectónicos
```
 
---
 
## Documentación
 
| Tema | Archivo |
|------|---------|
| Arquitectura hexagonal | `docs/architecture.md` |
| Pipeline 15 fases | `docs/ml_pipeline.md` |
| Detección de anomalías | `docs/anomaly_detection.md` |
| Drift y adaptación | `docs/drift_and_adaptation.md` |
| Compliance y auditoría | `docs/compliance_and_audit.md` |
| Feature flags | `docs/configuration.md` |
| ROI y casos de uso | `docs/roi_and_business_case.md` |
 
---
 
## Contacto
 
**Sergio Nicolás** — [GitHub](https://github.com/SNPL-glicth) · LinkedIn
 
Piloto industrial disponible. PRs bienvenidos — leer `docs/development.md` antes de contribuir.
 
---
 
> Licencia: Propietaria — Todos los derechos reservados. Consulte el archivo LICENSE para términos comerciales.

