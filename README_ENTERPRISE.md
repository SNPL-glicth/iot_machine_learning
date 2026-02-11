# UTSAE Enterprise — Fase 3

## Arquitectura Hexagonal (Ports & Adapters)

```
┌─────────────────────────────────────────────────────────────┐
│                    ml_service (FastAPI)                      │
│                   Capa de Presentación                       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                     application/                             │
│                   Capa de Aplicación                         │
│                                                              │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────┐  │
│  │ PredictSensor    │ │ DetectAnomalies  │ │ Analyze     │  │
│  │ ValueUseCase     │ │ UseCase          │ │ Patterns    │  │
│  └────────┬─────────┘ └────────┬─────────┘ └──────┬──────┘  │
│           │                    │                   │         │
│  ┌────────▼────────────────────▼───────────────────▼──────┐  │
│  │                    DTOs (prediction_dto.py)             │  │
│  └────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                       domain/                                │
│                   Capa de Dominio                            │
│                  (Lógica de negocio pura)                    │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │ entities/   │  │   ports/     │  │   services/      │    │
│  │             │  │              │  │                   │    │
│  │ Prediction  │  │ Prediction   │  │ PredictionDomain │    │
│  │ Anomaly     │  │ Port         │  │ Service          │    │
│  │ Pattern     │  │ AnomalyPort  │  │ AnomalyDomain   │    │
│  │ SensorRead  │  │ PatternPort  │  │ Service          │    │
│  │             │  │ StoragePort  │  │ PatternDomain    │    │
│  │             │  │ AuditPort    │  │ Service          │    │
│  └─────────────┘  └──────────────┘  └──────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │ (implementa ports)
┌────────────────────────▼────────────────────────────────────┐
│                   infrastructure/                            │
│                 Capa de Infraestructura                      │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐   │
│  │ ml/      │ │ security/│ │ adapters/│ │ ml/           │   │
│  │ engines/ │ │          │ │          │ │ explainability│   │
│  │          │ │ audit_   │ │ predict_ │ │               │   │
│  │ ensemble │ │ logger   │ │ cache    │ │ feature_      │   │
│  │          │ │ access_  │ │ batch_   │ │ importance    │   │
│  │ ml/      │ │ control  │ │ predictor│ │ counterfact.  │   │
│  │ anomaly/ │ │          │ │          │ │               │   │
│  │ voting   │ │          │ │          │ │               │   │
│  │          │ │          │ │          │ │               │   │
│  │ ml/      │ │          │ │          │ │               │   │
│  │ patterns/│ │          │ │          │ │               │   │
│  │ cusum    │ │          │ │          │ │               │   │
│  │ pelt     │ │          │ │          │ │               │   │
│  │ delta_sp │ │          │ │          │ │               │   │
│  │ regime   │ │          │ │          │ │               │   │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Principios

1. **Dependencias hacia adentro:** Infrastructure → Application → Domain
2. **Domain no conoce Infrastructure:** Solo define ports (interfaces ABC)
3. **Inyección de dependencias:** Application recibe adapters vía constructor
4. **Testeable:** Mockear adapters para unit tests (sin BD, sin Redis)

---

## Cumplimiento ISO 27001

### A.12.4.1 — Event Logging

Todos los eventos ML se registran en `FileAuditLogger`:

```json
{
  "timestamp": "2026-02-10T23:24:00+00:00",
  "event_type": "prediction",
  "user_id": "system",
  "action": "predict",
  "resource": "sensor_42",
  "result": "success",
  "details": {
    "predicted_value": 22.5,
    "confidence": 0.85,
    "engine": "taylor",
    "trace_id": "abc123def456"
  },
  "integrity_hash": "a1b2c3d4e5f6g7h8"
}
```

**Eventos auditados:**
- Predicciones generadas
- Anomalías detectadas
- Cambios de configuración (before/after state)
- Acceso a datos de sensores

### A.12.4.3 — Administrator Logs

Cambios de configuración incluyen estado previo y nuevo:

```json
{
  "event_type": "config_change",
  "user_id": "admin_user",
  "before_state": {"value": 2},
  "after_state": {"value": 3}
}
```

### A.12.4.4 — Clock Synchronization

Todos los timestamps en UTC ISO 8601 (`datetime.now(timezone.utc).isoformat()`).

### A.9.2 — User Access Management (RBAC)

| Rol | Permisos |
|-----|----------|
| `viewer` | Lectura de predicciones, anomalías, métricas |
| `operator` | Viewer + ejecución de predicciones + lectura de datos |
| `admin` | Todo: config, modelos, audit logs |
| `auditor` | Lectura de audit logs, métricas, config |

Acceso granular por sensor vía `allowed_sensor_ids`.

### Integridad de Audit Trail

Cada entrada incluye `integrity_hash` (SHA-256 truncado a 16 chars) calculado
sobre el contenido antes de agregar el hash.  Permite detectar tampering.

---

## Detección Avanzada de Patrones

### Delta Spike Classifier

Distingue cambios legítimos (válvula se abre) de ruido (glitch de sensor):

| Criterio | Delta Spike | Noise Spike |
|----------|-------------|-------------|
| Magnitud | > 2σ | > 2σ |
| Persistencia | > 0.6 (se mantiene) | < 0.5 (vuelve al nivel) |
| Trend alignment | Alineado con tendencia | Contra-tendencia |

**Activación:**
```bash
export ML_ENABLE_DELTA_SPIKE_DETECTION=true
```

### Change Point Detection (CUSUM)

Detecta cambios estructurales en la serie temporal:
- **Online:** 1 valor a la vez (para streaming)
- **Batch:** Ventana completa (para batch processing)

Parámetros:
- `threshold`: Sensibilidad (menor = más sensible, default 5.0)
- `drift`: Mínimo cambio a detectar (default 0.5)

**Activación:**
```bash
export ML_ENABLE_CHANGE_POINT_DETECTION=true
```

### Regime Detection

Identifica modos operacionales del sensor (idle, active, peak):
- Usa K-means para segmentar distribución histórica
- Fallback a percentiles si sklearn no está disponible

**Activación:**
```bash
export ML_ENABLE_REGIME_DETECTION=true
```

---

## Detección de Anomalías (Voting Ensemble)

Combina 4 métodos con voting ponderado:

| Método | Peso | Detecta |
|--------|------|---------|
| IsolationForest | 0.40 | Outliers globales |
| Z-score | 0.25 | Desviación estadística |
| LocalOutlierFactor | 0.20 | Outliers locales |
| IQR | 0.15 | Fuera de rango intercuartílico |

**Reduce falsos positivos ~70%** vs método único.

**Activación:**
```bash
export ML_ENABLE_VOTING_ANOMALY=true
export ML_ANOMALY_VOTING_THRESHOLD=0.5
export ML_ANOMALY_CONTAMINATION=0.1
```

---

## Ensemble Predictor

Combina múltiples motores con pesos dinámicos:
- Weighted average de predicciones
- Auto-tuning de pesos cada 10 updates (inverse error weighting)
- Fallback si algún engine falla (fail-open)

**Activación:**
```bash
export ML_ENABLE_ENSEMBLE_PREDICTOR=true
```

---

## Explicabilidad ML

### Feature Importance (Taylor)

Descompone la predicción en contribuciones:
- **valor_actual:** Término constante f(t)
- **velocidad:** Término lineal f'(t)·h
- **aceleración:** Término cuadrático f''(t)·h²/2
- **jerk:** Término cúbico f'''(t)·h³/6

Ejemplo de salida:
```
Partiendo de 22.00, la tendencia aumenta a 0.500/paso, acelerando a 0.050/paso².
```

### Counterfactual Explanations

Responde: "Si X hubiera sido Y, el resultado sería Z":
```json
{
  "scenario": "Si el último valor fuera 24.20 (+2.20)",
  "original_prediction": 22.55,
  "counterfactual_prediction": 24.75,
  "delta": 2.20,
  "sensitivity": 1.0
}
```

**Activación:**
```bash
export ML_ENABLE_EXPLAINABILITY=true
```

---

## Optimización de Performance

### Cache de Predicciones

- **In-memory** con LRU eviction y TTL
- Key: hash(sensor_id + últimos 5 valores + engine_name)
- Invalidación automática si valor cambia > 10%

```bash
export ML_ENABLE_PREDICTION_CACHE=true
export ML_CACHE_TTL_SECONDS=60
export ML_CACHE_MAX_ENTRIES=1000
```

**Métricas esperadas:**
- Hit rate: 40-60%
- Latencia hit: <1ms
- Latencia miss: 0ms overhead

### Batch Processing

- Paralelización con ThreadPoolExecutor
- Circuit breaker (N errores consecutivos → stop)

```bash
export ML_BATCH_MAX_WORKERS=4
export ML_BATCH_CIRCUIT_BREAKER_THRESHOLD=10
```

---

## Feature Flags Completos

```bash
# --- Fase 1 (existentes) ---
export ML_ROLLBACK_TO_BASELINE=false
export ML_USE_TAYLOR_PREDICTOR=false
export ML_USE_KALMAN_FILTER=false
export ML_ENABLE_AB_TESTING=false

# --- Fase 3 Enterprise ---
export ML_ENABLE_DELTA_SPIKE_DETECTION=false
export ML_ENABLE_REGIME_DETECTION=false
export ML_ENABLE_ENSEMBLE_PREDICTOR=false
export ML_ENABLE_AUDIT_LOGGING=false
export ML_ENABLE_PREDICTION_CACHE=false
export ML_ENABLE_VOTING_ANOMALY=false
export ML_ENABLE_CHANGE_POINT_DETECTION=false
export ML_ENABLE_EXPLAINABILITY=false

# --- Config ---
export ML_CACHE_TTL_SECONDS=60
export ML_CACHE_MAX_ENTRIES=1000
export ML_BATCH_MAX_WORKERS=4
export ML_BATCH_CIRCUIT_BREAKER_THRESHOLD=10
export ML_ANOMALY_VOTING_THRESHOLD=0.5
export ML_ANOMALY_CONTAMINATION=0.1
```

**Todos los flags default a `false`** — el sistema se comporta idéntico
al pre-enterprise con todos los flags apagados.

---

## Migración desde Fase 1

1. **Backup** de BD y configuración actual
2. **Activar audit logging** (`ML_ENABLE_AUDIT_LOGGING=true`)
3. **Activar cache** (`ML_ENABLE_PREDICTION_CACHE=true`)
4. **Activar detección de patrones** gradualmente:
   - Primero: `ML_ENABLE_CHANGE_POINT_DETECTION=true`
   - Luego: `ML_ENABLE_DELTA_SPIKE_DETECTION=true`
5. **A/B testing** ensemble vs Taylor solo
6. **Monitorear métricas** 1 semana
7. **Consolidar** configuración final

### Rollback de emergencia

```bash
export ML_ROLLBACK_TO_BASELINE=true
```

Esto fuerza baseline para TODO, ignorando todos los demás flags.

---

## Ejecución de Tests

```bash
# Todos los tests (Fase 1 + Enterprise)
python -m pytest iot_machine_learning/tests/ -v

# Solo tests de dominio
python -m pytest iot_machine_learning/tests/unit/domain/ -v

# Solo tests de infraestructura
python -m pytest iot_machine_learning/tests/unit/infrastructure/ -v

# Solo tests de integración
python -m pytest iot_machine_learning/tests/integration/ -v

# Con cobertura
python -m pytest iot_machine_learning/tests/ --cov=iot_machine_learning -v
```

---

## Estructura de Archivos Enterprise

```
iot_machine_learning/
├── domain/                          # Lógica de negocio pura
│   ├── entities/
│   │   ├── sensor_reading.py       # SensorReading, SensorWindow
│   │   ├── prediction.py           # Prediction, PredictionConfidence
│   │   ├── anomaly.py              # AnomalyResult, AnomalySeverity
│   │   └── pattern.py              # PatternResult, ChangePoint, DeltaSpikeResult
│   ├── ports/
│   │   ├── prediction_port.py      # ABC PredictionPort
│   │   ├── anomaly_detection_port.py
│   │   ├── pattern_detection_port.py
│   │   ├── storage_port.py         # ABC StoragePort
│   │   └── audit_port.py           # ABC AuditPort (ISO 27001)
│   └── services/
│       ├── prediction_domain_service.py
│       ├── anomaly_domain_service.py
│       └── pattern_domain_service.py
├── application/
│   ├── use_cases/
│   │   ├── predict_sensor_value.py
│   │   ├── detect_anomalies.py
│   │   └── analyze_patterns.py
│   └── dto/
│       └── prediction_dto.py
├── infrastructure/
│   ├── ml/
│   │   ├── engines/
│   │   │   └── ensemble_engine.py
│   │   ├── anomaly/
│   │   │   └── voting_anomaly_detector.py
│   │   ├── patterns/
│   │   │   ├── change_point_detector.py  (CUSUM + PELT)
│   │   │   ├── delta_spike_classifier.py
│   │   │   └── regime_detector.py
│   │   └── explainability/
│   │       └── feature_importance.py     (+ counterfactual)
│   ├── security/
│   │   ├── audit_logger.py              (ISO 27001)
│   │   └── access_control.py            (RBAC)
│   └── adapters/
│       ├── prediction_cache.py          (LRU + TTL)
│       └── batch_predictor.py           (paralelo + circuit breaker)
└── tests/
    ├── unit/
    │   ├── domain/
    │   │   └── test_entities.py
    │   └── infrastructure/
    │       ├── test_delta_spike.py
    │       ├── test_change_point.py
    │       ├── test_voting_anomaly.py
    │       ├── test_audit_logger.py
    │       ├── test_access_control.py
    │       ├── test_cache.py
    │       └── test_explainability.py
    └── integration/
        └── test_enterprise_flow.py
```

---

## Troubleshooting

| Problema | Causa | Solución |
|----------|-------|----------|
| Predicciones lentas (>100ms) | Cache desactivado | `ML_ENABLE_PREDICTION_CACHE=true` |
| Muchos falsos positivos | Detector único | `ML_ENABLE_VOTING_ANOMALY=true` |
| Circuit breaker activo | Muchos errores BD | Verificar conexión, resetear CB |
| Audit log no escribe | Permisos de archivo | Verificar permisos del directorio |
| sklearn no disponible | No instalado | Regime/LOF usan fallback automático |
| PELT no disponible | ruptures no instalado | Fallback automático a CUSUM |
