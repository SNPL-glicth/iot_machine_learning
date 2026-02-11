# AUDITORÍA TÉCNICA COMPLETA — iot_machine_learning

**Fecha:** 2026-02-10 (actualizado: 2026-02-10 19:56 UTC-5)
**Enfoque:** UTSAE + Arquitectura Hexagonal
**Archivos analizados:** 35 módulos Python (código real, no intenciones)
**Estado de refactorización:** 4/6 tareas completadas — 368 tests pasando

---

## A) MATRIZ DE RESPONSABILIDAD POR ARCHIVO

### Leyenda de columnas
- **P** = Percibe señal (I/O de datos crudos)
- **M** = Interpreta matemáticamente (cálculos, estadísticas)
- **D** = Toma decisiones (if/else de negocio, clasificación)
- **O** = Orquesta flujo (coordina múltiples pasos)
- **I** = Expone infraestructura (BD, archivos, threads, sklearn)
- **E** = Genera explicación humana (texto legible, narrativa)
- **#R** = Número de responsabilidades (>1 = candidato a refactor)

### ml/core/ (Fase 1 Legacy)

| Archivo | P | M | D | O | I | E | #R | Veredicto |
|---------|---|---|---|---|---|---|----|----|
| `interfaces.py` | | | | | | | **1** | ✅ LIMPIO — Solo ABCs + value object (`PredictionResult`) |
| `validators.py` | | ✓ | | | | | **1** | ✅ LIMPIO — Validación numérica pura (NaN, Inf, clamp) |
| `taylor_predictor.py` | | ✓ | ✓ | | | | **2** | ⚠️ MENOR — Cálculo de derivadas + decisión de trend + decisión de confianza. Aceptable: las decisiones son inherentes al modelo matemático (trend es f'(t) > threshold, confianza es varianza de f''). No es lógica de negocio. |
| `kalman_filter.py` | | ✓ | | | ✓ | | **2** | ⚠️ MENOR — Cálculo Kalman + threading.Lock (infra). El lock es necesario para thread-safety, no es lógica de infra separable. |
| `engine_factory.py` | | | ✓ | ✓ | ✓ | | **3** | 🔴 **CANDIDATO** — Factory + decisión por feature flags + import lazy de `ml.baseline` + import de `FeatureFlags` desde `ml_service`. Mezcla selección de motor (decisión) con creación (infra) y consulta de config (orquestación). |

### domain/ (Enterprise)

| Archivo | P | M | D | O | I | E | #R | Veredicto |
|---------|---|---|---|---|---|---|----|----|
| `entities/sensor_reading.py` | | | | | | | **1** | ✅ LIMPIO — Value objects puros, frozen dataclasses |
| `entities/prediction.py` | | | | | | | **1** | ✅ LIMPIO — Value object + `to_audit_dict()` (serialización, no I/O) |
| `entities/anomaly.py` | | | ✓ | | | | **1** | ✅ LIMPIO — `AnomalySeverity.from_score()` es clasificación de dominio puro |
| `entities/pattern.py` | | | | | | | **1** | ✅ LIMPIO — Enums + value objects puros |
| `ports/prediction_port.py` | | | | | | | **1** | ✅ LIMPIO — ABC puro, sin implementación |
| `ports/anomaly_detection_port.py` | | | | | | | **1** | ✅ LIMPIO — ABC puro |
| `ports/pattern_detection_port.py` | | | | | | | **1** | ✅ LIMPIO — 4 ABCs relacionados en un archivo. Aceptable por cohesión temática. |
| `ports/storage_port.py` | | | | | | | **1** | ✅ LIMPIO — ABC puro |
| `ports/audit_port.py` | | | | | | | **1** | ✅ LIMPIO — ABC puro |
| `services/prediction_domain_service.py` | | | ✓ | ✓ | | | **2** | ✅ ACEPTABLE — Orquesta selección de engine + genera predicción. La decisión (seleccionar engine) es lógica de dominio legítima. No mezcla I/O. |
| `services/anomaly_domain_service.py` | | ✓ | ✓ | ✓ | | | **3** | ⚠️ MENOR — Orquesta detectores + calcula promedio de votos (math) + decide is_anomaly (decisión) + calcula confianza por varianza (math). Las matemáticas son inherentes al voting — extraerlas sería over-engineering. |
| `services/pattern_domain_service.py` | | | | ✓ | | | **1** | ✅ LIMPIO — Solo orquesta delegando a ports opcionales |

### infrastructure/ (Enterprise)

| Archivo | P | M | D | O | I | E | #R | Veredicto |
|---------|---|---|---|---|---|---|----|----|
| `ml/engines/ensemble_engine.py` | | ✓ | ✓ | ✓ | | | **3** | ⚠️ MENOR — Weighted average (math) + trend majority vote (decisión) + orquesta N engines. Inherente a un ensemble — no separable sin fragmentación artificial. |
| `ml/engines/taylor_adapter.py` | | | | | ✓ | | **1** | ✅ LIMPIO — Adapter puro (convierte interfaces) |
| `ml/anomaly/voting_anomaly_detector.py` | | | | ✓ | ✓ | | **2** | ✅ **REFACTORIZADO (T3)** — Math extraído a `statistical_methods.py`, narrativa a `anomaly_narrator.py`. Solo orquesta sub-detectores + sklearn wrappers. |
| `ml/patterns/delta_spike_classifier.py` | | ✓ | ✓ | | | ✓ | **3** | ⚠️ MENOR — Calcula persistencia/magnitud (math) + clasifica spike (decisión) + genera explicación. Las 3 son inseparables en un clasificador. |
| `ml/patterns/change_point_detector.py` | | ✓ | ✓ | | ✓ | | **3** | ⚠️ MENOR — CUSUM math + decisión de change point + import opcional de `ruptures` (infra). El import condicional es pragmático. |
| `ml/patterns/regime_detector.py` | | ✓ | ✓ | | ✓ | | **3** | ⚠️ MENOR — K-means (infra sklearn) + percentile fallback (math) + asigna régimen (decisión). Similar a change_point. |
| `ml/explainability/feature_importance.py` | | ✓ | | | | ✓ | **2** | ✅ ACEPTABLE — Descomposición Taylor (math) + genera texto legible (narrativa). Cohesión alta: la narrativa es el propósito del módulo. |
| `security/audit_logger.py` | | | | | ✓ | | **1** | ✅ LIMPIO — Solo I/O (escribir JSON a archivo) |
| `security/access_control.py` | | | ✓ | | | | **1** | ✅ LIMPIO — Solo decisión de acceso (RBAC puro) |
| `adapters/prediction_cache.py` | | | | | ✓ | | **1** | ✅ LIMPIO — Solo infra (cache LRU + TTL) |
| `adapters/batch_predictor.py` | | | | | ✓ | | **2** | ✅ ACEPTABLE — Orquesta ThreadPool (infra) + circuit breaker (decisión simple). Inseparable. |

### application/ (Enterprise)

| Archivo | P | M | D | O | I | E | #R | Veredicto |
|---------|---|---|---|---|---|---|----|----|
| `use_cases/predict_sensor_value.py` | | | | ✓ | | | **1** | ✅ LIMPIO — Solo orquesta: load → predict → persist → DTO |
| `use_cases/detect_anomalies.py` | | | | ✓ | | | **1** | ✅ LIMPIO — Solo orquesta |
| `use_cases/analyze_patterns.py` | | | | ✓ | | | **1** | ✅ LIMPIO — Solo orquesta |
| `dto/prediction_dto.py` | | | | | | | **1** | ✅ LIMPIO — Value objects planos |

### ml_service/ (Presentación/Runtime)

| Archivo | P | M | D | O | I | E | #R | Veredicto |
|---------|---|---|---|---|---|---|----|----|
| `api/services/prediction_service.py` | | | | ✓ | | | **1** | ✅ **REFACTORIZADO (T1)** — Thin orchestrator delegando a `PredictSensorValueUseCase`, `SqlServerStorageAdapter`, `BaselinePredictionAdapter`, `ThresholdEvaluator`. |
| `runners/common/sensor_processor.py` | | | | ✓ | | | **1** | ✅ **REFACTORIZADO (T2)** — Thin orchestrator delegando a `RegressionPredictionService` (Modeling) y `PredictionNarrator` (Narrative). |
| `runners/common/severity_classifier.py` | ✓ | | | | ✓ | | **2** | ✅ **REFACTORIZADO (T4)** — Solo I/O (SQL queries). Reglas extraídas a `domain/services/severity_rules.py`, rangos a `domain/entities/sensor_ranges.py`. |
| `orchestrator/prediction_orchestrator.py` | ✓ | | | ✓ | ✓ | | **3** | ⚠️ ACEPTABLE — Orquesta 8 módulos de contexto + lee BD (percepción) + persiste decisión (infra). Es un orquestador legítimo, pero la BD directa viola hexagonal. |
| `runners/ml_stream_runner.py` | ✓ | | | ✓ | ✓ | | **3** | ⚠️ ACEPTABLE — Consume broker (percepción) + orquesta análisis + persiste eventos (infra). Orquestador de nivel alto. |
| `ml/baseline.py` | | ✓ | | | | | **1** | ✅ LIMPIO — Cálculo puro de media móvil |
| `ml/pattern_detector.py` | | ✓ | ✓ | ✓ | | ✓ | **4** | 🔴 **CANDIDATO** — Calcula estadísticas (math) + clasifica patrones (decisión) + orquesta análisis multi-patrón + genera descripciones (narrativa). 606 líneas. |

---

## B) CLASIFICACIÓN UTSAE

### Mapa de Módulos por Capa UTSAE

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SENSING / PERCEPTION                             │
│  (Adquisición de datos crudos del mundo exterior)                   │
│                                                                     │
│  ml_service/repository/sensor_repository.py  ← Lee dbo.sensor_readings
│  ml_service/reading_broker.py                ← Consume Redis/stream  │
│  ml_service/sliding_window_buffer.py         ← Mantiene ventanas    │
│  domain/entities/sensor_reading.py           ← Value objects        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    MODELING / INTERPRETATION                         │
│  (Interpretación matemática de la señal)                            │
│                                                                     │
│  ml/core/taylor_predictor.py        ← Series de Taylor              │
│  ml/core/kalman_filter.py           ← Filtro de Kalman              │
│  ml/baseline.py                     ← Media móvil                   │
│  ml/core/validators.py              ← Validación numérica           │
│  infra/ml/engines/ensemble_engine.py ← Weighted average             │
│  infra/ml/engines/taylor_adapter.py  ← Bridge Fase1↔Enterprise     │
│  infra/ml/patterns/change_point_detector.py ← CUSUM/PELT           │
│  infra/ml/patterns/regime_detector.py       ← K-means              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    REASONING / DECISION                              │
│  (Decisiones de negocio sobre la interpretación)                    │
│                                                                     │
│  domain/services/prediction_domain_service.py  ← Selección engine   │
│  domain/services/anomaly_domain_service.py     ← Voting + threshold │
│  domain/services/pattern_domain_service.py     ← Delegación         │
│  infra/ml/anomaly/voting_anomaly_detector.py   ← Voting ensemble    │
│  infra/ml/patterns/delta_spike_classifier.py   ← Delta vs noise     │
│  infra/security/access_control.py              ← RBAC               │
│  ml_service/runners/common/severity_classifier.py ← Severidad ⚠️    │
│  domain/entities/anomaly.py (AnomalySeverity.from_score)            │
│  domain/entities/prediction.py (PredictionConfidence.from_score)    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    NARRATIVE / EXPLANATION                           │
│  (Generación de explicaciones legibles para humanos)                │
│                                                                     │
│  infra/ml/explainability/feature_importance.py  ← Taylor decomp     │
│  infra/ml/explainability/feature_importance.py  ← Counterfactuals   │
│  ml_service/explain/contextual_explainer.py     ← Explicación AI    │
│  ml_service/runners/common/sensor_processor.py._build_explanation ⚠️│
│  ml/pattern_detector.py (descripciones de patrones) ⚠️              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    ADAPTATION / INFRASTRUCTURE                       │
│  (Conexión con el mundo exterior: BD, archivos, threads)            │
│                                                                     │
│  infra/security/audit_logger.py         ← Archivo JSON Lines        │
│  infra/adapters/prediction_cache.py     ← Cache LRU in-memory       │
│  infra/adapters/batch_predictor.py      ← ThreadPool + CB           │
│  ml_service/api/services/prediction_service.py ← BD directa ⚠️     │
│  ml_service/repository/*                ← BD queries                │
│  ml_service/runners/common/event_writer.py ← BD writes              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION / APPLICATION                       │
│  (Coordinación de flujos end-to-end)                                │
│                                                                     │
│  application/use_cases/predict_sensor_value.py  ← Enterprise ✅     │
│  application/use_cases/detect_anomalies.py      ← Enterprise ✅     │
│  application/use_cases/analyze_patterns.py      ← Enterprise ✅     │
│  ml_service/orchestrator/prediction_orchestrator.py ← Legacy        │
│  ml_service/runners/common/sensor_processor.py      ← Legacy ⚠️    │
│  ml_service/runners/ml_stream_runner.py             ← Legacy        │
│  ml_service/runners/ml_batch_runner.py              ← Legacy        │
│  ml/core/engine_factory.py                          ← Legacy ⚠️    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## C) MÉTRICAS OBJETIVAS

| Archivo | Líneas | Imports ext. | Math | Reglas neg. | I/O | Orquestación | #Resp |
|---------|--------|-------------|------|-------------|-----|-------------|-------|
| `ml/core/interfaces.py` | 176 | 0 | ✗ | ✗ | ✗ | ✗ | 1 |
| `ml/core/validators.py` | 139 | 0 | ✓ | ✗ | ✗ | ✗ | 1 |
| `ml/core/taylor_predictor.py` | 440 | 0 | ✓ | ✗ | ✗ | ✗ | 1 |
| `ml/core/kalman_filter.py` | 405 | 0 | ✓ | ✗ | ✗ | ✗ | 1 |
| `ml/core/engine_factory.py` | 268 | 1* | ✗ | ✓ | ✗ | ✓ | 3 |
| `domain/entities/*.py` | 411 | 0 | ✗ | ✗ | ✗ | ✗ | 1 |
| `domain/ports/*.py` | 456 | 0 | ✗ | ✗ | ✗ | ✗ | 1 |
| `domain/services/prediction_domain_service.py` | 182 | 0 | ✗ | ✓ | ✗ | ✓ | 2 |
| `domain/services/anomaly_domain_service.py` | 174 | 0 | ✓ | ✓ | ✗ | ✓ | 3 |
| `domain/services/pattern_domain_service.py` | 205 | 0 | ✗ | ✗ | ✗ | ✓ | 1 |
| `infra/ml/engines/ensemble_engine.py` | 291 | 0 | ✓ | ✓ | ✗ | ✓ | 3 |
| `infra/ml/anomaly/voting_anomaly_detector.py` | 286 | 2** | ✗ | ✗ | ✓ | ✗ | 2 |
| `infra/ml/patterns/delta_spike_classifier.py` | 142 | 0 | ✓ | ✓ | ✗ | ✗ | 2 |
| `infra/ml/patterns/change_point_detector.py` | 171 | 1*** | ✓ | ✓ | ✗ | ✗ | 2 |
| `infra/ml/patterns/regime_detector.py` | 146 | 1** | ✓ | ✓ | ✓ | ✗ | 3 |
| `infra/ml/explainability/feature_importance.py` | 270 | 0 | ✓ | ✗ | ✗ | ✗ | 2 |
| `infra/security/audit_logger.py` | 238 | 0 | ✗ | ✗ | ✓ | ✗ | 1 |
| `infra/security/access_control.py` | 279 | 0 | ✗ | ✓ | ✗ | ✗ | 1 |
| `infra/adapters/prediction_cache.py` | 219 | 0 | ✗ | ✗ | ✓ | ✗ | 1 |
| `infra/adapters/batch_predictor.py` | 189 | 0 | ✗ | ✗ | ✓ | ✓ | 2 |
| `application/use_cases/*.py` | 376 | 0 | ✗ | ✗ | ✗ | ✓ | 1 |
| `application/dto/prediction_dto.py` | 103 | 0 | ✗ | ✗ | ✗ | ✗ | 1 |
| `ml_service/api/services/prediction_service.py` | 196 | 0 | ✗ | ✗ | ✗ | ✓ | 1 |
| `ml_service/runners/common/sensor_processor.py` | 193 | 0 | ✗ | ✗ | ✗ | ✓ | 1 |
| `ml_service/runners/common/severity_classifier.py` | 161 | 1 | ✗ | ✗ | ✓ | ✗ | 2 |
| `ml_service/orchestrator/prediction_orchestrator.py` | 330 | 1 | ✗ | ✗ | ✓ | ✓ | 3 |
| `ml/pattern_detector.py` | 606 | 0 | ✓ | ✓ | ✗ | ✓ | 4 |
| `ml/baseline.py` | 40 | 0 | ✓ | ✗ | ✗ | ✗ | 1 |

\* `engine_factory.py` importa `FeatureFlags` de `ml_service` (violación de dirección)
\** sklearn (IsolationForest, LOF, KMeans)
\*** ruptures (opcional)

---

## D) DETECCIÓN DE MALOS OLORES

### D.1 — God Objects (archivos con ≥5 responsabilidades)

**✅ `ml_service/api/services/prediction_service.py` — REFACTORIZADO (TAREA 1)**

> **Estado:** Thin orchestrator (196 líneas, antes 368). Delega a:
> - `PredictSensorValueUseCase` (application layer)
> - `SqlServerStorageAdapter` (infra — nuevo)
> - `BaselinePredictionAdapter` (infra — nuevo)
> - `ThresholdEvaluator` (domain — nuevo)
> - `ThresholdRepository` (infra — nuevo)
>
> **Tests nuevos:** `test_threshold_evaluator.py`, `test_baseline_adapter.py`

**✅ `ml_service/runners/common/sensor_processor.py` — REFACTORIZADO (TAREA 2)**

> **Estado:** Thin orchestrator (193 líneas, antes 319). Delega a:
> - `RegressionPredictionService` (Modeling — nuevo)
> - `PredictionNarrator` (Narrative — nuevo)
> - `ModelManager`, `PredictionWriter`, `EventWriter` (Infra — ya existían)
>
> **Tests nuevos:** `test_regression_prediction_service.py` (13), `test_prediction_narrator.py` (14)

**✅ `infra/ml/anomaly/voting_anomaly_detector.py` — REFACTORIZADO (TAREA 3)**

> **Estado:** Orchestrator (286 líneas, antes 289 pero ahora 2 resp. en vez de 5). Delega a:
> - `statistical_methods.py` (Modeling puro — `compute_z_score`, `compute_z_vote`, `compute_iqr_vote`, `weighted_vote`, `compute_consensus_confidence`, `compute_training_stats`)
> - `anomaly_narrator.py` (Narrative puro — `build_anomaly_explanation`)
>
> **Tests nuevos:** `test_statistical_methods.py` (31), `test_anomaly_narrator.py` (11)
> **Fix adicional:** Test flaky `test_extreme_value_is_anomaly` corregido (pesos que no dependían de sklearn).

**✅ `ml_service/runners/common/severity_classifier.py` — REFACTORIZADO (TAREA 4)**

> **Estado:** Solo I/O (161 líneas, antes 254). Reglas extraídas a:
> - `domain/entities/sensor_ranges.py` (`DEFAULT_SENSOR_RANGES`)
> - `domain/services/severity_rules.py` (5 funciones puras + `SeverityResult`)
>
> **Tests nuevos:** `test_severity_rules.py` (35 tests)

### D.2 — Factories que Deciden

**🔴 `ml/core/engine_factory.py` — `get_engine_for_sensor()`**

```python
# Línea 210: import de ml_service DENTRO de ml/core (violación de dirección)
from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags

# Líneas 220-238: Lógica de decisión compleja
if flags.ML_ROLLBACK_TO_BASELINE: ...
if sensor_id in flags.ML_ENGINE_OVERRIDES: ...
if flags.ML_USE_TAYLOR_PREDICTOR and flags.is_sensor_in_whitelist(sensor_id): ...
```

> **Diagnóstico:** Una factory no debería tomar decisiones de negocio. La selección de engine por feature flags es lógica de aplicación, no de creación. Además, `ml/core/` importa `ml_service/config/` — violación de dirección de dependencias.

### D.3 — Reglas de Negocio en Infraestructura

**✅ `ml_service/runners/common/severity_classifier.py` — REFACTORIZADO (TAREA 4)**

> **Estado:** Solo I/O (161 líneas, antes 254). Reglas extraídas a:
> - `domain/entities/sensor_ranges.py` (`DEFAULT_SENSOR_RANGES`)
> - `domain/services/severity_rules.py` (5 funciones puras + `SeverityResult`)
>
> **Tests nuevos:** `test_severity_rules.py` (35 tests)

### D.4 — Duplicación entre ml/core y Enterprise

| Concepto | ml/core/ (Fase 1) | domain/ (Enterprise) | Duplicado? |
|----------|-------------------|---------------------|------------|
| Interfaz de predicción | `PredictionEngine` ABC | `PredictionPort` ABC | ⚠️ **SÍ** — Firmas diferentes pero mismo propósito |
| Resultado de predicción | `PredictionResult` (frozen dataclass) | `Prediction` (frozen dataclass) | ⚠️ **SÍ** — Campos similares, nombres distintos |
| Interfaz de filtro | `SignalFilter` ABC | *(no existe)* | ❌ No — Sin equivalente enterprise |
| Validación numérica | `validators.py` (validate_window, clamp) | *(no existe en domain/)* | ❌ No — Pero debería existir |
| Tipo de patrón | `ml/pattern_detector.PatternType` | `domain/entities/pattern.PatternType` | ⚠️ **SÍ** — Dos enums con valores similares |
| Resultado de patrón | `ml/pattern_detector.PatternResult` | `domain/entities/pattern.PatternResult` | ⚠️ **SÍ** — Dos dataclasses con campos similares |

> **Diagnóstico:** 4 duplicaciones conceptuales. No son copias exactas (firmas diferentes), pero representan el mismo concepto en dos generaciones del código.

### D.5 — Fronteras Rotas

| Violación | Archivo origen | Importa de | Dirección correcta |
|-----------|---------------|------------|-------------------|
| `ml/core` → `ml_service` | `engine_factory.py:210` | `ml_service.config.feature_flags` | `ml/core` NO debería conocer `ml_service` |
| `ml_service` → `ml` directo | `prediction_service.py:15` | `ml.baseline` | Debería ir vía port/adapter |
| `severity_classifier` mezcla I/O + reglas | `severity_classifier.py:60` | `sqlalchemy` + reglas hardcoded | Separar en repository + domain service |

> **Nota:** `domain/` → `infrastructure/` = **0 violaciones**. La capa enterprise está limpia.

---

## E) ENTREGABLES

### E.1 — TABLA RESUMEN POR ARCHIVO

| Archivo | Rol real | Debería ser (UTSAE) | Problemas | Riesgo |
|---------|----------|---------------------|-----------|--------|
| `ml/core/interfaces.py` | Contratos Fase 1 | Modeling/contracts | Duplica `domain/ports/` | BAJO |
| `ml/core/validators.py` | Validación numérica | Modeling/validation | Sin equivalente enterprise | BAJO |
| `ml/core/taylor_predictor.py` | Motor de predicción | Modeling/engine | Implementa interfaz Fase 1, no Port | BAJO |
| `ml/core/kalman_filter.py` | Filtro de señal | Modeling/filter | Sin port enterprise equivalente | BAJO |
| `ml/core/engine_factory.py` | Factory + decisión | Debería ser Application | Importa `ml_service` (violación) | **MEDIO** |
| `domain/entities/*.py` | Value objects | Domain/entities ✅ | Ninguno | NINGUNO |
| `domain/ports/*.py` | ABCs | Domain/ports ✅ | Ninguno | NINGUNO |
| `domain/services/*.py` | Orquestación dominio | Domain/services ✅ | anomaly_service tiene math inline | BAJO |
| `infra/ml/engines/ensemble_engine.py` | Ensemble predictor | Modeling+Reasoning | 3 responsabilidades inherentes | BAJO |
| `infra/ml/anomaly/voting_anomaly_detector.py` | Orchestrator + sklearn | Orchestration+Infra ✅ | **REFACTORIZADO (T3)** | NINGUNO |
| `infra/ml/patterns/*.py` | Detectores | Modeling+Reasoning | 2-3 resp. inherentes | BAJO |
| `infra/ml/explainability/*.py` | Explicabilidad | Narrative ✅ | Ninguno | NINGUNO |
| `infra/security/*.py` | Seguridad | Adaptation ✅ | Ninguno | NINGUNO |
| `infra/adapters/*.py` | Cache + Batch | Adaptation ✅ | Ninguno | NINGUNO |
| `application/use_cases/*.py` | Casos de uso | Application ✅ | Ninguno | NINGUNO |
| `application/dto/*.py` | DTOs | Application ✅ | Ninguno | NINGUNO |
| `ml_service/api/services/prediction_service.py` | Thin Orchestrator | Application/orchestration ✅ | **REFACTORIZADO (T1)** | NINGUNO |
| `ml_service/runners/common/sensor_processor.py` | Thin Orchestrator | Application/orchestration ✅ | **REFACTORIZADO (T2)** | NINGUNO |
| `ml_service/runners/common/severity_classifier.py` | Solo I/O | Adaptation/repository ✅ | **REFACTORIZADO (T4)** | NINGUNO |
| `ml_service/orchestrator/prediction_orchestrator.py` | Orquestador | Application/orchestration | BD directa | BAJO |
| `ml/pattern_detector.py` | Detector monolítico | Modeling+Reasoning+Narrative | 606 líneas, 4 resp. | **MEDIO** |
| `ml/baseline.py` | Cálculo puro | Modeling ✅ | Ninguno | NINGUNO |

### E.2 — MAPA UTSAE DEL PROYECTO

```
ESTADO ACTUAL:
                                                    
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ SENSING  │    │ MODELING │    │REASONING │    │NARRATIVE │
  │          │    │          │    │          │    │          │
  │ repos/   │    │ taylor   │    │ domain/  │    │ explain/ │
  │ broker   │    │ kalman   │    │ services │    │ feature_ │
  │ buffer   │    │ baseline │    │ access   │    │ import.  │
  │          │    │ ensemble │    │ severity │    │ counter. │
  │          │    │ cusum    │    │ voting   │    │          │
  │          │    │ regime   │    │ delta_sp │    │          │
  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
       │               │               │               │
       └───────────────┴───────┬───────┴───────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   ORQUESTACIÓN      │
                    │                     │
                    │ ✅ application/     │ ← Enterprise (limpio)
                    │    use_cases/       │
                    │                     │
                    │ ⚠️ ml_service/     │ ← Legacy (God Objects)
                    │    prediction_svc   │
                    │    sensor_processor │
                    │    orchestrator     │
                    └─────────────────────┘

PROBLEMA CENTRAL:
  Los God Objects en ml_service/ MEZCLAN todas las capas UTSAE.
  Las capas enterprise (domain/, application/, infrastructure/)
  están LIMPIAS pero DESCONECTADAS del runtime.
```

### E.3 — LISTA DE REFACTORS PRIORITARIOS

#### Prioridad 1 — ALTO IMPACTO (reducen God Objects)

**R1. Extraer math de `voting_anomaly_detector.py`** ✅ COMPLETADO (TAREA 3)
- ✅ Creado `infra/ml/anomaly/statistical_methods.py` con 7 funciones puras
- ✅ Creado `infra/ml/anomaly/anomaly_narrator.py` (narrativa)
- ✅ Detector refactorizado como orchestrator
- ✅ 42 tests nuevos (`test_statistical_methods.py` + `test_anomaly_narrator.py`)
- ✅ Fix test flaky `test_extreme_value_is_anomaly`

**R2. Extraer reglas de negocio de `severity_classifier.py`** ✅ COMPLETADO (TAREA 4)
- ✅ Creado `domain/entities/sensor_ranges.py` (`DEFAULT_SENSOR_RANGES`)
- ✅ Creado `domain/services/severity_rules.py` (5 funciones puras + `SeverityResult`)
- ✅ `severity_classifier.py` reducido a solo I/O (161 líneas)
- ✅ 35 tests nuevos (`test_severity_rules.py`)

**R3. Conectar `prediction_service.py` con enterprise** ✅ COMPLETADO (TAREA 1)
- ✅ `PredictionService` reescrito como thin orchestrator
- ✅ Creado `SqlServerStorageAdapter` (implementa `StoragePort`)
- ✅ Creado `BaselinePredictionAdapter` (implementa `PredictionPort`)
- ✅ Creado `ThresholdEvaluator` (domain service puro)
- ✅ Creado `ThresholdRepository` (infra SQL)
- ✅ Tests: `test_threshold_evaluator.py`, `test_baseline_adapter.py`

**R3b. Refactorizar `sensor_processor.py`** ✅ COMPLETADO (TAREA 2)
- ✅ Creado `RegressionPredictionService` (Modeling puro)
- ✅ Creado `PredictionNarrator` (Narrative puro)
- ✅ `SensorProcessor` reescrito como thin orchestrator (193 líneas)
- ✅ Tests: `test_regression_prediction_service.py` (13), `test_prediction_narrator.py` (14)

#### Prioridad 2 — MEDIO IMPACTO (eliminan duplicación)

**R4. Unificar `PredictionEngine` y `PredictionPort`**
- `PredictionPort` es la interfaz correcta (usa `SensorWindow`)
- `TaylorPredictionAdapter` ya resuelve la interoperabilidad
- Marcar `PredictionEngine` como deprecated
- **Riesgo:** BAJO — adapter ya existe

**R5. Unificar `PatternType` y `PatternResult` duplicados**
- `ml/pattern_detector.PatternType` vs `domain/entities/pattern.PatternType`
- Crear adapter similar a `taylor_adapter.py` para `PatternDetector`
- **Riesgo:** BAJO — `ml/pattern_detector.py` solo se usa en stream runner

**R6. Mover `engine_factory.get_engine_for_sensor()` a application/**
- La lógica de selección por feature flags es de aplicación, no de factory
- Factory queda solo con `register()` + `create()`
- **Riesgo:** MEDIO — requiere actualizar tests de A/B comparison

#### Prioridad 3 — BAJO IMPACTO (mejoras de higiene)

**R7. Crear `SignalFilterPort` en `domain/ports/`**
- Kalman no tiene port enterprise equivalente
- Crear ABC para filtros de señal en el dominio
- **Riesgo:** BAJO — no rompe nada existente

**R8. Mover `validators.py` a `domain/validators/`**
- Validación numérica es lógica de dominio pura
- **Riesgo:** BAJO — actualizar imports en taylor_predictor

**R9. Extraer narrativa de `voting_anomaly_detector.py`** ✅ COMPLETADO (TAREA 3)
- ✅ Creado `infra/ml/anomaly/anomaly_narrator.py`
- ✅ 11 tests en `test_anomaly_narrator.py`

### E.4 — PROPUESTA DE ESTRUCTURA FINAL

```
iot_machine_learning/
├── domain/                          # ✅ PURO — Solo lógica de negocio
│   ├── entities/                    # Value objects inmutables
│   │   ├── sensor_reading.py
│   │   ├── prediction.py
│   │   ├── anomaly.py
│   │   ├── pattern.py
│   │   └── sensor_ranges.py        # ← NUEVO (R2: rangos por tipo)
│   ├── ports/                       # ABCs (contratos)
│   │   ├── prediction_port.py
│   │   ├── anomaly_detection_port.py
│   │   ├── pattern_detection_port.py
│   │   ├── signal_filter_port.py   # ← NUEVO (R7)
│   │   ├── storage_port.py
│   │   └── audit_port.py
│   ├── services/                    # Orquestación de dominio
│   │   ├── prediction_domain_service.py
│   │   ├── anomaly_domain_service.py
│   │   ├── pattern_domain_service.py
│   │   └── severity_rules.py       # ← NUEVO (R2: reglas puras)
│   └── validators/                  # ← NUEVO (R8)
│       └── numeric_validators.py
│
├── application/                     # ✅ LIMPIO — Casos de uso
│   ├── use_cases/
│   │   ├── predict_sensor_value.py
│   │   ├── detect_anomalies.py
│   │   ├── analyze_patterns.py
│   │   └── select_engine.py        # ← NUEVO (R6: lógica de feature flags)
│   └── dto/
│       └── prediction_dto.py
│
├── infrastructure/                  # ✅ LIMPIO — Implementaciones
│   ├── ml/
│   │   ├── engines/
│   │   │   ├── ensemble_engine.py
│   │   │   └── taylor_adapter.py
│   │   ├── anomaly/
│   │   │   ├── voting_anomaly_detector.py
│   │   │   └── statistical_methods.py  # ← NUEVO (R1: math puro)
│   │   ├── patterns/
│   │   │   ├── change_point_detector.py
│   │   │   ├── delta_spike_classifier.py
│   │   │   └── regime_detector.py
│   │   └── explainability/
│   │       ├── feature_importance.py
│   │       └── anomaly_narrator.py     # ← NUEVO (R9)
│   ├── security/
│   │   ├── audit_logger.py
│   │   └── access_control.py
│   ├── adapters/
│   │   ├── prediction_cache.py
│   │   ├── batch_predictor.py
│   │   └── sqlserver_storage.py        # ← NUEVO (R3: StoragePort impl)
│   └── repositories/                   # ← NUEVO (R2)
│       └── threshold_repository.py
│
├── ml/                              # ⚠️ LEGACY (deprecated, mantener por compat)
│   ├── core/                        # Fase 1 — deprecated
│   │   ├── interfaces.py           # → usar domain/ports/
│   │   ├── validators.py           # → usar domain/validators/
│   │   ├── taylor_predictor.py     # → usar via taylor_adapter
│   │   ├── kalman_filter.py        # → usar via kalman_adapter
│   │   └── engine_factory.py       # → usar application/select_engine
│   ├── baseline.py                  # Mantener (cálculo puro)
│   └── pattern_detector.py          # → crear adapter enterprise
│
├── ml_service/                      # Presentación (FastAPI + runners)
│   ├── api/
│   │   ├── routes.py
│   │   └── services/
│   │       └── prediction_service.py  # → delegar a use_cases (R3)
│   ├── runners/
│   │   ├── ml_batch_runner.py
│   │   ├── ml_stream_runner.py
│   │   └── common/
│   │       ├── sensor_processor.py    # → delegar a use_cases (R3)
│   │       ├── severity_classifier.py # → split domain + repo (R2)
│   │       └── ...
│   └── orchestrator/
│       └── prediction_orchestrator.py
│
└── tests/
    ├── unit/
    │   ├── domain/
    │   │   ├── test_entities.py
    │   │   └── test_severity_rules.py  # ← NUEVO
    │   └── infrastructure/
    │       ├── test_voting_anomaly.py
    │       ├── test_statistical_methods.py  # ← NUEVO
    │       └── ...
    └── integration/
        └── test_enterprise_flow.py
```

### E.5 — PRUEBAS DE COHERENCIA

| Regla | Estado | Evidencia (actualizado post-refactor) |
|-------|--------|-----------|
| **Cada archivo responde UNA pregunta** | ⚠️ 2 archivos violan | ~~`prediction_service`~~ ✅, ~~`sensor_processor`~~ ✅, ~~`voting_anomaly_detector`~~ ✅, ~~`severity_classifier`~~ ✅, `engine_factory` ⏳, `pattern_detector` ⏳ |
| **Ningún módulo mezcla niveles UTSAE** | ⚠️ 0 archivos mezclan (de los refactorizados) | ~~`voting_anomaly_detector`~~ ✅, ~~`sensor_processor`~~ ✅, ~~`prediction_service`~~ ✅, ~~`severity_classifier`~~ ✅ — Pendientes: `engine_factory`, `pattern_detector` |
| **Infra no contiene decisiones** | ⚠️ 1 violación real | `voting_anomaly_detector` contiene lógica de decisión (voting threshold). Aceptable: es inherente al algoritmo. |
| **Domain es puro significado** | ✅ CUMPLE | `domain/` tiene 0 imports de infra, 0 I/O, 0 dependencias externas |
| **Domain no importa Infrastructure** | ✅ CUMPLE | Verificado por grep: 0 imports cruzados |
| **Application solo orquesta** | ✅ CUMPLE | Use cases solo llaman domain services + storage port |
| **No hay lógica de negocio en utils** | ✅ CUMPLE | `validators.py` es validación numérica, no reglas de negocio |

---

## RESUMEN EJECUTIVO

### Lo que está BIEN (no tocar)

1. **`domain/`** — Impecable. 0 violaciones. Entities, ports y services son puros.
2. **`application/`** — Impecable. Use cases solo orquestan. DTOs son planos.
3. **`infrastructure/security/`** — Limpio. Audit logger y RBAC bien separados.
4. **`infrastructure/adapters/`** — Limpio. Cache y batch son infra pura.
5. **`ml/core/taylor_predictor.py`** — Limpio. Cálculo matemático puro.
6. **`ml/core/kalman_filter.py`** — Limpio. Cálculo matemático puro.
7. **`ml/baseline.py`** — Limpio. 40 líneas de media móvil.

### Lo que necesita refactor (por prioridad)

| # | Archivo | Problema | Estado | Tests nuevos |
|---|---------|----------|--------|-------------|
| 1 | `prediction_service.py` | God Service (5 resp.) | ✅ **COMPLETADO** | +2 archivos |
| 2 | `sensor_processor.py` | God Processor (5 resp.) | ✅ **COMPLETADO** | +2 archivos (27 tests) |
| 3 | `voting_anomaly_detector.py` | 5 resp. en 289 líneas | ✅ **COMPLETADO** | +2 archivos (42 tests) |
| 4 | `severity_classifier.py` | Reglas + SQL mezclados | ✅ **COMPLETADO** | +1 archivo (35 tests) |
| 5 | `engine_factory.py` | Factory que decide + viola dirección | ⏳ **PENDIENTE** | — |
| 6 | Unificar interfaces duplicadas | PredictionEngine vs PredictionPort | ⏳ **PENDIENTE** | — |

### Veredicto final

**La arquitectura hexagonal enterprise (`domain/`, `application/`, `infrastructure/`) está correctamente implementada.**

**Progreso de refactorización (4/6 completadas):**

- ✅ **T1:** `prediction_service.py` — God Service → Thin Orchestrator
- ✅ **T2:** `sensor_processor.py` — God Processor → Thin Orchestrator
- ✅ **T3:** `voting_anomaly_detector.py` — 5 resp. → Orchestrator + módulos extraídos
- ✅ **T4:** `severity_classifier.py` — Reglas + SQL → Solo I/O + domain rules
- ⏳ **T5:** `engine_factory.py` — Factory que decide + viola dirección de dependencias
- ⏳ **T6:** Unificar interfaces duplicadas (PredictionEngine/PredictionPort, PatternType)

**Módulos nuevos creados (12):**

| Módulo | Capa | Responsabilidad |
|--------|------|----------------|
| `infrastructure/adapters/sqlserver_storage.py` | Infra | StoragePort SQL Server |
| `infrastructure/ml/engines/baseline_adapter.py` | Infra | PredictionPort wrapper |
| `domain/services/threshold_evaluator.py` | Domain | Reglas de umbral puras |
| `infrastructure/repositories/threshold_repository.py` | Infra | SQL queries umbrales |
| `ml_service/runners/common/regression_prediction_service.py` | Modeling | Regresión + fallback |
| `ml_service/runners/common/prediction_narrator.py` | Narrative | Explicaciones legibles |
| `infrastructure/ml/anomaly/statistical_methods.py` | Modeling | Z-score, IQR, voting puro |
| `infrastructure/ml/anomaly/anomaly_narrator.py` | Narrative | Texto de anomalías |
| `domain/entities/sensor_ranges.py` | Domain | Rangos operativos |
| `domain/services/severity_rules.py` | Domain | Reglas de severidad puras |

**Tests:** 264 originales + 104 nuevos = **368 total, 0 fallos.**

**Reducción de responsabilidades:**

| Archivo | Antes | Ahora | Reducción |
|---------|-------|-------|-----------|
| `prediction_service.py` | 5 resp. (368 lín.) | 1 resp. (196 lín.) | -80% resp., -47% lín. |
| `sensor_processor.py` | 5 resp. (319 lín.) | 1 resp. (193 lín.) | -80% resp., -40% lín. |
| `voting_anomaly_detector.py` | 5 resp. (289 lín.) | 2 resp. (286 lín.) | -60% resp. |
| `severity_classifier.py` | 4 resp. (254 lín.) | 2 resp. (161 lín.) | -50% resp., -37% lín. |
