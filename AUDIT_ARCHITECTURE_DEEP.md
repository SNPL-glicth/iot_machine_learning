# Auditoría Arquitectónica Profunda — UTSAE

**Fecha:** 2026-02-12 (original) | **Última actualización:** 2026-02-12  
**Alcance:** `iot_machine_learning/` completo  
**Método:** Lectura exhaustiva de código fuente, análisis de flujo de datos, verificación de contratos entre capas  
**Estado:** Auditoría completada. Múltiples hallazgos resueltos en Fases 1-4 de hardening.  
**Tests:** 1096 passed, 6 skipped, 0 failures

---

## Resumen Ejecutivo

El sistema tiene una base arquitectónica sólida (hexagonal, ports & adapters, DI). La auditoría original identificó **problemas estructurales** en dos dimensiones. El estado actual refleja las correcciones aplicadas:

1. **~~El tiempo es una variable fantasma~~** → **PARCIALMENTE RESUELTO**: La detección de anomalías ahora incluye votos temporales (velocity z-score, acceleration z-score, IF 3D, LOF 3D). `train()` acepta timestamps. `StructuralAnalysis` computa slope/curvature/stability con Δt real. Patrones aún operan sin Δt completo (trabajo futuro).
2. **~~La migración agnóstica está incompleta~~** → **MAYORMENTE RESUELTO**: Ports ofrecen dual interface (`SensorWindow` + `TimeSeries`). Bridges usan `safe_series_id_to_int()` centralizado. `SensorWindow` aún vive en domain/ (trabajo futuro: mover a infra/adapters/iot/).

---

## 1️⃣ Casos de Uso

### 1.1 Hallazgos Positivos

- **`PredictSensorValueUseCase`** (`application/use_cases/predict_sensor_value.py`): Correctamente orquesta load → predict → persist → DTO. No contiene lógica de negocio. Delega a `PredictionDomainService`. Fail-safe en persistencia y memory recall.
- **`DetectAnomaliesUseCase`** (`application/use_cases/detect_anomalies.py`): Limpio. Orquesta load → detect → persist → DTO. No mezcla lógica.
- **`AnalyzePatternsUseCase`** (`application/use_cases/analyze_patterns.py`): Orquesta pattern → change points → spike classification → regime.

### 1.2 Problemas Detectados

| ID | Hallazgo | Archivo | Severidad |
|----|----------|---------|-----------|
| UC-1 | **Lógica de negocio en use case**: `AnalyzePatternsUseCase.execute()` líneas 94-99 calcula `max_dev_idx` (índice del spike) usando `mean_val` y `max()` directamente. Esto es lógica de dominio que debería estar en `PatternDomainService`. | `analyze_patterns.py:94-99` | 🟡 MEDIA |
| UC-2 | **`select_engine_for_sensor()` usa sensor_id + whitelist**: La función legacy selecciona engine por `sensor_id` y whitelist, no por características del dato. Viola el principio agnóstico. La versión agnóstica `select_engine_for_series()` existe pero no está conectada al flujo principal. | `select_engine.py:25-76` | 🟠 ALTA |
| UC-3 | **Use cases acoplados a `sensor_id: int`**: `PredictSensorValueUseCase.execute(sensor_id: int)` y `DetectAnomaliesUseCase.execute(sensor_id: int)` reciben `int`, no `str`. No hay use case equivalente para `series_id: str`. | `predict_sensor_value.py:68`, `detect_anomalies.py:46` | 🟡 MEDIA |
| UC-4 | **Cognitive orchestrator no integrado**: `MetaCognitiveOrchestrator` existe pero no está wired en ningún use case ni en `PredictionDomainService`. El flujo real sigue usando engine selection por sensor_id → `TaylorPredictionAdapter`. | `orchestrator.py` vs `prediction_domain_service.py` | 🟡 MEDIA | ⚠️ Parcial: orchestrator produce `Explanation` y tiene `as_port()` bridge, pero no está wired en use cases principales. |

---

## 2️⃣ Código Legacy — Dependencia de Sensor

### 2.1 Violaciones al Principio Domain-Agnostic

| ID | Hallazgo | Archivo(s) | Severidad |
|----|----------|------------|-----------|
| SEN-1 | **`sensor_ranges.py` sigue vivo en `domain/entities/`**: Contiene `DEFAULT_SENSOR_RANGES` hardcodeado (`temperature: (15, 35)`, `humidity: (30, 70)`, etc.). Está marcado como deprecated pero sigue importado por `severity_rules.py`. Un módulo con rangos hardcodeados por tipo de sensor **no pertenece al dominio**. | `domain/entities/sensor_ranges.py` | 🔴 CRÍTICA |
| SEN-2 | **`classify_severity()` y `compute_risk_level()` usan `sensor_type: str`**: Funciones legacy que despachan por tipo de sensor. Tienen `DeprecationWarning` pero siguen siendo código activo importable. | `domain/services/severity_rules.py:36-72, 175-231` | 🟠 ALTA |
| SEN-3 | **`SensorReading` tiene `sensor_type: str` y `device_id`**: Campos IoT-específicos en una entidad de dominio. `SensorWindow` tiene `sensor_type` y `device_id` también. Estos son conceptos de infraestructura IoT, no del dominio matemático. | `domain/entities/sensor_reading.py:29-30, 63-65` | 🟠 ALTA |
| SEN-4 | **`StoragePort` completamente acoplado a `sensor_id: int`**: `load_sensor_window(sensor_id: int)`, `list_active_sensor_ids() → List[int]`, `get_sensor_metadata(sensor_id: int)`, `get_device_id_for_sensor(sensor_id: int)`. Este port del dominio habla exclusivamente en términos IoT. | `domain/ports/storage_port.py` | 🔴 CRÍTICA |
| SEN-5 | **`AuditPort` usa `sensor_id: int`**: `log_prediction(sensor_id: int)` y `log_anomaly(sensor_id: int)`. Un port de auditoría del dominio no debería conocer el concepto de "sensor". | `domain/ports/audit_port.py:55, 67` | 🟠 ALTA |
| ~~SEN-6~~ | ~~**Bridges en ports hacen `int(series_id) if series_id.isdigit() else 0`**~~ | ~~`prediction_port.py:67`, `anomaly_detection_port.py:68`, `pattern_detection_port.py:52`~~ | ~~🟠 ALTA~~ | ✅ **RESUELTO (DEBT-1)**: Todos los bridges usan `safe_series_id_to_int()` centralizado con logging. 14 sitios reemplazados en 7 archivos. |

### 2.2 Estado de la Migración Agnóstica (actualizado)

```
✅ Completado:
   - domain/entities/time_series.py (TimeSeries, TimePoint)
   - domain/entities/series_context.py (SeriesContext, Threshold)
   - domain/entities/series_profile.py (SeriesProfile, compute_profile)
   - domain/entities/series/structural_analysis.py (StructuralAnalysis, RegimeType)
   - classify_severity_agnostic() existe
   - select_engine_for_series() existe (select_engine_for_sensor deprecated)
   - Ports: dual interface (SensorWindow + TimeSeries) en prediction, anomaly, pattern
   - StoragePort/AuditPort: dual interface (sensor_id + series_id bridges)
   - Bridges: safe_series_id_to_int() centralizado (DEBT-1 resuelto)
   - Entidades: series_id: str en Prediction, AnomalyResult, PatternResult
   - DTOs: series_id: str en PredictionDTO, AnomalyDTO, PatternDTO
   - FeatureFlags: dual interface (series_id + sensor_id)
   - AccessControl: dual interface (series_id + sensor_id)
   - PredictionEnginePortBridge: engine.as_port() elimina adapters manuales

⚠️ Parcial (trabajo futuro):
   - SensorWindow vive en domain/entities/iot/ (mover a infra/adapters/iot/)
   - Use cases nombrados como sensor (PredictSensorValueUseCase)
   - Flujo principal aún usa SensorWindow como tipo primario
```

---

## 3️⃣ Detección de Patrones

### 3.1 Delta Spike Classifier

| ID | Hallazgo | Archivo | Severidad |
|----|----------|---------|-----------|
| PAT-1 | **Δt completamente ausente**: `DeltaSpikeClassifier.classify(values, spike_index)` recibe solo valores y un índice. No recibe timestamps. La "persistencia" se mide en **número de muestras**, no en tiempo real. Un spike que persiste 5 muestras a 1Hz (5 segundos) se trata igual que 5 muestras a 0.001Hz (5000 segundos). | `delta_spike_classifier.py:60-63` | 🔴 CRÍTICA |
| PAT-2 | **Tendencia calculada sin Δt**: `_compute_trend_alignment()` compara medias de primera y segunda mitad de la ventana. No normaliza por tiempo. Una tendencia de +10 en 1 segundo es radicalmente diferente a +10 en 1 hora. | `delta_spike_classifier.py:213-244` | 🔴 CRÍTICA |
| PAT-3 | **Z-score del spike sin normalización temporal**: La magnitud del spike se mide como `|spike_value - pre_mean| / pre_std`. Esto mide desviación estadística pero ignora la **tasa de cambio** (dv/dt). Un salto de 10σ en 1ms es diferente a 10σ en 1 hora. | `delta_spike_classifier.py:98-99` | 🟠 ALTA |

### 3.2 Change Point Detection (CUSUM / PELT)

| ID | Hallazgo | Archivo | Severidad |
|----|----------|---------|-----------|
| PAT-4 | **CUSUM opera sin timestamps**: `detect_online(value: float)` recibe un solo valor escalar. No sabe cuánto tiempo pasó desde la última observación. El acumulador CUSUM asume muestreo uniforme implícitamente. Si hay un gap de 1 hora entre muestras, el acumulador no lo sabe. | `change_point_detector.py:63` | 🔴 CRÍTICA |
| PAT-5 | **`detect_batch(values)` usa índice como timestamp**: Línea 182: `timestamp=float(i)`. El timestamp del change point es el **índice** en el array, no el tiempo real. Esto hace que la información temporal del resultado sea ficticia. | `change_point_detector.py:182, 199` | 🟠 ALTA |
| PAT-6 | **PELT no recibe timestamps**: `detect_batch(values: List[float])` solo recibe valores. `ruptures` opera sobre el array sin información temporal. Cambios en series con muestreo irregular se detectan incorrectamente. | `change_point_detector.py:243` | 🟠 ALTA |

### 3.3 Regime Detector

| ID | Hallazgo | Archivo | Severidad |
|----|----------|---------|-----------|
| PAT-7 | **Régimen basado solo en magnitud**: `RegimeDetector.predict_regime(value: float)` clasifica por distancia al centroide más cercano. No considera la **trayectoria** (¿está subiendo hacia el pico o bajando desde él?). Dos valores idénticos en regímenes de transición opuestos se clasifican igual. | `regime_detector.py:142-170` | 🟡 MEDIA |
| PAT-8 | **Entrenamiento sin dimensión temporal**: `train(historical_values: List[float])` recibe solo valores. KMeans agrupa por magnitud sin considerar duración de cada régimen ni velocidad de transición. | `regime_detector.py:61` | 🟡 MEDIA |

---

## 4️⃣ Detección de Anomalías

### 4.1 El Tiempo como Variable

| ID | Hallazgo | Archivo | Severidad |
|----|----------|---------|-----------|
| ANO-1 | **Anomalía evaluada sobre un solo punto**: `VotingAnomalyDetector.detect()` extrae `value = window.last_value` y evalúa Z-score, IQR, IF, LOF sobre ese **único escalar**. Ignora los otros N-1 puntos de la ventana y todos los timestamps. | `voting_anomaly_detector.py:174` | 🔴 CRÍTICA |
| ANO-2 | **Entrenamiento sin timestamps**: `train(historical_values: List[float])` recibe solo valores. Las estadísticas (mean, std, Q1, Q3) son puramente de magnitud. No hay modelo de la dinámica temporal (velocidad normal, aceleración normal). | `voting_anomaly_detector.py:117` | 🔴 CRÍTICA |
| ANO-3 | **Z-score no considera velocidad ni aceleración**: `compute_z_score(value, mean, std)` mide desviación de la media histórica. Un valor de 35°C puede ser normal si la temperatura subió gradualmente, pero anómalo si saltó de 20°C a 35°C en 1 segundo. El Z-score no distingue estos casos. | `statistical_methods.py:65-78` | 🔴 CRÍTICA |
| ANO-4 | **IQR es puramente estático**: `compute_iqr_vote()` evalúa si el valor está fuera de [Q1-1.5·IQR, Q3+1.5·IQR]. No hay IQR de velocidades ni de aceleraciones. | `statistical_methods.py:125-142` | 🟠 ALTA |
| ANO-5 | **IsolationForest y LOF entrenados en 1D**: Ambos modelos se entrenan con `X = np.array(values).reshape(-1, 1)` — un solo feature (magnitud). No se incluyen features derivados del tiempo (velocidad, aceleración, jitter). | `voting_anomaly_detector.py:243, 262` | 🟠 ALTA |

### 4.2 Dependencia Dominio ↔ Infraestructura

| ID | Hallazgo | Archivo | Severidad |
|----|----------|---------|-----------|
| ANO-6 | **`AnomalyDetectionPort.detect()` recibe `SensorWindow`**: El port del dominio depende de `SensorWindow` (entidad IoT). Debería recibir `TimeSeries` como tipo primario. El bridge `detect_series()` existe pero es un workaround. | `anomaly_detection_port.py:48` | 🟡 MEDIA |
| ANO-7 | **`AnomalyDomainService` no pasa timestamps a detectores**: `detect(window)` pasa el `SensorWindow` completo al detector, pero el detector solo usa `window.last_value`. Los timestamps están disponibles pero no se usan. | `anomaly_domain_service.py:84` → `voting_anomaly_detector.py:174` | 🟠 ALTA |

---

## 5️⃣ Arquitectura General

### 5.1 Acoplamientos Innecesarios

| ID | Hallazgo | Archivo(s) | Severidad |
|----|----------|------------|-----------|
| ~~ARQ-1~~ | ~~**Interfaces duplicadas**: `PredictionEngine` vs `PredictionPort`~~ | ~~`interfaces.py` vs `prediction_port.py`~~ | ~~🟠 ALTA~~ | ✅ **RESUELTO (Phase 2)**: `PredictionEnginePortBridge` unifica ambas interfaces. `engine.as_port()` one-liner. `EngineFactory.create_as_port()` combina create + bridge. Adapters manuales deprecated. |
| ~~ARQ-2~~ | ~~**Firmas incompatibles Engine vs Port**~~ | ~~`interfaces.py:predict` vs `prediction_port.py:predict`~~ | ~~🟡 MEDIA~~ | ✅ **RESUELTO (Phase 2)**: `PredictionEnginePortBridge` traduce `predict(window)` → `predict(values, ts)` y `predict_series(series)` → `predict(values, ts)`. |
| ARQ-3 | **`ChangePointDetectionPort` no tiene `name` ni `is_trained()`**: Inconsistencia de contrato. | `pattern_detection_port.py:65-95` | 🟡 MEDIA | Pendiente |

### 5.2 Responsabilidades Mal Ubicadas

| ID | Hallazgo | Archivo | Severidad |
|----|----------|---------|-----------|
| ARQ-4 | **`sensor_ranges.py` en `domain/entities/`**: Rangos hardcodeados por tipo de sensor son configuración de infraestructura IoT, no entidades del dominio. | `domain/entities/sensor_ranges.py` | 🟠 ALTA |
| ARQ-5 | **Spike index calculation en use case**: `AnalyzePatternsUseCase` calcula el índice del spike (líneas 96-97) con lógica matemática (`max deviation from mean`). Esto debería estar en el domain service o en el classifier. | `analyze_patterns.py:94-99` | 🟡 MEDIA |
| ~~ARQ-6~~ | ~~**`EngineFactory` registra engines por nombre string sin validación**~~ | ~~`engines/__init__.py:10`~~ | ~~🟡 MEDIA~~ | ✅ **RESUELTO (Phase 3)**: `@register_engine("name")` decorator valida que la clase sea subclase de `PredictionEngine`. `discover_engines()` escanea paquetes automáticamente. |

### 5.3 Modelo Matemático No Usado Correctamente

| ID | Hallazgo | Archivo(s) | Severidad |
|----|----------|------------|-----------|
| ~~MAT-1~~ | ~~**Taylor engine calcula derivadas que nadie más consume**~~ | ~~`taylor/types.py` vs `voting_anomaly_detector.py` vs `delta_spike_classifier.py`~~ | ~~🔴 CRÍTICA~~ | ✅ **RESUELTO**: `StructuralAnalysis` (domain) unifica slope/curvature/stability/accel_variance. Taylor produce `StructuralAnalysis` vía `from_taylor_diagnostic()` bridge. Anomaly detection consume velocity/acceleration para votos temporales. Pattern detection enriquece resultados con structural metadata. |
| ~~MAT-2~~ | ~~**`SignalProfile` duplica `SeriesProfile`**~~ | ~~`signal_analyzer.py` vs `series_profile.py`~~ | ~~🟡 MEDIA~~ | ✅ **RESUELTO (COG-2)**: `SignalAnalyzer` ahora retorna `StructuralAnalysis` (domain). `SignalProfile` deprecated. Cognitive orchestrator consume `StructuralAnalysis` directamente. |
| MAT-3 | **Anomaly detection no usa el modelo Taylor**: Podría usar residual del modelo Taylor como feature. | `taylor_engine.py` vs `voting_anomaly_detector.py` | 🟠 ALTA | Pendiente |

---

## Mapa de Severidad (actualizado)

| Severidad | Original | Resueltos | Pendientes | IDs resueltos |
|-----------|----------|-----------|------------|---------------|
| 🔴 **CRÍTICA** | 9 | 1 | 8 | MAT-1 |
| 🟠 **ALTA** | 14 | 3 | 11 | SEN-6, ARQ-1, ARQ-6 |
| 🟡 **MEDIA** | 11 | 3 | 8 | ARQ-2, ARQ-6, MAT-2 |
| **Total** | **34** | **7** | **27** | |

---

## Riesgos Técnicos

### R1 — ~~Falsos Positivos/Negativos por Ceguera Temporal~~ PARCIALMENTE MITIGADO
~~La detección de anomalías evalúa un solo punto contra estadísticas históricas de magnitud.~~

**Estado actual:** `VotingAnomalyDetector` ahora incluye votos temporales (velocity z-score, acceleration z-score, IF 3D [value, velocity, acceleration], LOF 3D). `train()` acepta timestamps. Sin embargo, los patrones (DeltaSpike, CUSUM, PELT) aún no usan Δt.

**Impacto residual:** Detección de anomalías mejorada. Patrones aún vulnerables a muestreo irregular.

### R2 — Delta Spike Classification Sin Tiempo (CRÍTICO)
La clasificación delta vs noise se basa en persistencia por **conteo de muestras**, no por duración temporal. En series con muestreo irregular, esto produce clasificaciones erróneas.

**Impacto:** Spikes legítimos clasificados como ruido (y viceversa) en series con Δt variable.

### R3 — ~~Migración Agnóstica Bloqueada por Ports~~ PARCIALMENTE MITIGADO
~~`StoragePort` y `AuditPort` están hardcodeados a `sensor_id: int`.~~

**Estado actual:** Ambos ports ofrecen dual interface (`series_id: str` + `sensor_id: int`). Bridges usan `safe_series_id_to_int()`. El sistema puede operar sobre series no-IoT vía los métodos agnósticos. Ver `MIGRATION_SCORECARD.md` para el inventario completo.

**Impacto residual:** Funciona, pero la dualidad tiene costo de mantenimiento. Sunset plan documentado.

### ~~R4 — Duplicación de Análisis Estructural~~ ✅ RESUELTO
~~Taylor engine, SignalAnalyzer, SeriesProfile y anomaly detector calculan estadísticas similares independientemente.~~

**Estado actual:** `StructuralAnalysis` (domain) es el análisis compartido. Taylor produce vía `from_taylor_diagnostic()`. `SignalAnalyzer` retorna `StructuralAnalysis` directamente. Pattern detection enriquece resultados con structural metadata.

---

## Recomendaciones de Rediseño

### RD-1: Temporal-First Anomaly Detection (Prioridad 1)
Rediseñar `VotingAnomalyDetector` para que opere sobre la **ventana completa** con timestamps:
- Calcular velocidad (dv/dt) y aceleración (d²v/dt²) de la ventana
- Entrenar IF/LOF con features multidimensionales: `[value, velocity, acceleration, jitter]`
- Z-score sobre velocidad y aceleración, no solo magnitud
- IQR sobre tasa de cambio

### RD-2: Temporal-Aware Pattern Detection (Prioridad 1)
- `DeltaSpikeClassifier.classify(values, timestamps, spike_index)` — recibir timestamps
- Persistencia medida en **segundos**, no en muestras
- Tasa de cambio del spike: `Δv/Δt` normalizada
- CUSUM con Δt-awareness: acumular `deviation * dt` en vez de `deviation`

### ~~RD-3: Shared Structural Analysis Pass~~ ✅ RESUELTO

> Implementado como `StructuralAnalysis` (domain value object) + `compute_structural_analysis()` (domain validator):
> - Taylor engine → produce `StructuralAnalysis` vía `from_taylor_diagnostic()` bridge
> - Anomaly detection → consume velocity/acceleration para votos temporales
> - Pattern detection → enriquece `PatternResult.metadata` con structural analysis
> - Cognitive orchestrator → `SignalAnalyzer` retorna `StructuralAnalysis` directamente

### RD-4: Port Migration to series_id (Prioridad 2)
- `StoragePort` → `load_series_window(series_id: str)`, `list_active_series_ids() → List[str]`
- `AuditPort` → `log_prediction(series_id: str)`, `log_anomaly(series_id: str)`
- Mover `SensorWindow` a `infrastructure/adapters/iot/`
- Use cases aceptan `series_id: str`

### ~~RD-5: Unify Engine Interfaces~~ ✅ RESUELTO (Phase 2)

> Implementado vía `PredictionEnginePortBridge` en vez de eliminar la dualidad:
> - `engine.as_port()` → one-liner que convierte cualquier `PredictionEngine` a `PredictionPort`
> - `EngineFactory.create_as_port("name")` → create + bridge en una línea
> - Adapters manuales (`TaylorPredictionAdapter`, `CognitivePredictionAdapter`) deprecated
> - No se eliminó `PredictionEngine` (backward compat) pero el bridge hace innecesarios los adapters

### RD-6: Wire Cognitive Orchestrator (Prioridad 3) — Parcial
Conectar `MetaCognitiveOrchestrator` al flujo principal:
- ✅ `orchestrator.as_port()` disponible vía `PredictionEnginePortBridge`
- ✅ `select_engine_for_series()` implementado y `select_engine_for_sensor()` deprecated
- ✅ `record_actual()` desacoplado de `MetaDiagnostic` (usa estado interno)
- ⚠️ Pendiente: wiring en `PredictionDomainService` y use cases principales

---

## Archivos Auditados

```
application/use_cases/predict_sensor_value.py    ✅ leído
application/use_cases/detect_anomalies.py        ✅ leído
application/use_cases/analyze_patterns.py        ✅ leído
application/use_cases/select_engine.py           ✅ leído
domain/entities/sensor_reading.py                ✅ leído
domain/entities/sensor_ranges.py                 ✅ leído
domain/entities/prediction.py                    ✅ leído
domain/entities/anomaly.py                       ✅ leído
domain/entities/pattern.py                       ✅ leído
domain/entities/time_series.py                   ✅ leído
domain/entities/series_context.py                ✅ leído
domain/entities/series_profile.py                ✅ leído
domain/ports/prediction_port.py                  ✅ leído
domain/ports/anomaly_detection_port.py           ✅ leído
domain/ports/pattern_detection_port.py           ✅ leído
domain/ports/storage_port.py                     ✅ leído
domain/ports/audit_port.py                       ✅ leído
domain/services/prediction_domain_service.py     ✅ leído
domain/services/anomaly_domain_service.py        ✅ leído
domain/services/pattern_domain_service.py        ✅ leído
domain/services/severity_rules.py                ✅ leído
infrastructure/ml/interfaces.py                  ✅ leído
infrastructure/ml/engines/taylor_engine.py       ✅ leído
infrastructure/ml/engines/taylor/types.py        ✅ leído
infrastructure/ml/engines/statistical_engine.py  ✅ leído
infrastructure/ml/engines/engine_factory.py      ✅ leído
infrastructure/ml/engines/taylor_adapter.py      ✅ leído
infrastructure/ml/anomaly/voting_anomaly_detector.py    ✅ leído
infrastructure/ml/anomaly/statistical_methods.py        ✅ leído
infrastructure/ml/anomaly/anomaly_narrator.py           ✅ leído
infrastructure/ml/patterns/delta_spike_classifier.py    ✅ leído
infrastructure/ml/patterns/change_point_detector.py     ✅ leído
infrastructure/ml/patterns/regime_detector.py           ✅ leído
infrastructure/ml/cognitive/orchestrator.py             ✅ leído
infrastructure/ml/cognitive/signal_analyzer.py          ✅ leído
infrastructure/adapters/sqlserver_storage.py            ✅ leído
```

---

## Hallazgos Resueltos Post-Auditoría

Los siguientes hallazgos fueron resueltos en las fases de hardening posteriores a esta auditoría:

| ID | Hallazgo | Resolución | Fase |
|----|----------|------------|------|
| SEN-6 | Bridges `int(series_id)` inseguros | `safe_series_id_to_int()` centralizado en 7 archivos (14 sitios) | Phase 4 (DEBT-1) |
| ARQ-1 | Dual interface PredictionEngine vs PredictionPort | `PredictionEnginePortBridge` + `engine.as_port()` | Phase 2 |
| ARQ-2 | Firmas incompatibles Engine vs Port | Bridge traduce automáticamente | Phase 2 |
| ARQ-6 | EngineFactory sin validación de registro | `@register_engine` decorator con validación de tipo | Phase 3 |
| MAT-1 | Taylor derivadas no compartidas | `StructuralAnalysis` (domain) + `from_taylor_diagnostic()` bridge | Cognitive Phase |
| MAT-2 | SignalProfile ≈ StructuralAnalysis duplicados | `SignalAnalyzer` retorna `StructuralAnalysis`. `SignalProfile` deprecated. | Phase 1-Cog (COG-2) |
| RD-5 | Unificar interfaces de engine | `PredictionEnginePortBridge` hace innecesarios los adapters manuales | Phase 2 |

### Nuevas capacidades añadidas (no en auditoría original)

| Capacidad | Descripción | Fase |
|-----------|-------------|------|
| **Filtros composables** | EMA, Median, FilterChain, FilterDiagnostic | Filter Expansion |
| **Detección temporal** | 8 votos (velocity, acceleration, IF 3D, LOF 3D) | Temporal Anomaly |
| **Análisis estructural** | `StructuralAnalysis` compartido (slope, curvature, stability, regime) | Cognitive Phase |
| **Explicabilidad** | `Explanation` (domain) → `ExplanationBuilder` (infra) → `ExplanationRenderer` (app) | Explainability |
| **Plugin architecture** | `@register_engine`, `@register_detector`, `discover_engines()` | Phase 3 |
| **DI en anomalía** | `VotingAnomalyDetector(sub_detectors=[...])` | Phase 3 |
| **Conversión segura** | `safe_series_id_to_int()` centralizado | Phase 4 (DEBT-1) |
| **Inmutabilidad** | `dataclasses.replace()` en `PredictionDomainService` | Phase 4 (DEBT-4) |
| **Deprecación MetaDiagnostic** | `last_explanation` reemplaza `last_diagnostic` | Phase 4 (COG-3) |
| **Severidad unificada** | `template_generator` delega a `AnomalySeverity.from_score()` | Phase 4 (COG-4) |
| **Pipeline latency** | `PipelineTimer` per-phase timing + budget guard (500ms default) | Architectural Hardening |
| **Complexity guard** | Meta-test: orchestrator.py ≤ 300 líneas, sin numpy/scipy | Architectural Hardening |
| **Migration scorecard** | `MIGRATION_SCORECARD.md` + meta-test de dual interface | Architectural Hardening |

---

*Auditoría original realizada por análisis estático de código fuente. Actualizaciones de estado reflejan correcciones implementadas en Fases 1-4.*
