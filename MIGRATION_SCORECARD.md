# Migration Scorecard — Legacy IoT → UTSAE Agnóstico

**Última actualización:** 2026-02-12  
**Estrategia:** Dual interface con sunset gradual  
**Tests:** 1096 passed, 6 skipped

---

## Decisión estratégica

UTSAE opera como **sistema híbrido**: el core (domain/application/infrastructure) es agnóstico (`series_id: str`), mientras `ml_service/` mantiene la identidad IoT (`sensor_id: int`) como bridge.

### ¿Por qué dualidad?

- **Romper `ml_service/` hoy** = romper producción sin beneficio inmediato
- **Mantener dualidad indefinida** = costo de mantenimiento creciente
- **Sunset gradual** = migrar call-sites uno a uno, con deprecation warnings como guía

### Horizonte de sunset

| Fase | Estado | Descripción |
|------|--------|-------------|
| **Fase 1: Coexistencia** | ✅ Completada | Dual interface en ports, entities, DTOs, feature flags, access control |
| **Fase 2: Deprecation** | ✅ Completada | `DeprecationWarning` en todas las APIs legacy |
| **Fase 3: Migración gradual** | 🔄 En progreso | Migrar call-sites en `ml_service/` uno a uno |
| **Fase 4: Limpieza** | ⏳ Futuro | Eliminar APIs legacy, mover `SensorWindow` a infra |

---

## Scorecard: APIs Legacy vs Agnósticas

### Ports (domain/ports/)

| Port | Método Legacy | Método Agnóstico | Estado |
|------|--------------|-----------------|--------|
| `PredictionPort` | `predict(SensorWindow)` | `predict_series(TimeSeries)` | ✅ Dual |
| `AnomalyDetectionPort` | `detect(SensorWindow)` | `detect_series(TimeSeries)` | ✅ Dual |
| `PatternDetectionPort` | `detect_pattern(SensorWindow)` | `detect_pattern_series(TimeSeries)` | ✅ Dual |
| `StoragePort` | `load_sensor_window(sensor_id: int)` | Bridge vía `safe_series_id_to_int` | ⚠️ Parcial |
| `AuditPort` | `log_prediction(sensor_id: int)` | Bridge vía `safe_series_id_to_int` | ⚠️ Parcial |

### Entidades (domain/entities/)

| Entidad | Campo Legacy | Campo Agnóstico | Estado |
|---------|-------------|----------------|--------|
| `Prediction` | — | `series_id: str` | ✅ Migrado |
| `AnomalyResult` | — | `series_id: str` | ✅ Migrado |
| `PatternResult` | — | `series_id: str` | ✅ Migrado |
| `SensorReading` | `sensor_id: int` | `to_time_series()` bridge | ⚠️ Legacy + bridge |
| `SensorWindow` | `sensor_id: int` | `to_time_series()` bridge | ⚠️ Legacy + bridge |

### Selección de motor

| API | Tipo | Estado |
|-----|------|--------|
| `select_engine_for_sensor(sensor_id, flags)` | Legacy | ⚠️ Deprecated (emite warning) |
| `select_engine_for_series(profile, flags)` | Agnóstico | ✅ Activo |
| `EngineFactory.get_engine_for_sensor()` | Legacy | ⚠️ Deprecated (emite warning) |
| `EngineFactory.create(name)` | Agnóstico | ✅ Activo |
| `EngineFactory.create_as_port(name)` | Agnóstico | ✅ Activo |

### Feature Flags

| API | Tipo | Estado |
|-----|------|--------|
| `ML_ENGINE_OVERRIDES: Dict[int, str]` | Legacy | ⚠️ Coexiste |
| `ML_ENGINE_SERIES_OVERRIDES: Dict[str, str]` | Agnóstico | ✅ Activo |
| `is_sensor_in_whitelist(sensor_id: int)` | Legacy | ⚠️ Delega a agnóstico |
| `is_series_in_whitelist(series_id: str)` | Agnóstico | ✅ Activo |
| `get_active_engine_name(sensor_id: int)` | Legacy | ⚠️ Delega a agnóstico |
| `get_active_engine_for_series(series_id: str)` | Agnóstico | ✅ Activo |

### Access Control

| API | Tipo | Estado |
|-----|------|--------|
| `READ_SENSOR_DATA` | Legacy | ⚠️ Alias de `READ_SERIES_DATA` |
| `READ_SERIES_DATA` | Agnóstico | ✅ Activo |
| `can_access_sensor(sensor_id: int)` | Legacy | ⚠️ Delega a agnóstico |
| `can_access_series(series_id: str)` | Agnóstico | ✅ Activo |

### Adapters

| Adapter | Tipo | Estado |
|---------|------|--------|
| `TaylorPredictionAdapter` | Legacy | ⚠️ Deprecated → usar `engine.as_port()` |
| `CognitivePredictionAdapter` | Legacy | ⚠️ Deprecated → usar `orchestrator.as_port()` |
| `PredictionEnginePortBridge` | Agnóstico | ✅ Activo |

### Conversión de identidad

| Patrón | Tipo | Estado |
|--------|------|--------|
| `int(series_id) if series_id.isdigit() else 0` | Legacy (inseguro) | ❌ Eliminado |
| `safe_series_id_to_int(series_id)` | Agnóstico (seguro) | ✅ Aplicado en 7 archivos |

---

## Resumen cuantitativo

| Métrica | Valor |
|---------|-------|
| APIs legacy con `DeprecationWarning` | 6 |
| APIs legacy sin warning (pendientes) | 0 |
| APIs agnósticas activas | 15+ |
| Archivos con `sensor_id: int` en domain/ | 2 (`SensorReading`, `SensorWindow`) |
| Archivos con `sensor_id: int` en ml_service/ | ~10 (bridge IoT, correcto) |
| Meta-test de scorecard | `TestMigrationScorecard::test_report_legacy_vs_agnostic_usage` |

---

## Próximos pasos para completar la migración

| # | Acción | Prioridad | Impacto |
|---|--------|-----------|---------|
| 1 | Migrar `StoragePort` a `series_id: str` nativo | Media | Elimina bridges en 3 métodos |
| 2 | Migrar `AuditPort` a `series_id: str` nativo | Media | Elimina bridges en 2 métodos |
| 3 | Mover `SensorReading`/`SensorWindow` a `infrastructure/adapters/iot/` | Media | Domain queda 100% agnóstico |
| 4 | Renombrar `PredictSensorValueUseCase` → `PredictSeriesValueUseCase` | Baja | Cosmético |
| 5 | Eliminar `TaylorPredictionAdapter` y `CognitivePredictionAdapter` | Baja | Tras confirmar 0 call-sites |
| 6 | Eliminar `sensor_ranges.py` | Baja | Tras confirmar `ml_service/` no lo usa |
| 7 | Eliminar `SignalProfile` (cognitive/types.py) | Baja | Tras confirmar 0 call-sites |

---

## Criterio de sunset

Una API legacy se puede **eliminar** cuando:

1. Emite `DeprecationWarning` desde hace ≥ 2 releases
2. El meta-test confirma 0 call-sites internos (excluyendo tests de deprecation)
3. No hay consumidores externos conocidos
4. La alternativa agnóstica está documentada y testeada

---

*Este scorecard se actualiza con cada fase de migración. El meta-test `TestMigrationScorecard` valida automáticamente que la dual interface se mantiene intacta.*
