# Performance Backlog

Items sin infraestructura o datos disponibles para implementación inmediata.

## Circuit breaker para storage lento

- **Ubicación:** `infrastructure/adapters/sql_server_storage_adapter.py`
- **Approach:** Wrap storage con `CircuitBreakerStorageAdapter` que detecta timeouts consecutivos y abre el circuito
- **Prerequisito:** librería `pybreaker` o implementación propia basada en contador de fallos
- **Effort:** 2–3 días
- **Status:** BACKLOG — flag `ML_ENABLE_CIRCUIT_BREAKER` eliminado en audit Fase 3 por ser huérfano

## Connection pooling Redis/MLflow

- **Ubicación:** `infrastructure/persistence/redis/pools.py`
- **Approach:** `redis.ConnectionPool` con `max_connections` configurable por env
- **Prerequisito:** `REDIS_MAX_CONNECTIONS` en env
- **Effort:** 1 día
- **Status:** ✅ DONE — pools.py has general (150), stream (50), async pools.
  Configurable via `REDIS_MAX_CONNECTIONS`, `REDIS_STREAM_MAX_CONNECTIONS`.

## Batch prediction API

- **Ubicación:** `ml_service/api/routes.py`
- **Approach:** `POST /ml/predict/batch` con lista de `PredictionRequest`; parallelizar con `ThreadPoolExecutor`
- **Prerequisito:** Test de carga concurrente (ya existe `test_cognitive_pipeline_100_sensors_concurrent` como referencia)
- **Effort:** 2–3 días
- **Status:** BACKLOG

## PERF-P0: zenin_db pool scaling (DONE)

- **Ubicación:** `infrastructure/persistence/sql/zenin_db_connection.py`
- **Fix:** pool_size 5→20, max_overflow 10→30, configurable via env vars
  `ZENIN_DB_POOL_SIZE`, `ZENIN_DB_MAX_OVERFLOW`, `ZENIN_DB_POOL_TIMEOUT`,
  `ZENIN_DB_POOL_RECYCLE`, `ZENIN_DB_CONNECT_TIMEOUT`
- **Status:** ✅ DONE — max_capacity=50 by default (was 15)

## PERF-P0: Orchestrator thread pool leak (DONE)

- **Ubicación:** `ml_service/runners/adapters/orchestrator_prediction.py`
- **Fix:** Replaced per-call ThreadPoolExecutor(max_workers=1) with shared
  singleton executor. Configurable via `ML_ORCHESTRATOR_WORKERS` (default 4).
- **Status:** ✅ DONE — 0 thread leak under 1000-sensor load

## PERF-P0: Batch runner error isolation (DONE)

- **Ubicación:** `ml_service/runners/ml_batch_runner.py`
- **Fix:** Added try/except around future.result() in parallel mode.
  A failed sensor logs error and continues; does not crash the batch.
  ML_BATCH_PARALLEL_WORKERS default changed from 1 to 8.
- **Status:** ✅ DONE — 1000 sensors with 8 workers < 7s (10ms/sensor sim)

## Warm-start de engines (Taylor)

- **Ubicación:** `infrastructure/ml/engines/taylor/`
- **Approach:** Precómputo de coeficientes por serie, cache en Redis bajo key `taylor:warm:{series_id}`
- **Prerequisito:** Series con histórico ≥ `taylor.min_points`; TTL en cache
- **Effort:** 3–4 días
- **Status:** BACKLOG

## Profiling de memoria ContextStateManager

- **Ubicación:** `infrastructure/ml/cognitive/orchestration/orchestrator.py:129`
- **Approach:** `memory_profiler` + test con 10k series simuladas, detectar leaks en `_state_manager._states`
- **Prerequisito:** Datos sintéticos de 10k series
- **Effort:** 1–2 días
- **Status:** BACKLOG

## DynamicTuner — integración completa en pipeline

- **Ubicación:** `infrastructure/ml/cognitive/orchestration/phases/adapt_phase.py`
- **Approach:** Invocar `tuner.tune_learning_rate()` con error post-predict; persistir alpha en `SensorProfile`
- **Prerequisito:** `BatchEnterpriseContainer.get_dynamic_tuner()` ya existe (Fase 3); falta inyectar en orchestrator
- **Effort:** 2 días
- **Status:** PARCIAL — wiring de container listo, falta invocación en pipeline
