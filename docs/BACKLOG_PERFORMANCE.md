# Performance Backlog

Items sin infraestructura o datos disponibles para implementación inmediata.

## Circuit breaker para storage lento

- **Ubicación:** `infrastructure/adapters/sql_server_storage_adapter.py`
- **Approach:** Wrap storage con `CircuitBreakerStorageAdapter` que detecta timeouts consecutivos y abre el circuito
- **Prerequisito:** librería `pybreaker` o implementación propia basada en contador de fallos
- **Effort:** 2–3 días
- **Status:** BACKLOG — flag `ML_ENABLE_CIRCUIT_BREAKER` eliminado en audit Fase 3 por ser huérfano

## Connection pooling Redis/MLflow

- **Ubicación:** `ml_service/runners/wiring/container.py`
- **Approach:** `redis.ConnectionPool` con `max_connections` configurable por flag
- **Prerequisito:** `ML_REDIS_MAX_CONNECTIONS` en config
- **Effort:** 1 día
- **Status:** BACKLOG

## Batch prediction API

- **Ubicación:** `ml_service/api/routes.py`
- **Approach:** `POST /ml/predict/batch` con lista de `PredictionRequest`; parallelizar con `ThreadPoolExecutor`
- **Prerequisito:** Test de carga concurrente (ya existe `test_cognitive_pipeline_100_sensors_concurrent` como referencia)
- **Effort:** 2–3 días
- **Status:** BACKLOG

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
