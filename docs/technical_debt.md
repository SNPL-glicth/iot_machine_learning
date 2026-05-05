# Deuda Técnica Documentada

**Última actualización:** 2026-05-04
**Regla:** Este archivo se actualiza con cada PR que cierra o abre deuda.

> La deuda técnica documentada es preferible a la deuda oculta.
> Este archivo se actualiza con cada PR que cierra o abre deuda.

---

## Inventario de Deuda Técnica

| # | Ítem | Severidad | Ubicación | Impacto | Estado | ETA estimado |
|---|------|-----------|-----------|---------|--------|--------------|
| 1 | `sigma2_obs` hardcodeado en `BayesianUpdater` | Baja | `infrastructure/ml/inference/bayesian/posterior.py` | Todos los motores compartían σ²=1.0, distorsionando updates para sensores de escala distinta | **RESUELTO (2026-05-04)** | — |
| 2 | `amnesic_mode` descrito incorrectamente en chatflow | Baja | Chatflow JSON `systemMessagePrompt` | Operadores confundían fallo de persistencia con fallo de aprendizaje | **RESUELTO (2026-05-04)** | — |
| 3 | Bridge `series_id → int` con fallback a 0 | Media | `domain/ports/storage_port.py`, `domain/ports/audit_port.py` | `series_id` no numérico (ej. `"room_temp"`) se mapea a `sensor_id=0`, causando colisiones | Abierto | T16 (fase 3) |
| 4 | `ContextualDecisionEngine` importa de `ml_service` | Media | `infrastructure/ml/cognitive/decision/contextual_decision_engine.py` | Violación de dependencia hexagonal; `infrastructure/` no debe importar de `ml_service/` | Abierto | T16 |
| 5 | 214+ tests preexistentes fallando | Alta | `tests/` múltiples | Suite completa no pasa; dificulta CI/CD confiable | Parcial | T17 |
| 6 | MoE Gateway stub | Media | `infrastructure/ml/moe/` | Mixture of Experts registrado pero no integrado en pipeline numérico | Abierto | T18 |
| 7 | Shadow mode no implementado en pipeline numérico | Media | `infrastructure/ml/cognitive/orchestrator.py` | No hay forma de comparar nueva versión vs baseline sin afectar producción | Abierto | T19 |
| 8 | Multivariate score placeholder (parcialmente resuelto) | Media | `infrastructure/ml/anomaly/core/detector.py` | `MultivariateDetector` existe pero activación requiere flag + 3+ series correlacionadas; poco probado en producción | Parcial | T20 |
| 9 | Falta benchmark público NAB/Yahoo S5 | Alta | `tests/benchmark/` vacío | No hay validación externa de la calidad de detección de anomalías | Abierto | T21 |
| 10 | Memory leak en `SlidingWindowStore` | Alta | `ml_service/consumers/sliding_window.py` | Nunca evita sensores inactivos; crece indefinidamente con más sensores | Abierto | T22 |
| 11 | Batch runner secuencial para >500 sensores | Media | `ml_service/runners/ml_batch_runner.py` | Ciclo >60s con 500+ sensores → overlap entre ejecuciones | Abierto | T23 |
| 12 | Duplicación de predicciones stream + batch | Alta | `ml_service/consumers/stream_consumer.py` + `ml_service/runners/ml_batch_runner.py` | Ambos predicen para el mismo sensor sin deduplicación | Abierto | T24 |
| 13 | 5 modelos "Reading" diferentes entre servicios | Media | `domain/entities/iot/`, `ml_service/consumers/`, etc. | 5 implementaciones distintas del mismo concepto generan inconsistencias | Abierto | T25 |
| 14 | `ThresholdPolicy` desconectado de `ContextualDecisionEngine` | Media | `domain/policies/threshold_policy.py` + `infrastructure/ml/cognitive/decision/contextual_decision_engine.py` | Ajustar umbrales de uno no afecta al otro; requiere tuning manual duplicado | Abierto | T26 |
| 15 | NIS2 no implementado | Alta | — | Directiva en transposición; requerirá notificación de incidentes y gestión de riesgos de cadena de suministro | Abierto | 2027+ |
| 16 | IEC 62443 sin evaluación formal | Media | — | Seguridad industrial no auditada por terceros | Abierto | 2027 |

---

## Detalle de Ítems Críticos

### #5 — 214+ Tests Fallando

**Síntoma:** `pytest` reporta ~214 fallos en la suite completa.

**Causas identificadas:**
- Tests pre-existentes que rompieron con refactorings recientes (dual interface series_id, reorganización de fases).
- Tests de integración que dependen de servicios externos (Redis, SQL Server) no levantados.
- Meta-tests que fallan por cambios de estructura (ej. `test_pipeline_phase_order.py` ahora requiere índices +1 por IMP-1).

**Plan:**
1. Separar tests en "core" (siempre verdes) vs "integration" (requieren infraestructura).
2. CI/CD: job `test-core` obligatorio; `test-integration` informativo.
3. ETA: Sprint T17.

---

### #10 — Memory Leak SlidingWindowStore

**Síntoma:** `SlidingWindowStore` mantiene `deque` por `series_id` en memoria. Los sensores desactivados nunca se evictan.

**Impacto:** Con 1000+ sensores, RAM crece indefinidamente. Planta estimada: falla en ~1000 sensores.

**Plan:**
- Agregar TTL por serie (último acceso > N horas → evict).
- Agregar límite total de series (`max_series` con LRU).
- ETA: Sprint T22.

---

### #12 — Duplicación de Predicciones Stream + Batch

**Síntoma:** `ReadingsStreamConsumer` predice cada lectura nueva. `ml_batch_runner` predice cada 5 minutos para todos los sensores. El mismo sensor recibe dos predicciones consecutivas.

**Impacto:** Doble carga computacional, doble escritura a SQL, posible inconsistencia si los datos cambian entre stream y batch.

**Plan:**
- Agregar deduplicación por `(series_id, timestamp_bucket)`.
- Batch runner salta sensores que fueron procesados por stream en los últimos N minutos.
- ETA: Sprint T24.

---

## Ítems Resueltos Recientemente

### `sigma2_obs` hardcodeado → Varianza empírica por motor (2026-05-04)

**Antes:** `BayesianUpdater._update_gaussian()` usaba `sigma2_obs=1.0` fijo.
**Después:** `VarianceEstimator` mantiene deque de 20 errores por motor; calcula varianza empírica; fallback a 1.0 si <5 muestras.
**Archivos modificados:** `posterior.py`, `base.py`, `update_mixin.py`, `constants.py`, `cognitive_config.py`.
**Tests añadidos:** 4 tests nuevos en `test_bayesian_weight_tracker_sigma2.py`.

### `amnesic_mode` semántica corregida (2026-05-04)

**Antes:** Descrito como "sistema no está aprendiendo".
**Después:** Descrito como "fallo de persistencia activo — opera solo en RAM".
**Estado:** Pendiente aplicación en JSON del chatflow (archivo no localizado en workspace).
