# Plasticidad y Aprendizaje Adaptativo

**Última actualización:** 2026-05-17

---

## Resumen

La plasticidad en ZENIN se refiere a la capacidad del sistema de ajustar los pesos de los motores predictivos según el régimen de la señal y el historial de error por serie. No es un módulo independiente activo, sino un concepto distribuido entre varios componentes.

---

## Componentes Relacionados

### 1. BayesianWeightTracker (`infrastructure/ml/cognitive/bayesian_weight_tracker/`)

- **Archivos:** 28 módulos (adaptive LR, contextual, drift, checkpoint, posterior_cache, reset, update, weights mixins)
- **Función:** Update conjugado normal-normal con prior gaussiano N(μ,σ²)
- **σ²_obs empírica:** Ventana de 20 errores, mínimo 5 muestras, clamp a 0.01
- **Evicción:** LRU de 10 régimes, TTL 24h
- **Uso:** Consumido por `WeightResolutionService` para adaptar pesos base antes de la inhibición

### 2. WeightResolutionService (`infrastructure/ml/cognitive/orchestration/weight_resolution_service.py`)

- **Flujo:** Base weights → plasticity adaptation → inhibition → final weights
- **Propósito:** Single source of truth para resolución de pesos. Consolidado del Phase 3 refactor.
- **Nota:** La "plasticity adaptation" en este contexto es la aplicación del BayesianWeightTracker, no un tracker de plasticidad independiente.

### 3. Plasticity Base (`infrastructure/ml/cognitive/plasticity/`)

- **Estado actual:** Solo `base.py` (305 bytes) + `__init__.py`. Stub de abstracción.
- **Historia:** El `PlasticityTracker` documentado en memorias previas (EMA de inverse error por régimen) no está presente en el código actual del master branch.
- **Impacto:** La funcionalidad de plasticity está cubierta por el BayesianWeightTracker; no hay regresión funcional.

---

## Aprendizaje por Retroalimentación

### record_actual

- `record_actual_handler.py` en `perception/` recibe valores reales y actualiza errores por motor
- Los errores alimentan el BayesianWeightTracker para actualizar posteriors
- `EngineReliabilityTracker` (Beta-Bernoulli) usa estos errores para decidir exclusión hard en InhibitionGate

### Per-Sensor Learning (Fase 4)

- `SensorProfile` con `hampel_k` y `hampel_window` por equipo permite calibración individual
- `PredictionDriftDetector` puede llevar `equipment_class` en sus alertas
- Ver `tests/unit/ml/test_fase4_per_sensor_learning.py`

---

## Estado vs. Documentación Histórica

| Concepto | Documentado en memoria | Estado en código actual | Notas |
|----------|----------------------|------------------------|-------|
| `PlasticityTracker` con EMA por régimen | Memoria sistema (a8f92ae8) | **No encontrado** en master | Funcionalidad cubierta por BayesianWeightTracker |
| `PlasticityTracker` con inverse error | Memoria sistema (b497109f) | **No encontrado** en master | Funcionalidad cubierta por BayesianWeightTracker |
| `plasticity/` package | Memoria sistema (b497109f) | Solo `base.py` stub | El package completo no fue mergeado o fue removido |
| BayesianWeightTracker | README actual | **Verificado** en código | Implementación activa de adaptación de pesos |

---

## Referencias

- `infrastructure/ml/cognitive/bayesian_weight_tracker/`
- `infrastructure/ml/cognitive/orchestration/weight_resolution_service.py`
- `tests/unit/infrastructure/ml/cognitive/test_bayesian_weight_tracker_sigma2.py`
- `tests/unit/infrastructure/test_engine_reliability_tracker.py`
