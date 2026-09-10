# Plasticidad y Aprendizaje Adaptativo

**Última actualización:** 2026-07-03

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

### 3. RegimePlasticityManager (`infrastructure/ml/cognitive/plasticity/`)

- **Estado actual:** Implementado activamente en `regime_plasticity_manager.py` (Fase 11, 2026-09-10).
- **Mecanismo:** Ajuste bayesiano de pesos condicionado por régimen:
  $$w_i^{(t+1)}(\text{régimen}) = w_i^{(t)}(\text{régimen}) \cdot \exp(-\eta \cdot \text{Loss}_i)$$
- **Simplex & Cota:** Normalización a suma 1.0 con `min_weight` (0.05) para evitar inanición permanente.
- **Circuit Breaker:** Si un experto supera `max_consecutive_failures` (default: 4) en un régimen, se inhibe automáticamente con peso cero ($w_i = 0$).
- **Coordinador:** Gestionado por `MetacognitiveCoordinator` (`infrastructure/ml/cognitive/metacognitive_coordinator.py`), que alimenta los pesos actualizados al orquestador MoE.
- **Persistencia:** Implementa el protocolo `StatePersistable` (`export_state` / `import_state`) para snapshots JSON/Redis.

---

## Aprendizaje por Retroalimentación

### 1. Bucle Metacognitivo Post-Mortem (Fase 11)

- `PostMortemEvaluator` en `domain/entities/cognitive/` mantiene una cola circular de predicciones.
- Al cumplirse el horizonte temporal ($H=11$ pasos), evalúa el desenlace real contra las predicciones de cada experto.
- Determina el experto ganador (`winning_expert`), la corrección direccional y clasifica la causa del fallo mediante `FailureTaxonomy`.
- `MetacognitiveTracker` calcula el Índice de Meta-Competencia ($\Omega_t$). Si detecta ceguera correlacionada, eleva el factor $\lambda_t \to 1.0$ forzando abstención (`HOLD`).

### 2. record_actual (Legacy / Percepción)

- `record_actual_handler.py` en `perception/` recibe valores reales y actualiza errores por motor.
- Los errores alimentan el `BayesianWeightTracker` para actualizar posteriors.
- `EngineReliabilityTracker` (Beta-Bernoulli) usa estos errores para decidir exclusión hard en `InhibitionGate`.

### 3. Per-Sensor Learning (Fase 4)

- `SensorProfile` con `hampel_k` y `hampel_window` por equipo permite calibración individual.
- `PredictionDriftDetector` puede llevar `equipment_class` en sus alertas.
- Ver `tests/unit/ml/test_fase4_per_sensor_learning.py`.

---

## Estado vs. Documentación Histórica

| Concepto | Documentado en memoria | Estado en código actual | Notas |
|----------|----------------------|------------------------|-------|
| `RegimePlasticityManager` | Fase 11 | **Implementado y verificado** | Ajuste exponencial simplex + circuit-breakers por régimen |
| `MetacognitiveCoordinator` | Fase 11 | **Implementado y verificado** | Orquestación post-mortem + tracker de meta-competencia $\Omega_t$ |
| `PlasticityTracker` con EMA | Memoria previa | Reemplazado por `RegimePlasticityManager` | Cobertura bayesiana y contextual activa |
| BayesianWeightTracker | README actual | **Verificado** en código | Actualización conjugada Normal-Normal por serie |

---

## Referencias

- `infrastructure/ml/cognitive/plasticity/regime_plasticity_manager.py`
- `infrastructure/ml/cognitive/metacognitive_coordinator.py`
- `domain/entities/cognitive/` (`failure_taxonomy.py`, `post_mortem_evaluator.py`, `metacognitive_tracker.py`)
- `tests/unit/cognitive/test_metacognitive_system.py`
- `infrastructure/ml/cognitive/bayesian_weight_tracker/`
- `infrastructure/ml/cognitive/orchestration/weight_resolution_service.py`
