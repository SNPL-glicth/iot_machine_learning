# Reglas Arquitectónicas — UTSAE

**Última actualización:** 2026-02-12  
**Aplica a:** `iot_machine_learning/` completo  
**Enforced by:** Meta-tests en `test_pipeline_timer_and_guards.py`

---

## Regla 1: El orquestador DELEGA, nunca IMPLEMENTA

**Archivo:** `infrastructure/ml/cognitive/orchestrator.py`  
**Límite:** ≤ 300 líneas  
**Meta-test:** `TestOrchestratorComplexityGuard::test_orchestrator_line_count`

El `MetaCognitiveOrchestrator` es el archivo más crítico del sistema. Su única responsabilidad es **coordinar** el pipeline cognitivo. Toda lógica computacional vive en sub-módulos:

| Responsabilidad | Sub-módulo | El orquestador... |
|----------------|------------|-------------------|
| Análisis de señal | `signal_analyzer.py` | Llama `analyze()` |
| Inhibición | `inhibition.py` | Llama `compute()` |
| Plasticidad | `plasticity.py` | Llama `get_weights()`, `update()` |
| Fusión | `engine_selector.py` | Llama `fuse()` |
| Explicación | `explanation_builder.py` | Llama `build()` |
| Timing | `types.PipelineTimer` | Llama `start()`, `stop()` |

### ❌ Prohibido en el orquestador

- Imports de `numpy`, `scipy`, o cualquier librería de cálculo pesado
- Funciones de más de 30 líneas (extraer a sub-módulo)
- Lógica de selección de engine (eso es `engine_selector.py`)
- Lógica de métricas/logging complejo (eso es `MetricsCollector`)
- Cálculos matemáticos directos (eso es `signal_analyzer.py` o domain)

### ✅ Permitido en el orquestador

- Coordinación del pipeline (llamar sub-módulos en orden)
- Budget guard (cortar a fallback si se excede el presupuesto)
- Almacenamiento de estado mínimo (`_last_regime`, `_last_perceptions`)
- Logging estructurado de 1-2 líneas por evento

---

## Regla 2: Presupuesto de latencia del pipeline cognitivo

**Default:** 500ms  
**Configurable:** `MetaCognitiveOrchestrator(budget_ms=...)`  
**Meta-test:** `TestOrchestratorPipelineTiming`

El pipeline cognitivo tiene un **presupuesto de tiempo**. Si las fases PERCEIVE + PREDICT ya exceden el presupuesto, el orquestador corta inmediatamente a fallback en vez de ejecutar INHIBIT → ADAPT → FUSE → EXPLAIN.

```
Pipeline: Perceive → Predict → [budget check] → Adapt → Inhibit → Fuse → Explain
                                     ↓
                              budget_exceeded → Fallback (media móvil)
```

### Fases instrumentadas

| Fase | Qué mide | Componente |
|------|----------|------------|
| `perceive_ms` | Análisis estructural de la señal | `SignalAnalyzer.analyze()` |
| `predict_ms` | Recolección de percepciones de todos los engines | `_collect_perceptions()` |
| `adapt_ms` | Resolución de pesos (plasticidad) | `_resolve_weights()` |
| `inhibit_ms` | Cómputo de inhibición | `InhibitionGate.compute()` |
| `fuse_ms` | Fusión ponderada | `WeightedFusion.fuse()` |
| `explain_ms` | Construcción de `Explanation` | `ExplanationBuilder.build()` |

### Acceso al timing

```python
result = orchestrator.predict(values, timestamps)
timing = result.metadata["pipeline_timing"]
# {"perceive_ms": 0.5, "predict_ms": 12.3, ..., "total_ms": 15.2, "over_budget": false}

# O vía property:
timer = orchestrator.last_pipeline_timing
print(timer.total_ms, timer.is_over_budget)
```

---

## Regla 3: Dependencias apuntan hacia adentro

```
ml_service/ → application/ → domain/
                    ↑
            infrastructure/
```

- `domain/` **nunca** importa de `infrastructure/`, `application/`, o `ml_service/`
- `application/` **nunca** importa de `infrastructure/` o `ml_service/`
- `infrastructure/` importa de `domain/` (dirección correcta)
- `ml_service/` puede importar de cualquier capa interna

---

## Regla 4: Entidades inmutables

Todas las entidades de dominio son `frozen=True`. Para modificar:

```python
from dataclasses import replace
new_prediction = replace(prediction, audit_trace_id=trace_id)
```

**Nunca** reconstruir manualmente un dataclass frozen con todos sus campos.

---

## Regla 5: Conversión segura de identidad

Toda conversión `series_id: str` → `sensor_id: int` usa:

```python
from iot_machine_learning.domain.validators.input_guard import safe_series_id_to_int
sensor_id = safe_series_id_to_int(series_id)
```

**Nunca** usar `int(series_id)` directamente. El meta-test `TestNoRemainingBareIntConversions` lo detecta.

---

## Regla 6: Severidad unificada

`AnomalySeverity.from_score()` es la **fuente única de verdad** para clasificación de severidad. Ningún otro archivo debe hardcodear umbrales de severidad.

```python
from iot_machine_learning.domain.entities.results.anomaly import AnomalySeverity
severity = AnomalySeverity.from_score(0.75)  # → AnomalySeverity.HIGH
```

---

## Regla 7: Extensibilidad sin modificación

Agregar un engine o detector **no requiere modificar archivos existentes**:

```python
@register_engine("my_engine")
class MyEngine(PredictionEngine): ...

@register_detector("my_detector")
def create_my_detector(config): ...
```

---

## Meta-tests que enforzan estas reglas

| Test | Regla | Archivo |
|------|-------|---------|
| `test_orchestrator_line_count` | Regla 1 (≤300 líneas) | `test_pipeline_timer_and_guards.py` |
| `test_orchestrator_no_direct_math` | Regla 1 (sin numpy/scipy) | `test_pipeline_timer_and_guards.py` |
| `test_orchestrator_delegates_to_submodules` | Regla 1 (delegación) | `test_pipeline_timer_and_guards.py` |
| `test_predict_includes_pipeline_timing` | Regla 2 (timing) | `test_pipeline_timer_and_guards.py` |
| `test_no_bare_int_series_id_in_ports` | Regla 5 (safe conversion) | `test_technical_debt_cleanup.py` |
| `test_report_legacy_vs_agnostic_usage` | Scorecard (informativo) | `test_pipeline_timer_and_guards.py` |
