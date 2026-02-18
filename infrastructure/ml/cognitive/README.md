# Cognitive ML Infrastructure

Sistema cognitivo de ML con meta-cognición, plasticidad adaptativa y explicabilidad estructurada.

## Módulos

### Orquestación
| Archivo | Líneas | Responsabilidad |
|---|---|---|
| `orchestrator.py` | 296 | `MetaCognitiveOrchestrator` — pipeline principal |
| `orchestrator_helpers.py` | 108 | `collect_perceptions`, `create_fallback_result` |
| `record_actual_handler.py` | 114 | Dispatch legacy vs advanced plasticity en `record_actual` |
| `plasticity_factory.py` | 49 | Factory para `AdvancedPlasticityCoordinator` |

### Señal y percepción
| Archivo | Líneas | Responsabilidad |
|---|---|---|
| `signal_analyzer.py` | 182 | `SignalAnalyzer` → `StructuralAnalysis` |
| `types.py` | 232 | `EnginePerception`, `InhibitionState`, `PipelineTimer`, `MetaDiagnostic` |

### Inhibición y fusión
| Archivo | Líneas | Responsabilidad |
|---|---|---|
| `inhibition.py` | 129 | `InhibitionGate` — supresión de engines inestables |
| `inhibition_rules.py` | 132 | Reglas puras: `evaluate_inhibition`, `build_health_summary` |
| `engine_selector.py` | 113 | `WeightedFusion` — fusión ponderada |

### Plasticidad adaptativa
| Archivo | Líneas | Responsabilidad |
|---|---|---|
| `plasticity.py` | 124 | `PlasticityTracker` — tracking de errores por régimen |
| `weight_adjustment_service.py` | 122 | `WeightAdjustmentService` — ajuste de pesos |
| `advanced_plasticity_coordinator.py` | 249 | Coordina los 4 componentes avanzados |
| `adaptive_learning_rate.py` | 240 | `AdaptiveLearningRate` — LR dinámico |
| `contextual_plasticity_tracker.py` | 264 | `ContextualPlasticityTracker` — MAE por contexto |
| `contextual_weight_calculator.py` | 69 | Cálculo puro de inverse-MAE weights |
| `engine_health_monitor.py` | 240 | `EngineHealthMonitor` — auto-inhibición por salud |

### Explicabilidad
| Archivo | Líneas | Responsabilidad |
|---|---|---|
| `builder.py` | 168 | `ExplanationBuilder` — clase principal (fluent API) |
| `explanation_builder.py` | 13 | Facade backward-compatible → `builder.py` |
| `phase_setters.py` | 32 | Facade → `perception_phases.py` + `fusion_phases.py` |
| `perception_phases.py` | 152 | Fases PERCEIVE, FILTER, PREDICT, ADAPT |
| `fusion_phases.py` | 181 | Fases INHIBIT, FUSE, fallback, audit_trace_id |

### Adaptador
| Archivo | Líneas | Responsabilidad |
|---|---|---|
| `cognitive_adapter.py` | 71 | `CognitivePredictionAdapter` — bridge a `PredictionPort` |

## Pipeline cognitivo

```
predict(values)
  ├── PERCEIVE  → SignalAnalyzer → StructuralAnalysis
  ├── PREDICT   → engines paralelos → EnginePerception[]
  ├── INHIBIT   → InhibitionGate → pesos suprimidos
  ├── ADAPT     → PlasticityTracker / AdvancedPlasticityCoordinator
  ├── FUSE      → WeightedFusion → valor final
  └── EXPLAIN   → ExplanationBuilder → domain Explanation
```

## Regla de complejidad

`orchestrator.py` debe mantenerse ≤ 300 líneas. El meta-test en
`tests/unit/infrastructure/test_pipeline_timer_and_guards.py` lo verifica automáticamente.

## Uso

```python
from infrastructure.ml.cognitive import MetaCognitiveOrchestrator

orc = MetaCognitiveOrchestrator(engines=[taylor, baseline])
result = orc.predict(values=[20.1, 20.3, 20.5], series_id="sensor_42")
explanation = orc.last_explanation  # domain Explanation value object
```

## Estructura de archivos

```
cognitive/
├── orchestrator.py
├── orchestrator_helpers.py
├── record_actual_handler.py
├── plasticity_factory.py
├── signal_analyzer.py
├── types.py
├── inhibition.py
├── inhibition_rules.py
├── engine_selector.py
├── plasticity.py
├── weight_adjustment_service.py
├── advanced_plasticity_coordinator.py
├── adaptive_learning_rate.py
├── contextual_plasticity_tracker.py
├── contextual_weight_calculator.py
├── engine_health_monitor.py
├── builder.py
├── explanation_builder.py   ← facade
├── phase_setters.py         ← facade
├── perception_phases.py
├── fusion_phases.py
├── cognitive_adapter.py
└── __init__.py
```
