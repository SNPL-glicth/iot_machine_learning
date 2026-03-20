# Cognitive ML Infrastructure

Sistema cognitivo de ML con meta-cognición, plasticidad adaptativa y explicabilidad estructurada.

## Package Structure (Reorganized 2026-03-20)

### 📁 orchestration/
Core orchestration pipeline
- `orchestrator.py` (296 lines) — `MetaCognitiveOrchestrator` main class
- `pipeline_executor.py` — Pipeline execution logic with WeightCache
- `fallback_handler.py` — Fallback handling

### 📁 fusion/
Weighted fusion and engine selection
- `engine_selector.py` (113 lines) — `WeightedFusion` main class
- `fusion_phases.py` (181 lines) — INHIBIT, FUSE phases for ExplanationBuilder
- `weight_mediator.py` — Weight conflict resolution
- `weight_adjustment_service.py` (122 lines) — Dynamic weight adjustment
- `contextual_weight_calculator.py` (69 lines) — Pure inverse-MAE weight calculation

### 📁 inhibition/
Engine weight suppression
- `gate.py` (129 lines) — `InhibitionGate` + `InhibitionConfig`
- `rules.py` (132 lines) — Pure inhibition logic

### 📁 explanation/
Domain Explanation object construction
- `builder.py` (168 lines) — `ExplanationBuilder` fluent API
- `explanation_builder.py` (13 lines) — Backward-compatible facade

### 📁 perception/
Engine perception collection and orchestration
- `helpers.py` (108 lines) — `collect_perceptions`, `create_fallback_result`
- `phases.py` (152 lines) — PERCEIVE, FILTER, PREDICT, ADAPT phases
- `phase_setters.py` (32 lines) — Facade for phase setters
- `record_actual_handler.py` (114 lines) — Dispatch legacy/advanced plasticity

### 📁 plasticity/
Regime-contextual weight learning (base + advanced)
- `base.py` (124 lines) — `PlasticityTracker` simple EMA-based
- `factory.py` (49 lines) — Factory for `AdvancedPlasticityCoordinator`
- `advanced_plasticity_coordinator.py` (249 lines) — Coordinates 4 advanced components
- `adaptive_learning_rate.py` (240 lines) — Context-aware learning rates
- `contextual_plasticity_tracker.py` (264 lines) — MAE tracking by context

### 📁 analysis/
Signal analysis and type definitions
- `signal_analyzer.py` (182 lines) — `SignalAnalyzer` → `StructuralAnalysis`
- `types.py` (232 lines) — `EnginePerception`, `InhibitionState`, `PipelineTimer`, `MetaDiagnostic`

### 📁 monitoring/
Engine health monitoring
- `engine_health_monitor.py` (240 lines) — Auto-inhibition by health

### 📁 text/
Text-specific cognitive engine (18 files)

### 📁 universal/
Universal cognitive engines (14 files)

### Root Files
- `cognitive_adapter.py` (71 lines) — Bridge to `PredictionPort`
- `__init__.py` — Public API re-exports
- `README.md` — This file

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

## Folder Structure

```
cognitive/
├── __init__.py                    ← Public API
├── cognitive_adapter.py           ← PredictionPort bridge
├── README.md
├── orchestration/                 ← Core pipeline
│   ├── orchestrator.py
│   ├── pipeline_executor.py
│   └── fallback_handler.py
├── fusion/                        ← Weighted fusion
│   ├── engine_selector.py
│   ├── fusion_phases.py
│   ├── weight_mediator.py
│   ├── weight_adjustment_service.py
│   └── contextual_weight_calculator.py
├── inhibition/                    ← Weight suppression
│   ├── gate.py
│   └── rules.py
├── explanation/                   ← Explanation builder
│   ├── builder.py
│   └── explanation_builder.py    ← facade
├── perception/                    ← Perception collection
│   ├── helpers.py
│   ├── phases.py
│   ├── phase_setters.py          ← facade
│   └── record_actual_handler.py
├── plasticity/                    ← Weight learning
│   ├── base.py
│   ├── factory.py
│   ├── advanced_plasticity_coordinator.py
│   ├── adaptive_learning_rate.py
│   └── contextual_plasticity_tracker.py
├── analysis/                      ← Signal + types
│   ├── signal_analyzer.py
│   └── types.py
├── monitoring/                    ← Health monitor
│   └── engine_health_monitor.py
├── text/                          ← Text engine (18 files)
└── universal/                     ← Universal engines (14 files)
```

## Import Examples

```python
# Public API (unchanged)
from infrastructure.ml.cognitive import (
    MetaCognitiveOrchestrator,
    PlasticityTracker,
    InhibitionGate,
    WeightedFusion,
    ExplanationBuilder,
)

# Subpackage imports (new paths)
from infrastructure.ml.cognitive.fusion import WeightedFusion
from infrastructure.ml.cognitive.inhibition import InhibitionGate, InhibitionConfig
from infrastructure.ml.cognitive.plasticity import PlasticityTracker, build_advanced_plasticity
from infrastructure.ml.cognitive.explanation import ExplanationBuilder
```
