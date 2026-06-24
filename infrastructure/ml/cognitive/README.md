# Cognitive ML Infrastructure

Sistema cognitivo de ML con meta-cognición, plasticidad adaptativa y explicabilidad estructurada.

## Package Structure (Reorganized 2026-03-20)

### 📁 orchestration/
Core orchestration pipeline with 25+ phase files
- `orchestrator.py` (296 lines) — `MetaCognitiveOrchestrator` main class
- `pipeline_executor.py` — Pipeline execution logic with WeightCache
- `fallback_handler.py` — Fallback handling
- `phases/` — 25+ phase files: action_guard, adapt, assembly, boundary_check, causal, coherence_check, confidence_calibration, context, decision_arbiter, drift_detection, drift_response, explain, fuse, inhibit, memory, narrative_unification, observability, perceive, predict, prediction_readiness_gate, protocols, seasonal_decomposition, shadow_evaluation

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

### 📁 drift/
Online drift detection algorithms
- `page_hinkley.py` — Page-Hinkley test for change detection
- `adwin.py` — ADWIN (Adaptive Windowing) algorithm
- `error_drift_detector.py` — Error-based drift detection

### 📁 compliance/
HMAC-SHA256 audit trail for forensic compliance
- `compliance_exporter.py` — NDJSON append-only exporter
- `compliance_record.py` — Signed prediction record model
- `hmac_key_manager.py` — Key management for HMAC signing

### 📁 narrative/
Unified explanation narrative generation
- `generator.py` — Narrative generator from reasoning traces
- `phrase_bank.py` — Domain-specific phrase templates
- `embedding_network.py` — Embedding-based narrative selection
- `layers.py` — Neural layers for narrative assembly

### 📁 decision/
Contextual decision engine with multiple strategies
- `contextual_decision_engine.py` — Main decision engine
- `aggressive/`, `conservative/`, `cost_optimized/` — Strategy implementations

### 📁 causal/
Causal correlation between sensor series
- `causal_correlation.py` — Correlation analysis
- `event_propagation.py` — Event propagation graph

### 📁 memory/
Cognitive memory stores for anomaly and operational patterns
- `anomaly_memory_store.py` — Historical anomaly memory
- `operational_memory_pipeline.py` — Operational pattern memory

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
  ├── PERCEIVE       → SignalAnalyzer → StructuralAnalysis
  ├── PREDICT        → engines paralelos → EnginePerception[]
  ├── INHIBIT        → InhibitionGate → pesos suprimidos
  ├── ADAPT          → PlasticityTracker / AdvancedPlasticityCoordinator
  ├── FUSE           → WeightedFusion → valor final
  ├── EXPLAIN        → ExplanationBuilder → domain Explanation
  ├── CONTEXT        → Contexto operacional del sensor
  ├── DRIFT_DETECT   → Page-Hinkley + ADWIN
  ├── DRIFT_RESPONSE → Accion correctiva
  ├── CAUSAL         → Correlacion causal entre series
  ├── MEMORY         → Memoria cognitiva (Weaviate)
  ├── NARRATIVE      → Unificacion de narrativa
  ├── OBSERVABILITY  → Metricas y trazas
  └── SHADOW_EVAL    → Evaluacion en segundo plano
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
├── severity_classifier.py         ← Severity classification
├── README.md
├── orchestration/                 ← Core pipeline
│   ├── orchestrator.py
│   ├── pipeline_executor.py
│   ├── fallback_handler.py
│   └── phases/                    ← 25+ phase files
├── fusion/                        ← Weighted fusion
│   ├── engine_selector.py
│   ├── fusion_phases.py
│   ├── hampel_filter.py
│   ├── weight_mediator.py
│   ├── weight_adjustment_service.py
│   └── contextual_weight_calculator.py
├── inhibition/                    ← Weight suppression
│   ├── gate.py
│   ├── rules.py
│   ├── smart_rules.py
│   └── adaptive_config.py
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
├── drift/                         ← Online drift detection
│   ├── page_hinkley.py
│   ├── adwin.py
│   └── error_drift_detector.py
├── compliance/                    ← HMAC audit trail
│   ├── compliance_exporter.py
│   ├── compliance_record.py
│   └── hmac_key_manager.py
├── narrative/                     ← Narrative generation
│   ├── generator.py
│   ├── phrase_bank.py
│   ├── embedding_network.py
│   └── layers.py
├── decision/                      ← Decision strategies
│   ├── contextual_decision_engine.py
│   ├── aggressive/
│   ├── conservative/
│   └── cost_optimized/
├── causal/                        ← Causal correlation
│   ├── causal_correlation.py
│   └── event_propagation.py
├── memory/                        ← Cognitive memory
│   ├── anomaly_memory_store.py
│   └── operational_memory_pipeline.py
├── neural/                        ← Neural engines
│   ├── hybrid_engine.py
│   ├── attention/
│   ├── classical/
│   ├── competition/
│   ├── pipeline/
│   ├── plasticity/
│   └── snn/
├── regime/                        ← Regime classification
│   ├── classifier.py
│   ├── factory.py
│   ├── heuristic.py
│   └── router.py
├── sanitize/                      ← Data sanitization
│   ├── imputer.py
│   ├── cusum.py
│   ├── phase.py
│   └── bounds_provider.py
├── seasonal/                      ← Seasonal decomposition
│   ├── fft_seasonality.py
│   └── stl_decomposer.py
├── bayesian_weight_tracker/       ← Per-sensor weight learning (33 files)
├── error_store/
├── explainability/
├── hyperparameters/
├── observability/
├── reliability/
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
