## Development Guide

**Última actualización:** 2026-05-12

### Adding a New Prediction Engine

```python
from infrastructure.ml.interfaces import PredictionEngine

class MyCustomEngine(PredictionEngine):
    @property
    def name(self) -> str:
        return "my_custom"
    
    def predict(self, window: TimeSeriesWindow) -> PredictionResult:
        # Your prediction logic here
        return PredictionResult(
            predicted_value=self._compute(window),
            confidence=0.8,
            trend="stable"
        )
```

Register in orchestrator:

```python
orchestrator = MetaCognitiveOrchestrator(
    engines=[TaylorEngine(), BaselineEngine(), MyCustomEngine()],
    enable_plasticity=True
)
```

### Adding a New Tool

```python
from domain.tools.tool import Tool
from domain.tools.tool_guard import SafetyLevel

class MyCustomTool(Tool):
    @property
    def name(self) -> str:
        return "custom_action"
    
    @property
    def parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            }
        }
    
    def can_execute(self, context) -> GuardResult:
        # Return AUTO, ASK, or DENY
        return GuardResult(SafetyLevel.AUTO)
    
    def execute(self, params, context):
        # Execution logic
        return ToolResult(success=True, data={})
```

---

## Current Limitations

### Known Constraints

**Heuristic Tool Mapping:**
- DecisionEngine uses rule-based mapping from predictions to tool calls
- No learned policy optimization (e.g., RL) yet
- Mapping rules are hardcoded per domain

**Limited Refinement Strategies:**
- Iterative loop has placeholder `_refine()` method
- Currently returns input unchanged
- Future: window expansion, outlier removal, smoothing

**No Full Reinforcement Learning:**
- Plasticity uses Bayesian updates (supervised-style)
- No trial-and-error learning from action outcomes
- No exploration vs exploitation tradeoff

**Single-Node Plasticity:**
- Redis shared state works for multi-worker
- No distributed consensus for weight updates
- Edge deployment requires state synchronization

**Anomaly Integration:**
- Anomaly detection runs separate from prediction pipeline
- No unified narrative between prediction and anomaly
- Narrative unification is placeholder-only

### Performance Limits

| Metric | Tested Limit | Theoretical Limit |
|--------|---------------|-------------------|
| Sensors | 1,000 | 10,000 (with sharding) |
| Latency (p99) | 150ms | 50ms (with caching) |
| Throughput | 10K msgs/sec | 100K (with batching) |
| Plasticity regimes | 10 | 100 (with LRU tuning) |

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **Scalability** | Horizontal scaling, partitioned sliding windows, async DB writes | Planned |
| **Distributed Learning** | Federated plasticity, model compression for edge, A/B testing | Planned |
| **Causal Reasoning** | DoWhy integration, counterfactual explanations | Future |
| **Natural Language Interface** | Text commands to tool execution | Future |

For detailed design decisions, see `docs/ARCHITECTURE.md`.
