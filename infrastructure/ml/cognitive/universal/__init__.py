"""Universal cognitive engines — input-agnostic deep analysis.

Provides two engines:

1. **UniversalAnalysisEngine** — Perceive → Analyze → Remember → Reason → Explain
   - Handles text, numbers, CSV, JSON, special chars
   - Reuses PlasticityTracker, InhibitionGate, WeightedFusion, ExplanationBuilder
   - Produces Explanation domain object

2. **UniversalComparativeEngine** — Compare current vs historical
   - Recalls top 3 similar past analyses
   - Computes severity/urgency/topic deltas
   - Produces human-readable comparison

Both engines learn by DOMAIN (infrastructure, security, trading, operations)
not by data type (text vs numeric).
"""

from .analysis import (
    UniversalAnalysisEngine,
    UniversalInput,
    UniversalResult,
    UniversalContext,
    InputType,
)
from .comparative import (
    UniversalComparativeEngine,
    ComparisonContext,
    ComparisonResult,
)

__all__ = [
    "UniversalAnalysisEngine",
    "UniversalInput",
    "UniversalResult",
    "UniversalContext",
    "InputType",
    "UniversalComparativeEngine",
    "ComparisonContext",
    "ComparisonResult",
]
