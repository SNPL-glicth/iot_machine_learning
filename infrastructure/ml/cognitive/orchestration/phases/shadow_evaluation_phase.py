"""Shadow Evaluation Phase — Fase 5 Scalability.

Executes experimental engines in shadow mode, recording their
performance without affecting the final prediction response.

R-7: Safe experimentation without user impact.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Any

if TYPE_CHECKING:
    from . import PipelineContext

from ...interfaces import PredictionEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShadowResult:
    """Result of a shadow engine execution."""
    engine_name: str
    predicted_value: float
    confidence: float
    latency_ms: float
    error_vs_actual: Optional[float] = None


@dataclass
class ShadowEvaluationSummary:
    """Summary of all shadow evaluations for a prediction."""
    results: List[ShadowResult] = field(default_factory=list)
    best_engine: Optional[str] = None
    worst_engine: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [
                {
                    "engine": r.engine_name,
                    "predicted": round(r.predicted_value, 4),
                    "confidence": round(r.confidence, 3),
                    "latency_ms": round(r.latency_ms, 2),
                    "error": round(r.error_vs_actual, 4) if r.error_vs_actual else None,
                }
                for r in self.results
            ],
            "best_engine": self.best_engine,
            "worst_engine": self.worst_engine,
        }


class ShadowEvaluationPhase:
    """Phase X: Shadow mode evaluation of experimental engines.
    
    Runs experimental engines without affecting the final prediction.
    Records performance metrics for offline analysis and A/B testing.
    """
    
    def __init__(
        self,
        shadow_engines: List[PredictionEngine],
        enabled: bool = True,
        sample_rate: float = 1.0,
    ) -> None:
        self._engines = shadow_engines
        self._enabled = enabled
        self._sample_rate = sample_rate
    
    @property
    def name(self) -> str:
        return "shadow_eval"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute shadow evaluation without modifying main prediction."""
        if not self._enabled or not self._engines:
            return ctx.with_field(experimental_metadata={"shadow": None})
        
        # Sampling: only run shadow mode on subset of predictions
        if self._sample_rate < 1.0:
            import random
            if random.random() > self._sample_rate:
                return ctx.with_field(experimental_metadata={"shadow": "skipped"})
        
        values = ctx.values if hasattr(ctx, 'values') else []
        timestamps = ctx.timestamps if hasattr(ctx, 'timestamps') else None
        
        shadow_results: List[ShadowResult] = []
        
        for engine in self._engines:
            if not engine.can_handle(len(values)):
                continue
            
            try:
                start = time.perf_counter()
                result = engine.predict(values, timestamps)
                latency_ms = (time.perf_counter() - start) * 1000
                
                shadow_results.append(ShadowResult(
                    engine_name=engine.name,
                    predicted_value=result.predicted_value,
                    confidence=result.confidence,
                    latency_ms=latency_ms,
                ))
                
                logger.debug(
                    "shadow_engine_executed",
                    extra={
                        "engine": engine.name,
                        "latency_ms": round(latency_ms, 2),
                        "series_id": ctx.series_id if hasattr(ctx, 'series_id') else 'unknown',
                    },
                )
                
            except Exception as e:
                logger.warning(
                    "shadow_engine_failed",
                    extra={"engine": engine.name, "error": str(e)},
                )
        
        # Build summary
        summary = ShadowEvaluationSummary(results=shadow_results)
        
        # Rank engines by error if actual value available
        if ctx.fused_value is not None:
            ranked = sorted(
                shadow_results,
                key=lambda r: abs(r.predicted_value - ctx.fused_value),
            )
            if ranked:
                summary.best_engine = ranked[0].engine_name
                summary.worst_engine = ranked[-1].engine_name
        
        # Inject into experimental_metadata (immutable ctx extension)
        experimental = {
            "shadow": summary.to_dict(),
            "shadow_engines_tested": len(shadow_results),
        }
        
        return ctx.with_field(experimental_metadata=experimental)
