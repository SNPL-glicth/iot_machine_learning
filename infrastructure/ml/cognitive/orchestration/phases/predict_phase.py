"""Predict Phase — MED-1 Refactoring.

Collects perceptions from all engines and handles fallback.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import PipelineContext

from ...explanation import ExplanationBuilder
from ...perception.helpers import collect_perceptions, consume_engine_failures
from ..fallback_handler import handle_fallback

logger = logging.getLogger(__name__)


class PredictPhase:
    """Phase 2: Engine perception collection."""
    
    @property
    def name(self) -> str:
        return "predict"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute prediction phase."""
        orchestrator = ctx.orchestrator
        
        # Create explanation builder
        builder = ExplanationBuilder(ctx.series_id)
        if ctx.profile:
            builder.set_signal(ctx.profile)
        
        # Collect perceptions (IMP-2: runs in parallel when possible)
        perceptions = collect_perceptions(orchestrator._engines, ctx.values, ctx.timestamps)
        engine_failures = consume_engine_failures()
        
        # Handle no valid perceptions
        if not perceptions:
            result, diag, expl, reg, perc = handle_fallback(
                ctx.values, ctx.profile, builder, ctx.timer, "no_valid_perceptions"
            )
            return ctx.with_field(
                is_fallback=True,
                fallback_reason="no_valid_perceptions",
                diagnostic=diag,
                explanation=expl,
                engine_failures=engine_failures,
                metadata={
                    "cognitive_diagnostic": diag.to_dict() if diag else None,
                    "explanation": expl.to_dict() if expl else None,
                    "engine_failures": engine_failures,
                },
            )
        
        # Check budget
        if ctx.timer.total_ms > ctx.timer.budget_ms:
            logger.warning("pipeline_budget_exceeded", extra={
                "phase": "predict",
                "elapsed_ms": round(ctx.timer.total_ms, 2),
                "budget_ms": ctx.timer.budget_ms,
            })
            result, diag, expl, reg, perc = handle_fallback(
                ctx.values, ctx.profile, builder, ctx.timer, "budget_exceeded"
            )
            return ctx.with_field(
                is_fallback=True,
                fallback_reason="budget_exceeded",
                diagnostic=diag,
                explanation=expl,
                engine_failures=engine_failures,
                metadata={
                    "cognitive_diagnostic": diag.to_dict() if diag else None,
                    "explanation": expl.to_dict() if expl else None,
                    "engine_failures": engine_failures,
                },
            )
        
        builder.set_perceptions(perceptions, n_engines_total=len(orchestrator._engines))
        
        return ctx.with_field(
            perceptions=perceptions,
            explanation=builder,
            engine_failures=engine_failures,
        )
