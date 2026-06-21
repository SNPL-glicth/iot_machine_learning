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

        # Fase 2: sensor profile + structural engine filter
        sensor_profile = None
        repo = getattr(orchestrator, '_sensor_profile_repository', None)
        if repo and ctx.series_id != "unknown":
            try:
                sensor_profile = repo.get_by_series_id(ctx.series_id)
            except Exception as e:
                logger.debug(f"predict_phase_profile_load_failed: {e}")

        ineligible: set[str] = set()
        if sensor_profile is not None:
            ef = getattr(orchestrator, '_engine_filter', None)
            if ef:
                ineligible = ef.get_ineligible_engines(sensor_profile.equipment_class)
                if ineligible:
                    logger.debug(
                        "structural_engine_filter",
                        extra={"series": ctx.series_id, "eq": sensor_profile.equipment_class.value, "in": list(ineligible)},
                    )

        engines = [e for e in orchestrator._engines if e.name not in ineligible]

        # COG-SEV-1: Check budget BEFORE executing engines
        if ctx.timer.total_ms > ctx.timer.budget_ms:
            logger.warning("pipeline_budget_exceeded_before_predict", extra={
                "phase": "predict",
                "elapsed_ms": round(ctx.timer.total_ms, 2),
                "budget_ms": ctx.timer.budget_ms,
                "reason": "budget_exceeded_before_engine_execution",
            })
            result, diag, expl, reg, perc = handle_fallback(
                ctx.values, ctx.profile, builder, ctx.timer, "budget_exceeded"
            )
            return ctx.with_field(
                is_fallback=True,
                fallback_reason="budget_exceeded_before_predict",
                diagnostic=diag,
                explanation=expl,
                engine_failures={},  # No engines executed
                metadata={
                    "cognitive_diagnostic": diag.to_dict() if diag else None,
                    "explanation": expl.to_dict() if expl else None,
                    "engine_failures": {},
                },
            )

        # Collect perceptions (IMP-2: runs in parallel when possible)
        perceptions = collect_perceptions(engines, ctx.values, ctx.timestamps)
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

        # Cold start: blend equipment-aware weights when data is scarce
        COLD_START_THRESHOLD = 50
        n_points = len(ctx.values or [])
        plasticity_weights = None
        if sensor_profile is not None and n_points < COLD_START_THRESHOLD:
            wi = getattr(orchestrator, '_weight_initializer', None)
            if wi:
                names = [p.engine_name for p in perceptions]
                cold = wi.get_initial_weights(
                    sensor_profile.equipment_class, [n for n in names if n not in ineligible]
                )
                base = {}
                wr = getattr(orchestrator, '_weight_resolver', None)
                if wr and hasattr(wr, '_base_weights'):
                    base = {k: v for k, v in wr._base_weights.items() if k in names}
                for n in names:
                    base.setdefault(n, 1.0 / len(names))
                b = n_points / COLD_START_THRESHOLD
                for n in base:
                    if n in cold:
                        base[n] = (1 - b) * cold[n] + b * base[n]
                plasticity_weights = base
                logger.debug(
                    "cold_start_weights",
                    extra={"series": ctx.series_id, "n": n_points, "blend": round(b, 2)},
                )

        builder.set_perceptions(perceptions, n_engines_total=len(engines))

        # Record prediction metrics (Phase 3C)
        if ctx.metrics_collector is not None:
            try:
                for perception in perceptions:
                    ctx.metrics_collector.record_retrieval(
                        hit=True,
                        similarity=perception.confidence,
                    )
            except Exception as e:
                logger.debug(f"metrics_collection_failed: {e}")

        return ctx.with_field(
            perceptions=perceptions,
            explanation=builder,
            engine_failures=engine_failures,
            plasticity_weights=plasticity_weights,
        )
