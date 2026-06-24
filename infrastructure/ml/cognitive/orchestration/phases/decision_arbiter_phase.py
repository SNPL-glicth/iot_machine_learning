"""Decision Arbiter Phase — real profile_engine injection fixes Rules 3-4.

Previously profile_engine == fusion_engine == ctx.selected_engine, which
meant Rules 3 and 4 could never fire.  Now profile_engine is sourced from:
1. ctx.sensor_profile.preferred_engine (if available)
2. Equipment-class inference (TEMPERATURE→statistical, PRESSURE→kalman, etc.)
3. Default → fusion_engine (no override)

Returns authority: flags|profile|fusion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from . import PipelineContext

from iot_machine_learning.domain.services.prediction.engine_decision_arbiter import (
    EngineDecisionArbiter,
)

logger = logging.getLogger(__name__)

# Equipment class → preferred engine mapping
_EQUIPMENT_ENGINE_MAP: Dict[str, str] = {
    "TEMPERATURE": "statistical",
    "PRESSURE": "kalman",
    "CYCLIC": "statistical",
    "LEVEL": "taylor",
    "VIBRATION": "kalman",
}


def _resolve_profile_engine(ctx: PipelineContext) -> str:
    """Determine the real profile_engine from sensor context.

    Priority:
    1. sensor_profile.preferred_engine (if set)
    2. equipment_class → engine mapping
    3. fusion_engine (no override)
    """
    fc = getattr(ctx, "feature_context", None)
    if fc is not None:
        sp = getattr(fc, "sensor_profile", None)
        if sp is not None:
            preferred = getattr(sp, "preferred_engine", None)
            if preferred:
                return preferred
            eq = getattr(sp, "equipment_class", None)
            if eq is not None:
                eq_str = eq.value if hasattr(eq, "value") else str(eq)
                mapped = _EQUIPMENT_ENGINE_MAP.get(eq_str.upper())
                if mapped:
                    logger.debug(
                        "profile_engine_from_equipment_class",
                        extra={"equipment_class": eq_str, "engine": mapped},
                    )
                    return mapped

    # Try direct attribute access
    sp = getattr(ctx, "sensor_profile", None)
    if sp is not None:
        preferred = getattr(sp, "preferred_engine", None)
        if preferred:
            return preferred
        eq = getattr(sp, "equipment_class", None)
        if eq is not None:
            eq_str = eq.value if hasattr(eq, "value") else str(eq)
            mapped = _EQUIPMENT_ENGINE_MAP.get(eq_str.upper())
            if mapped:
                return mapped

    return "fusion"


class DecisionArbiterPhase:
    """Phase 6: Engine decision arbitration with real profile_engine."""

    @property
    def name(self) -> str:
        return "decision_arbiter"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            flags = ctx.flags
            flag_engine = (
                flags.get_active_engine_for_series(ctx.series_id)
                if hasattr(flags, "get_active_engine_for_series")
                else "unknown"
            )

            # Real profile_engine (not ctx.selected_engine!)
            profile_engine = _resolve_profile_engine(ctx)

            # Fusion engine from context
            fusion_engine = ctx.selected_engine or "fusion"

            arbiter = EngineDecisionArbiter()
            engine_decision = arbiter.arbitrate(
                flag_engine=flag_engine,
                profile_engine=profile_engine,
                fusion_engine=fusion_engine,
                series_id=ctx.series_id,
                rollback_to_baseline=getattr(flags, "ML_ROLLBACK_TO_BASELINE", False),
                series_overrides=getattr(flags, "ML_ENGINE_SERIES_OVERRIDES", {}),
            )

            logger.info(
                "engine_decision_arbitrated",
                extra={
                    "series_id": ctx.series_id,
                    "chosen_engine": engine_decision.chosen_engine,
                    "authority": engine_decision.authority,
                    "reason": engine_decision.reason,
                    "profile_engine": profile_engine,
                    "fusion_engine": fusion_engine,
                },
            )

            return ctx.with_field(engine_decision=engine_decision)

        except Exception as e:
            logger.debug(f"engine_decision_arbiter_skipped: {e}")
            return ctx
