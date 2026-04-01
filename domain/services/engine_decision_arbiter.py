"""Engine decision arbiter — single source of truth for engine selection.

Resolves conflicts between multiple decision layers:
- FeatureFlags (flags layer)
- SeriesProfile-based selection (profile layer)
- WeightedFusion output (fusion layer)

Hierarchy (strict):
1. ML_ROLLBACK_TO_BASELINE → baseline, authority=flags
2. ML_ENGINE_SERIES_OVERRIDES → specified engine, authority=flags
3. Flag != Profile → profile wins, override logged
4. Profile != Fusion → profile wins, override logged
5. Consensus → fusion_engine, authority=fusion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class EngineDecision:
    """Final engine decision with authority trace.

    Attributes:
        chosen_engine: The engine name to use.
        authority: Which layer had final say (flags|profile|fusion).
        reason: Human-readable explanation of decision.
        overrides: List of conflicts that were overridden.
    """

    chosen_engine: str
    authority: str
    reason: str
    overrides: List[str] = field(default_factory=list)


class EngineDecisionArbiter:
    """Single source of truth for engine selection.

    Stateless domain service — no I/O, no persistence.
    All logic is deterministic based on inputs.
    """

    def arbitrate(
        self,
        flag_engine: str,
        profile_engine: str,
        fusion_engine: str,
        series_id: str,
        rollback_to_baseline: bool,
        series_overrides: Optional[dict[str, str]] = None,
    ) -> EngineDecision:
        """Determine final engine based on strict hierarchy.

        Args:
            flag_engine: Engine selected by FeatureFlags logic.
            profile_engine: Engine selected by series profile analysis.
            fusion_engine: Engine selected by WeightedFusion.
            series_id: Series identifier.
            rollback_to_baseline: ML_ROLLBACK_TO_BASELINE flag value.
            series_overrides: Dict of series_id → engine overrides.

        Returns:
            EngineDecision with chosen engine and authority trace.
        """
        overrides: List[str] = []
        series_overrides = series_overrides or {}

        # Rule 1: Panic button has absolute priority
        if rollback_to_baseline:
            return EngineDecision(
                chosen_engine="baseline_moving_average",
                authority="flags",
                reason="ML_ROLLBACK_TO_BASELINE panic button active",
                overrides=[],
            )

        # Rule 2: Series-specific override from flags
        if series_id in series_overrides:
            return EngineDecision(
                chosen_engine=series_overrides[series_id],
                authority="flags",
                reason=f"Series {series_id} has ML_ENGINE_SERIES_OVERRIDES entry",
                overrides=[],
            )

        # Rule 3: Flag vs Profile conflict
        if flag_engine != profile_engine:
            overrides.append(f"flag({flag_engine}) overridden by profile({profile_engine})")
            # Profile wins
            chosen = profile_engine
            authority = "profile"
            reason = f"Profile engine {profile_engine} overrides flag engine {flag_engine}"

            # Rule 4: Profile vs Fusion conflict
            if profile_engine != fusion_engine:
                overrides.append(f"fusion({fusion_engine}) overridden by profile({profile_engine})")

            return EngineDecision(
                chosen_engine=chosen,
                authority=authority,
                reason=reason,
                overrides=overrides,
            )

        # Rule 4: Profile vs Fusion conflict (when flag == profile)
        if profile_engine != fusion_engine:
            overrides.append(f"fusion({fusion_engine}) overridden by profile({profile_engine})")
            return EngineDecision(
                chosen_engine=profile_engine,
                authority="profile",
                reason=f"Profile engine {profile_engine} overrides fusion engine {fusion_engine}",
                overrides=overrides,
            )

        # Rule 5: Consensus — all layers agree
        return EngineDecision(
            chosen_engine=fusion_engine,
            authority="fusion",
            reason=f"All layers agree: {fusion_engine}",
            overrides=[],
        )
