"""Tests for EngineDecisionArbiter domain service.

Covers the strict hierarchy of engine selection:
1. ML_ROLLBACK_TO_BASELINE → baseline, authority=flags
2. series_id in ML_ENGINE_SERIES_OVERRIDES → that engine, authority=flags
3. flag != profile → profile wins, override logged
4. profile != fusion → profile wins, override logged
5. Consensus → fusion_engine, authority=fusion
"""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.services.engine_decision_arbiter import (
    EngineDecision,
    EngineDecisionArbiter,
)


class TestArbiterBasics:
    """Basic construction and sanity checks."""

    def test_arbiter_construction(self):
        """EngineDecisionArbiter can be instantiated."""
        arbiter = EngineDecisionArbiter()
        assert arbiter is not None

    def test_engine_decision_dataclass(self):
        """EngineDecision is a proper frozen dataclass."""
        decision = EngineDecision(
            chosen_engine="taylor",
            authority="profile",
            reason="Test reason",
            overrides=["flag(baseline) overridden by profile(taylor)"],
        )
        assert decision.chosen_engine == "taylor"
        assert decision.authority == "profile"
        assert len(decision.overrides) == 1


class TestRule1RollbackToBaseline:
    """Rule 1: ML_ROLLBACK_TO_BASELINE has absolute priority."""

    def test_rollback_to_baseline_true(self):
        """rollback_to_baseline=True → always baseline, authority=flags."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="taylor",
            profile_engine="kalman",
            fusion_engine="ensemble",
            series_id="test_series",
            rollback_to_baseline=True,
            series_overrides={},
        )
        assert decision.chosen_engine == "baseline_moving_average"
        assert decision.authority == "flags"
        assert "panic button" in decision.reason.lower()
        assert decision.overrides == []

    def test_rollback_ignores_all_other_inputs(self):
        """Rollback ignores series_overrides and all engine inputs."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="anything",
            profile_engine="anything_else",
            fusion_engine="third_thing",
            series_id="test_series",
            rollback_to_baseline=True,
            series_overrides={"test_series": "custom_engine"},
        )
        assert decision.chosen_engine == "baseline_moving_average"


class TestRule2SeriesOverride:
    """Rule 2: series_id in ML_ENGINE_SERIES_OVERRIDES wins."""

    def test_series_override_applies(self):
        """series_id in overrides → that engine, authority=flags."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="baseline",
            profile_engine="taylor",
            fusion_engine="ensemble",
            series_id="series_123",
            rollback_to_baseline=False,
            series_overrides={"series_123": "custom_engine"},
        )
        assert decision.chosen_engine == "custom_engine"
        assert decision.authority == "flags"
        assert "series_123" in decision.reason
        assert decision.overrides == []

    def test_series_override_not_in_dict_continues(self):
        """If series_id not in overrides, continue to next rule."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="baseline",
            profile_engine="taylor",
            fusion_engine="ensemble",
            series_id="series_999",
            rollback_to_baseline=False,
            series_overrides={"series_123": "custom_engine"},
        )
        # Should continue to flag vs profile check
        assert decision.chosen_engine == "taylor"  # profile wins

    def test_empty_overrides_dict_continues(self):
        """Empty overrides dict continues to next rule."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="baseline",
            profile_engine="taylor",
            fusion_engine="ensemble",
            series_id="any_series",
            rollback_to_baseline=False,
            series_overrides={},
        )
        assert decision.chosen_engine == "taylor"

    def test_none_overrides_defaults_to_empty(self):
        """None overrides defaults to empty dict."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="baseline",
            profile_engine="taylor",
            fusion_engine="ensemble",
            series_id="any_series",
            rollback_to_baseline=False,
            series_overrides=None,
        )
        assert decision.chosen_engine == "taylor"


class TestRule3FlagVsProfileConflict:
    """Rule 3: flag != profile → profile wins."""

    def test_flag_differs_from_profile_profile_wins(self):
        """When flag != profile, profile wins with override logged."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="baseline",
            profile_engine="taylor",
            fusion_engine="taylor",
            series_id="series_1",
            rollback_to_baseline=False,
            series_overrides={},
        )
        assert decision.chosen_engine == "taylor"
        assert decision.authority == "profile"
        assert "profile engine" in decision.reason.lower()
        assert len(decision.overrides) == 1
        assert "flag(baseline)" in decision.overrides[0]
        assert "profile(taylor)" in decision.overrides[0]

    def test_flag_equals_profile_no_override(self):
        """When flag == profile, no override needed."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="taylor",
            profile_engine="taylor",
            fusion_engine="ensemble",
            series_id="series_1",
            rollback_to_baseline=False,
            series_overrides={},
        )
        # This continues to rule 4 (profile vs fusion)
        assert decision.chosen_engine == "taylor"
        assert len(decision.overrides) == 1  # fusion overridden


class TestRule4ProfileVsFusionConflict:
    """Rule 4: profile != fusion → profile wins."""

    def test_profile_differs_from_fusion_profile_wins(self):
        """When profile != fusion, profile wins with override logged."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="taylor",
            profile_engine="taylor",
            fusion_engine="ensemble",
            series_id="series_1",
            rollback_to_baseline=False,
            series_overrides={},
        )
        assert decision.chosen_engine == "taylor"
        assert decision.authority == "profile"
        assert len(decision.overrides) == 1
        assert "fusion(ensemble)" in decision.overrides[0]
        assert "profile(taylor)" in decision.overrides[0]

    def test_profile_equals_fusion_consensus(self):
        """When all equal, authority=fusion (rule 5)."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="taylor",
            profile_engine="taylor",
            fusion_engine="taylor",
            series_id="series_1",
            rollback_to_baseline=False,
            series_overrides={},
        )
        assert decision.chosen_engine == "taylor"
        assert decision.authority == "fusion"
        assert decision.overrides == []


class TestRule5Consensus:
    """Rule 5: All layers agree → fusion authority."""

    def test_full_consensus(self):
        """All three engines agree → fusion authority, no overrides."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="ensemble",
            profile_engine="ensemble",
            fusion_engine="ensemble",
            series_id="series_1",
            rollback_to_baseline=False,
            series_overrides={},
        )
        assert decision.chosen_engine == "ensemble"
        assert decision.authority == "fusion"
        assert "all layers agree" in decision.reason.lower()
        assert decision.overrides == []

    def test_consensus_baseline(self):
        """Consensus on baseline is valid."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="baseline",
            profile_engine="baseline",
            fusion_engine="baseline",
            series_id="series_1",
            rollback_to_baseline=False,
            series_overrides={},
        )
        assert decision.chosen_engine == "baseline"
        assert decision.authority == "fusion"


class TestDoubleConflict:
    """Both flag != profile AND profile != fusion."""

    def test_all_three_different(self):
        """All three engines different → profile wins with two overrides."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="baseline",
            profile_engine="taylor",
            fusion_engine="ensemble",
            series_id="series_1",
            rollback_to_baseline=False,
            series_overrides={},
        )
        assert decision.chosen_engine == "taylor"
        assert decision.authority == "profile"
        assert len(decision.overrides) == 2
        assert "flag(baseline)" in decision.overrides[0]
        assert "fusion(ensemble)" in decision.overrides[1]


class TestPriorityOrdering:
    """Verify strict priority order of rules."""

    def test_rollback_beats_series_override(self):
        """Rule 1 beats Rule 2."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="taylor",
            profile_engine="taylor",
            fusion_engine="taylor",
            series_id="series_123",
            rollback_to_baseline=True,  # Rule 1
            series_overrides={"series_123": "custom"},  # Rule 2
        )
        assert decision.chosen_engine == "baseline_moving_average"
        assert decision.authority == "flags"

    def test_series_override_beats_flag_profile_conflict(self):
        """Rule 2 beats Rule 3."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="baseline",
            profile_engine="taylor",
            fusion_engine="ensemble",
            series_id="series_123",
            rollback_to_baseline=False,
            series_overrides={"series_123": "custom_override"},  # Rule 2
        )
        assert decision.chosen_engine == "custom_override"
        assert decision.authority == "flags"
        assert decision.overrides == []  # No conflict, override applied directly


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_series_id(self):
        """Empty series_id should still work."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="baseline",
            profile_engine="baseline",
            fusion_engine="baseline",
            series_id="",
            rollback_to_baseline=False,
            series_overrides={},
        )
        assert decision.chosen_engine == "baseline"
        assert decision.authority == "fusion"

    def test_unknown_engines(self):
        """Unusual engine names work fine."""
        arbiter = EngineDecisionArbiter()
        decision = arbiter.arbitrate(
            flag_engine="experimental_v2",
            profile_engine="experimental_v2",
            fusion_engine="experimental_v2",
            series_id="test",
            rollback_to_baseline=False,
            series_overrides={},
        )
        assert decision.chosen_engine == "experimental_v2"
        assert decision.authority == "fusion"
