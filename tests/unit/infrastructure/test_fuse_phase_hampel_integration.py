"""FusePhase ↔ Hampel integration tests (IMP-2 §C)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
    EnginePerception,
    InhibitionState,
)
from iot_machine_learning.infrastructure.ml.cognitive.fusion import WeightedFusion
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import (
    PipelineContext,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import (
    FusePhase,
)


def _perc(name: str, value: float, conf: float = 0.8) -> EnginePerception:
    return EnginePerception(
        engine_name=name, predicted_value=value, confidence=conf, trend="stable"
    )


def _state(name: str, weight: float = 1.0) -> InhibitionState:
    return InhibitionState(
        engine_name=name, base_weight=weight, inhibited_weight=weight
    )


def _profile_mock(std: float = 1.0):
    prof = MagicMock()
    prof.std = std
    return prof


def _build_ctx(
    perceptions: List[EnginePerception],
    states: List[InhibitionState],
) -> PipelineContext:
    orch = MagicMock()
    orch._fusion = WeightedFusion()
    orch._correlation_port = None
    orch._storage = None
    return PipelineContext(
        orchestrator=orch,
        values=[1.0, 2.0, 3.0],
        timestamps=None,
        series_id="sid",
        flags=None,
        timer=None,
        perceptions=perceptions,
        inhibition_states=states,
        profile=_profile_mock(),
        neighbors=None,
        neighbor_values=None,
        neighbor_trends=None,
        explanation=None,
    )


# =========================================================================


class TestFusePhaseHampelIntegration:
    def test_clean_perceptions_no_rejection_flag(self) -> None:
        perceptions = [
            _perc("a", 10.0),
            _perc("b", 10.5),
            _perc("c", 9.5),
        ]
        states = [_state("a"), _state("b"), _state("c")]
        ctx = _build_ctx(perceptions, states)
        
        result = FusePhase().execute(ctx)
        
        assert result.fusion_flags == []
        # Fused value should be ~10
        assert 9.0 < result.fused_value < 11.0
        assert result.fusion_method == "weighted_average"

    def test_outlier_rejected_before_fusion(self) -> None:
        # Outlier at 1000; fused value must NOT be dragged toward it.
        perceptions = [
            _perc("a", 10.0),
            _perc("b", 11.0),
            _perc("c", 1000.0),
        ]
        states = [_state("a"), _state("b"), _state("c")]
        ctx = _build_ctx(perceptions, states)
        
        result = FusePhase().execute(ctx)
        
        # Flag present
        assert "hampel_rejected:1" in result.fusion_flags
        # Fused value stays near 10-11, NOT pulled to ~337 (raw average).
        assert result.fused_value < 20.0
        # Diagnostic populated.
        assert result.hampel_diagnostic is not None
        rejected = result.hampel_diagnostic["rejected"]
        assert len(rejected) == 1
        assert rejected[0]["engine"] == "c"
        assert rejected[0]["predicted_value"] == 1000.0

    def test_selected_engine_not_outlier(self) -> None:
        """Outlier should not be picked as selected_engine."""
        perceptions = [
            _perc("a", 10.0),
            _perc("b", 11.0),
            _perc("outlier", 1000.0),
        ]
        states = [_state("a", 0.3), _state("b", 0.3), _state("outlier", 0.4)]
        ctx = _build_ctx(perceptions, states)
        
        result = FusePhase().execute(ctx)
        
        assert result.selected_engine != "outlier"
        assert result.selected_engine in {"a", "b"}

    def test_metadata_carries_hampel_diagnostic(self) -> None:
        """AssemblyPhase should include result.hampel_diagnostic in metadata."""
        perceptions = [
            _perc("a", 10.0),
            _perc("b", 11.0),
            _perc("c", 500.0),
        ]
        states = [_state("a"), _state("b"), _state("c")]
        ctx = _build_ctx(perceptions, states)
        result_ctx = FusePhase().execute(ctx)
        
        assert result_ctx.hampel_diagnostic is not None
        assert "median" in result_ctx.hampel_diagnostic
        assert "mad" in result_ctx.hampel_diagnostic
        assert "rejected" in result_ctx.hampel_diagnostic

    def test_kill_switch_disables_hampel(self) -> None:
        """ML_HAMPEL_ENABLED=0 → outlier survives into fusion."""
        perceptions = [
            _perc("a", 10.0),
            _perc("b", 11.0),
            _perc("outlier", 1000.0),
        ]
        states = [_state("a"), _state("b"), _state("outlier")]
        ctx = _build_ctx(perceptions, states)
        
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import (
            fuse_phase as fp_module,
        )
        with patch.object(fp_module, "ML_HAMPEL_ENABLED", False):
            result = FusePhase().execute(ctx)
        
        assert result.fusion_flags == []
        assert result.hampel_diagnostic == {}  # skipped entirely
        # Fused value dragged by outlier (raw weighted average of 10/11/1000 ≈ 340)
        assert result.fused_value > 100.0

    def test_below_min_perceptions_no_filter(self) -> None:
        """With only 2 perceptions, Hampel is a no-op."""
        perceptions = [_perc("a", 10.0), _perc("b", 1000.0)]
        states = [_state("a"), _state("b")]
        ctx = _build_ctx(perceptions, states)
        
        result = FusePhase().execute(ctx)
        
        assert result.fusion_flags == []
        # Both perceptions used.
        assert result.fusion_method == "weighted_average"
