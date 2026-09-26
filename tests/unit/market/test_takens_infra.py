"""Unit tests for Phase 3 Takens Ring Buffer, Engine, and MoE Adapter."""

from __future__ import annotations

import numpy as np
import pytest

from domain.entities.rosa_roja.movement import Movement
from domain.entities.rosa_roja.trajectory import TerminalState, Trajectory
from domain.entities.takens import TakensParameters
from infrastructure.ml.adapters.takens_adapter import TakensExpertAdapter
from infrastructure.ml.engines.ramanujan_takens import RamanujanTakensEngine, TakensRingBuffer
from infrastructure.ml.engines.rosa_roja.algorithms.modules.module3_moe_gating import (
    MultiplicativeMoEGating,
)


def _build_dummy_trajectory(deltas: list[float]) -> Trajectory:
    movements: list[Movement] = []
    prev: Movement | None = None
    for i, d in enumerate(deltas):
        m = Movement.from_raw(
            delta_state=np.array([d, 0.0, 0.0]),
            delta_time=0.1,
            timestamp=float(i),
            mahalanobis_dist=1.0,
            prev_movement=prev,
        )
        movements.append(m)
        prev = m

    return Trajectory(
        movements=tuple(movements),
        coherence_score=0.85,
        invalidation_step=None,
        terminal_state=TerminalState(
            state_vector=np.array([deltas[-1], 0.0, 0.0]),
            step_index=len(deltas) - 1,
            confidence=0.85,
        ),
    )


def test_circular_buffer_wrap_around_and_branchless_modulo():
    buf = TakensRingBuffer(capacity=8)
    assert buf.capacity == 8
    assert buf.size == 0

    # 1. Push 5 elements
    for i in range(5):
        buf.push(float(i))
    assert buf.size == 5
    assert np.allclose(buf.get_flat_history(), [0.0, 1.0, 2.0, 3.0, 4.0])

    # 2. Push 5 more elements to trigger wrap-around: total 10 pushed, keeps last 8: [2..9]
    for i in range(5, 10):
        buf.push(float(i))
    assert buf.size == 8
    assert buf.is_full is True
    assert np.allclose(buf.get_flat_history(), [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    # 3. Partial history extraction
    assert np.allclose(buf.get_flat_history(4), [6.0, 7.0, 8.0, 9.0])


def test_ramanujan_takens_engine_prediction():
    engine = RamanujanTakensEngine(params=TakensParameters(m=3, tau_strides=(1, 2)))
    assert engine.name == "ramanujan_takens"

    # Feed steady orbit
    series = [10.0 + 0.1 * np.sin(i) for i in range(30)]
    res = engine.predict(series)

    assert 0.0 <= res.confidence <= 1.0
    assert "topological_audit" in res.metadata
    assert "d_effective" in res.metadata
    assert engine.latest_audit is not None


def test_takens_expert_adapter_in_moe_gating_and_veto():
    params = TakensParameters(
        m=3,
        tau_strides=(1, 2),
        tau_fnn=0.40,
        max_geodesic_distance=3.0,
    )
    engine = RamanujanTakensEngine(params=params, window_size=20)
    # Warm up engine history around level 5.0
    for _ in range(30):
        engine.push_observation(5.0)

    adapter = TakensExpertAdapter(
        engine=engine,
        is_critical=True,
        threshold=0.50,
        weight=1.5,
    )
    assert adapter.name == "ramanujan_takens"
    assert adapter.is_critical is True

    gating = MultiplicativeMoEGating()

    # Case 1: Nominal trajectory in proximity to 5.0 -> passes without veto
    traj_nominal = _build_dummy_trajectory([5.05] * 12)
    val_nominal = gating.evaluate_and_veto(
        trajectories=[traj_nominal],
        jury=[adapter],
        lambda_t=0.5,
        phi_ritmo=0.8,
    )
    assert val_nominal.veto_triggered is False
    assert val_nominal.chosen_trajectory is not None
    assert val_nominal.global_confidence > 0.0

    # Case 2: Out-of-manifold trajectory (deltas = 50.0) -> triggers critical veto
    traj_erratic = _build_dummy_trajectory([50.0] * 12)
    val_veto = gating.evaluate_and_veto(
        trajectories=[traj_erratic],
        jury=[adapter],
        lambda_t=0.5,
        phi_ritmo=0.8,
    )
    assert val_veto.veto_triggered is True
    assert val_veto.chosen_trajectory is None
    assert val_veto.veto_details is not None
    assert val_veto.veto_details.expert_name == "ramanujan_takens"
