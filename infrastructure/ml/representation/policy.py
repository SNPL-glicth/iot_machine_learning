"""Implementación agnóstica de la política de representación adaptativa.

Cumple el contrato BaseRepresentationPolicyPort del dominio.
Integra RegretRingBuffer, GenericShockSentinel, GenericRegimeShiftSentinel
y AgnosticPolicyStateMachine.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from iot_machine_learning.domain.entities.representation_evidence import (
    PolicyDecision,
    RepresentationLevel,
    SystemOperationalState,
)
from iot_machine_learning.domain.ports.representation_policy_port import (
    BaseRepresentationPolicyPort,
)
from .calibrators import EmpiricalDistributionProfile, NonParametricConformalCalibrator
from .ring_buffer import RegretRingBuffer
from .sentinels import GenericRegimeShiftSentinel, GenericShockSentinel
from .state_machine import AgnosticPolicyStateMachine, PolicyResolutionState


class AgnosticRepresentationPolicy(BaseRepresentationPolicyPort):
    """Enrutador de representación adaptativa para streams continuos."""

    def __init__(
        self,
        level_profile: EmpiricalDistributionProfile,
        shock_profile: EmpiricalDistributionProfile,
        block_size: int = 10,
        enable_hysteresis: bool = True,
        enable_backfill: bool = True,
        enable_safety: bool = True,
        cooldown_blocks: int = 3,
    ) -> None:
        self.block_size = block_size
        self.shock_sentinel = GenericShockSentinel(shock_profile)
        self.regime_sentinel = GenericRegimeShiftSentinel(level_profile)
        self.state_machine = AgnosticPolicyStateMachine(
            enable_hysteresis=enable_hysteresis,
            enable_backfill=enable_backfill,
            enable_representation_safety=enable_safety,
            cooldown_blocks=cooldown_blocks,
        )
        self.ring_buffer = RegretRingBuffer(capacity=block_size * 3)

        self._current_block: list[float] = []
        self._block_idx: int = 0
        self._prev_val: float = level_profile.median
        self._last_decision: PolicyDecision | None = None
        self._active_slice: tuple[RepresentationLevel, list[float]] = (
            RepresentationLevel.TEN_X,
            [],
        )

    def step(self, point: float, index: int) -> PolicyDecision:
        """Evalúa un nuevo punto del stream y decide la representación adecuada."""
        self._current_block.append(point)
        self.ring_buffer.append(index, point, float(index), str(index))

        # Si aún no se completa el bloque, mantenemos la decisión activa
        if len(self._current_block) < self.block_size:
            if self._last_decision is None:
                level_map = {
                    PolicyResolutionState.COMPRESSED: RepresentationLevel.TEN_X,
                    PolicyResolutionState.BALANCED: RepresentationLevel.TWO_X,
                    PolicyResolutionState.HIGH_RESOLUTION: RepresentationLevel.RAW,
                }
                self._last_decision = PolicyDecision(
                    level=level_map[self.state_machine.current_state],
                    operational_state=SystemOperationalState.RESTING,
                    reason="block_accumulating",
                )
            return self._last_decision

        # Bloque completo: evaluar sentinelas y máquina de estados
        block_vals = np.array(self._current_block, dtype=np.float64)
        is_shock, s_shock = self.shock_sentinel.inspect(block_vals, self._prev_val)
        is_regime, s_regime = self.regime_sentinel.inspect(block_vals)

        report = self.state_machine.evaluate_block(
            block_idx=self._block_idx,
            shock_triggered=is_shock,
            regime_triggered=is_regime,
            shock_surprise=s_shock,
            regime_surprise=s_regime,
        )

        decision = report.to_domain_decision()
        self._last_decision = decision

        # Preparar slice adaptativo
        step = report.subsample_step
        selected_points = [self._current_block[i] for i in range(0, self.block_size, step)]
        self._active_slice = (decision.level, selected_points)

        self._prev_val = float(block_vals[-1])
        self._current_block.clear()
        self._block_idx += 1

        return decision

    def get_effective_stream_slice(self) -> tuple[RepresentationLevel, Sequence[float]]:
        """Retorna el slice temporal adaptado a la representación activa."""
        return self._active_slice
