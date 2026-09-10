"""TrajectoryTracker: runtime monitoring of actual movements against predicted trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .trajectory import Trajectory
from .movement import Movement


@dataclass
class DeviationStatus:
    """Result of evaluating one actual movement against the planned step."""
    is_valid: bool
    step_index: int
    reason: str = ""


class TrajectoryTracker:
    """
    Monitors real incoming movements against the predicted active trajectory.
    
    The tracker advances only on valid steps. If a movement deviates beyond
    configured tolerances, it signals invalidation so the engine can react.
    """

    def __init__(
        self, 
        max_direction_dev_deg: float = 45.0, 
        max_velocity_rel_err: float = 0.5
    ):
        self.max_dir_dev_cos = float(math.cos(math.radians(max_direction_dev_deg)))
        self.max_vel_rel_err = max_velocity_rel_err
        self._active_trajectory: Optional[Trajectory] = None
        self._current_step: int = 0

    @property
    def has_active_trajectory(self) -> bool:
        return self._active_trajectory is not None

    @property
    def active_trajectory(self) -> Optional[Trajectory]:
        return self._active_trajectory

    def set_active_trajectory(
        self, 
        trajectory: Optional[Trajectory], 
        start_step: int = 0
    ) -> None:
        """Set a new active trajectory, starting evaluation at `start_step`."""
        self._active_trajectory = trajectory
        self._current_step = start_step if trajectory is not None else 0

    def evaluate_step(self, actual_movement: Movement) -> DeviationStatus:
        """
        Evaluates whether the actual movement matches the planned trajectory step.
        
        Returns DeviationStatus with is_valid=False if the active trajectory
        has ended, or if directional/velocity deviation exceeds thresholds.
        """
        if not self._active_trajectory or self._current_step >= len(self._active_trajectory.movements):
            return DeviationStatus(
                is_valid=False, 
                step_index=self._current_step, 
                reason="No_Active_Trajectory"
            )
        
        planned_m = self._active_trajectory.movements[self._current_step]
        
        # 1. Directional alignment (cosine similarity)
        dir_sim = float(np.dot(actual_movement.direction, planned_m.direction))
        if dir_sim < self.max_dir_dev_cos:
            return DeviationStatus(
                is_valid=False, 
                step_index=self._current_step, 
                reason=f"Direction_Deviation_Exceeded (sim={dir_sim:.2f} < {self.max_dir_dev_cos:.2f})"
            )
        
        # 2. Velocity relative error
        v_planned = planned_m.velocity
        if v_planned > 1e-6:
            v_err = abs(actual_movement.velocity - v_planned) / v_planned
            if v_err > self.max_vel_rel_err:
                return DeviationStatus(
                    is_valid=False, 
                    step_index=self._current_step, 
                    reason=f"Velocity_Error_Exceeded (err={v_err:.2f} > {self.max_vel_rel_err})"
                )
        
        self._current_step += 1
        return DeviationStatus(is_valid=True, step_index=self._current_step - 1)

    def reset(self) -> None:
        self._active_trajectory = None
        self._current_step = 0