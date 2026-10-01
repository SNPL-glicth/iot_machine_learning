"""Capa de representación adaptativa y sentinelas empíricos.

Exporta:
- RegretRingBuffer
- NonParametricConformalCalibrator, EmpiricalDistributionProfile
- GenericShockSentinel, GenericRegimeShiftSentinel
- AgnosticPolicyStateMachine, PolicyResolutionState, PolicyOperationalMode, PolicyDecisionReport
- AgnosticRepresentationPolicy
"""

from __future__ import annotations

from .calibrators import EmpiricalDistributionProfile, NonParametricConformalCalibrator
from .policy import AgnosticRepresentationPolicy
from .ring_buffer import RegretRingBuffer
from .sentinels import GenericRegimeShiftSentinel, GenericShockSentinel
from .state_machine import (
    AgnosticPolicyStateMachine,
    PolicyDecisionReport,
    PolicyOperationalMode,
    PolicyResolutionState,
)

__all__ = [
    "RegretRingBuffer",
    "EmpiricalDistributionProfile",
    "NonParametricConformalCalibrator",
    "GenericShockSentinel",
    "GenericRegimeShiftSentinel",
    "PolicyResolutionState",
    "PolicyOperationalMode",
    "PolicyDecisionReport",
    "AgnosticPolicyStateMachine",
    "AgnosticRepresentationPolicy",
]
