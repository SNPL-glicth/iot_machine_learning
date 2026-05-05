"""Domain policies — pure business rules and classification policies.

Policies encapsulate business decisions that are too complex for simple
value objects but do NOT contain I/O or orchestration logic.
"""

from __future__ import annotations

from .policy_result import SeverityPolicyResult
from .threshold_policy import ThresholdPolicy

__all__ = [
    "ThresholdPolicy",
    "SeverityPolicyResult",
]
