"""Tool guard and safety levels.

Defines safety levels and guard results for tool execution control.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class SafetyLevel(Enum):
    """Safety levels for tool execution control."""
    AUTO = auto()    # Execute automatically without human intervention
    ASK = auto()     # Request human approval before execution
    DENY = auto()    # Block execution entirely


@dataclass(frozen=True)
class GuardResult:
    """Result of safety guard evaluation."""
    allowed: bool
    level: SafetyLevel
    reason: str
    
    @classmethod
    def allow(cls, reason: str = ""):
        return cls(True, SafetyLevel.AUTO, reason)
    
    @classmethod
    def ask(cls, reason: str):
        return cls(True, SafetyLevel.ASK, reason)
    
    @classmethod
    def deny(cls, reason: str):
        return cls(False, SafetyLevel.DENY, reason)
