"""Phase setter functions for ExplanationBuilder — backward-compatible facade.

Implementation split into two focused modules:
- perception_phases.py — PERCEIVE, FILTER, PREDICT, ADAPT
- fusion_phases.py     — INHIBIT, FUSE, fallback, audit_trace_id

This module re-exports all functions so existing callers are unaffected.
"""

from .perception_phases import (
    set_adaptation,
    set_filter,
    set_perceptions,
    set_signal,
)
from .fusion_phases import (
    set_audit_trace_id,
    set_fallback,
    set_fusion,
    set_inhibition,
)

__all__ = [
    "set_signal",
    "set_filter",
    "set_perceptions",
    "set_adaptation",
    "set_inhibition",
    "set_fusion",
    "set_fallback",
    "set_audit_trace_id",
]
