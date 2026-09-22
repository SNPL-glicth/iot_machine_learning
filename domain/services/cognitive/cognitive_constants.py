"""Cognitive constants shared across domain services.

DRY: Single source of truth for magic numbers used in multiple services.
ISO 25010: All parameters injectable for testability.

NOTE: These are FALLBACK defaults. Services should read from FeatureFlags
at runtime for hot-reload capability (ISO 27001 A.12.1.2).
"""

from __future__ import annotations

# Confidence reduction factor when evidence is sparse (< 2 pieces)
# Used by: conservative/strategy.py
# Rationale: Lower confidence when we have insufficient data
CONFIDENCE_REDUCTION_SPARSE_EVIDENCE: float = 0.9

# Confidence floor for conservative decisions
# Conservative strategy guarantees minimum confidence even with poor data
CONFIDENCE_FLOOR_CONSERVATIVE: float = 0.6

# Confidence ceiling to avoid overconfidence
CONFIDENCE_CEILING_CONSERVATIVE: float = 0.95


def get_confidence_reduction_sparse() -> float:
    """Get confidence reduction factor for sparse evidence."""
    return CONFIDENCE_REDUCTION_SPARSE_EVIDENCE


def get_confidence_floor() -> float:
    """Get confidence floor for conservative strategy."""
    return CONFIDENCE_FLOOR_CONSERVATIVE


def get_confidence_ceiling() -> float:
    """Get confidence ceiling for conservative strategy."""
    return CONFIDENCE_CEILING_CONSERVATIVE

