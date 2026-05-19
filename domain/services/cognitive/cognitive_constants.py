"""Cognitive constants shared across domain services.

DRY: Single source of truth for magic numbers used in multiple services.
ISO 25010: All parameters injectable for testability.

NOTE: These are FALLBACK defaults. Services should read from FeatureFlags
at runtime for hot-reload capability (ISO 27001 A.12.1.2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iot_machine_learning.ml_service.config.flags import FeatureFlags


def _get_flags() -> "FeatureFlags":
    """Lazy import to avoid circular dependencies."""
    try:
        from iot_machine_learning.ml_service.config.feature_flags import get_feature_flags
        return get_feature_flags()
    except Exception:
        # Fallback to defaults if flags not available
        from iot_machine_learning.ml_service.config.flags import FeatureFlags
        return FeatureFlags()


# Confidence reduction factor when evidence is sparse (< 2 pieces)
# Used by: conservative/strategy.py
# Rationale: Lower confidence when we have insufficient data
# FALLBACK: 0.9; override via ML_DECISION_CONFIDENCE_REDUCTION_SPARSE
CONFIDENCE_REDUCTION_SPARSE_EVIDENCE: float = 0.9

# Confidence floor for conservative decisions
# Conservative strategy guarantees minimum confidence even with poor data
# FALLBACK: 0.6; override via ML_DECISION_CONFIDENCE_FLOOR
CONFIDENCE_FLOOR_CONSERVATIVE: float = 0.6

# Confidence ceiling to avoid overconfidence
# FALLBACK: 0.95; override via ML_DECISION_CONFIDENCE_CEILING
CONFIDENCE_CEILING_CONSERVATIVE: float = 0.95


def get_confidence_reduction_sparse() -> float:
    """Get confidence reduction factor from flags (hot-reload)."""
    return _get_flags().ML_DECISION_CONFIDENCE_REDUCTION_SPARSE


def get_confidence_floor() -> float:
    """Get confidence floor from flags (hot-reload)."""
    return _get_flags().ML_DECISION_CONFIDENCE_FLOOR


def get_confidence_ceiling() -> float:
    """Get confidence ceiling from flags (hot-reload)."""
    return _get_flags().ML_DECISION_CONFIDENCE_CEILING
