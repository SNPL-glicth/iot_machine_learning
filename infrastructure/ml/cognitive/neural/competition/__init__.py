"""Competition mechanism for neural vs universal engine selection.

Components:
    - NeuralArbiter: Decides winner between neural and universal engines
    - ConfidenceComparator: Compares confidence scores
    - OutcomeTracker: Tracks per-domain win history
"""

from .arbiter import NeuralArbiter
from .confidence_comparator import ConfidenceComparator
from .outcome_tracker import OutcomeTracker

__all__ = [
    "NeuralArbiter",
    "ConfidenceComparator",
    "OutcomeTracker",
]
