"""Perception subsystem — engine perception collection and orchestration.

Components:
    - PerceptionPhases: Phase orchestration
    - PhaseSetters: Signal/filter setters
    - Helpers: Collection helpers
    - RecordActualHandler: Actual value recording
"""

from .helpers import collect_perceptions
from .record_actual_handler import RecordActualHandler

__all__ = [
    "collect_perceptions",
    "RecordActualHandler",
]
