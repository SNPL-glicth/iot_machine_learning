"""Sanitize phase — input validation, clamping and ramp detection.

IMP-1 entry points.
"""

from .bounds_provider import (
    BoundsProvider,
    LocalWindowBoundsProvider,
    SeriesValuesBoundsProvider,
)
from .cusum import detect_ramp
from .phase import SanitizeConfig, SanitizePhase

__all__ = [
    "BoundsProvider",
    "LocalWindowBoundsProvider",
    "SanitizeConfig",
    "SanitizePhase",
    "SeriesValuesBoundsProvider",
    "detect_ramp",
]
