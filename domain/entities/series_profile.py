"""Re-export facade — backward compatibility.

Canonical location: ``domain.entities.series.series_profile``
"""

from .series.series_profile import (
    SeriesProfile,
    VolatilityLevel,
    StationarityHint,
    compute_profile,
)

__all__ = [
    "SeriesProfile",
    "VolatilityLevel",
    "StationarityHint",
    "compute_profile",
]
