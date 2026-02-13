"""Re-export facade — backward compatibility.

Canonical location: ``domain.entities.series.time_series``
"""

from .series.time_series import TimeSeries, TimePoint

__all__ = ["TimeSeries", "TimePoint"]
