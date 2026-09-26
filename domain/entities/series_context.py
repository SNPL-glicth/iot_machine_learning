"""Re-export facade — backward compatibility.

Canonical location: ``domain.entities.series.series_context``
"""

from .series.series_context import SeriesContext, Threshold

__all__ = ["SeriesContext", "Threshold"]
