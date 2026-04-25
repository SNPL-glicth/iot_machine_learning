"""SeriesValuesStore — rolling buffer of raw sensor values per series.

Redis-only, inert fallback. Used by :class:`SanitizePhase` (IMP-1) to
compute 6σ clamping bounds with cross-invocation memory.
"""

from .series_values_store import SeriesValuesStore

__all__ = ["SeriesValuesStore"]
