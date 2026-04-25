"""ML analyzers for numeric and tabular data.

This package contains pure analysis functions for:
- Numeric column analysis (structural, anomaly detection)
- Tabular document orchestration
"""

from .numeric_analyzer import analyze_numeric_column
from .tabular_analyzer import analyze_tabular_document

# window_analyzer depends on infrastructure.ml.models which is not present
# in leaner deployments. Load it lazily so importers that don't need
# WindowAnalyzer (e.g. the query route) are not blocked by the missing dep.
try:
    from .window_analyzer import WindowAnalyzer  # noqa: F401
except ImportError:  # pragma: no cover — optional dep
    WindowAnalyzer = None  # type: ignore[assignment]

__all__ = [
    "analyze_numeric_column",
    "analyze_tabular_document",
    "WindowAnalyzer",
]
