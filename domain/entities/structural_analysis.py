"""Re-export facade — backward compatibility.

Canonical location: ``domain.entities.series.structural_analysis``
"""

from .series.structural_analysis import (
    StructuralAnalysis,
    RegimeType,
    _classify_regime,
)

__all__ = ["StructuralAnalysis", "RegimeType", "_classify_regime"]
