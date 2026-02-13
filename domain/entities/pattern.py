"""Re-exports para backward compatibility.

Las entidades individuales ahora viven en ``patterns/``:
- ``patterns/pattern_result.py`` → PatternType, PatternResult
- ``patterns/change_point.py`` → ChangePointType, ChangePoint
- ``patterns/delta_spike.py`` → SpikeClassification, DeltaSpikeResult
- ``patterns/operational_regime.py`` → OperationalRegime
"""

from .patterns.pattern_result import PatternType, PatternResult
from .patterns.change_point import ChangePointType, ChangePoint
from .patterns.delta_spike import SpikeClassification, DeltaSpikeResult
from .patterns.operational_regime import OperationalRegime

__all__ = [
    "PatternType",
    "PatternResult",
    "ChangePointType",
    "ChangePoint",
    "SpikeClassification",
    "DeltaSpikeResult",
    "OperationalRegime",
]
