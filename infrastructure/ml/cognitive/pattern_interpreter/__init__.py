"""Pattern Interpreter - Human-readable interpretation of detected patterns.

Transforms technical pattern detection results into human-understandable
interpretations with domain context and severity classification.
"""

from .interpreter import PatternInterpreter
from .types import InterpretedPattern

__all__ = [
    "PatternInterpreter",
    "InterpretedPattern",
]
