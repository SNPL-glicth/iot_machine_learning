"""ExplanationBuilder — backward-compatible facade.

The implementation has been split into:
- builder.py       — ExplanationBuilder class (constructor + build())
- phase_setters.py — standalone setter functions (one per reasoning phase)

This module re-exports ExplanationBuilder so all existing imports continue
to work without modification.
"""

from .builder import ExplanationBuilder

__all__ = ["ExplanationBuilder"]
