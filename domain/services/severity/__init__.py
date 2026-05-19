"""Severity classification domain services."""
try:
    from .severity_rules import classify_severity_agnostic, SeverityResult
except ImportError:
    classify_severity_agnostic = None  # type: ignore[assignment,misc]
    SeverityResult = None  # type: ignore[assignment,misc]

try:
    from .severity_helpers import *  # noqa: F401,F403
except ImportError:
    pass

try:
    from .severity_legacy import *  # noqa: F401,F403
except ImportError:
    pass

__all__ = ["classify_severity_agnostic", "SeverityResult"]
