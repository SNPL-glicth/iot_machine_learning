"""Shared, lock-protected per-call failure collector."""

from __future__ import annotations

from threading import Lock
from typing import Dict, List

# Using a module-level list + Lock instead of threading.local so failures
# recorded inside ThreadPoolExecutor workers propagate back to the calling
# thread. Cleared at the start of every ``collect_perceptions()`` call.
# Assumes serial orchestrator use per process (current design).

_failures: List[Dict[str, str]] = []
_failures_lock = Lock()


def consume_engine_failures() -> List[Dict[str, str]]:
    """Return and clear the most recent batch of engine failures.

    Called by :class:`PredictPhase` immediately after
    :func:`collect_perceptions` so the list is surfaced in the
    ``metadata["engine_failures"]`` field of the final PredictionResult.
    """
    with _failures_lock:
        out = list(_failures)
        _failures.clear()
    return out


def _record_failure(engine_name: str, reason: str, error: str = "") -> None:
    entry = {"engine": engine_name, "reason": reason}
    if error:
        entry["error"] = error
    with _failures_lock:
        _failures.append(entry)


def _clear_failures() -> None:
    with _failures_lock:
        _failures.clear()
