"""Deadline-based timeout guard for single-threaded engine execution.

Uses :class:`threading.Timer` — no new dependencies.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class EngineTimeoutError(RuntimeError):
    """Raised when an engine exceeds its wall-clock deadline."""

    def __init__(self, engine_name: str, timeout_ms: float) -> None:
        self.engine_name = engine_name
        self.timeout_ms = timeout_ms
        super().__init__(
            f"Engine '{engine_name}' exceeded deadline of {timeout_ms} ms"
        )


def run_with_deadline(
    fn: Callable[[], T],
    timeout_ms: float,
    engine_name: str,
) -> T:
    """Execute ``fn()`` with a wall-clock deadline.

    Args:
        fn: Zero-argument callable to execute.
        timeout_ms: Maximum wall-clock time in milliseconds.
        engine_name: Engine name for the :class:`EngineTimeoutError` message.

    Returns:
        The result of ``fn()``.

    Raises:
        EngineTimeoutError: If ``fn()`` does not complete within ``timeout_ms``.
    """
    result: list = []
    exception: list = []
    timed_out = threading.Event()

    def _target() -> None:
        try:
            result.append(fn())
        except BaseException as exc:
            exception.append(exc)
        finally:
            timed_out.set()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    # Wait with timeout (convert ms to seconds)
    if not timed_out.wait(timeout=timeout_ms / 1000.0):
        raise EngineTimeoutError(engine_name, timeout_ms)

    thread.join(timeout=0.1)

    if exception:
        raise exception[0]

    return result[0]
