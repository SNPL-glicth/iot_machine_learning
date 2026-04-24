"""Error store package — single source of truth for engine prediction errors.

Public API::

    from iot_machine_learning.infrastructure.ml.cognitive.error_store import (
        EngineErrorStore,
    )

Invariant: every raw ``abs(predicted - actual)`` float that the cognitive
pipeline wishes to persist flows through ``EngineErrorStore.record``.
All consumers (inhibition, reliability, adaptor) read through
``get_recent`` / ``get_percentile`` / ``get_rmse_window``.
"""

from __future__ import annotations

from .engine_error_store import EngineErrorStore

__all__ = ["EngineErrorStore"]
