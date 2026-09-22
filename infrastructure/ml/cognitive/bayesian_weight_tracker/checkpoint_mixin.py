"""Checkpoint mixin for BayesianWeightTracker."""
from __future__ import annotations

import threading
from typing import Any, Dict

from .checkpoint import WeightTrackerCheckpoint


class CheckpointMixin:
    """Mixin providing checkpoint export/import."""

    # Declare expected parent attributes for mypy
    _persistence: Any
    _accuracy: Dict[str, Any]
    _priors: Dict[str, Any]
    _regime_last_access: Dict[str, Any]
    _regime_last_update: Dict[str, Any]
    _lock: threading.Lock
    _scope: str
    _config: Any

    def persist_immediately(self, engine_name: str, regime: str) -> None:
        """Immediately persist state."""
        self._persistence.persist_immediately(
            regime, engine_name, self._accuracy, self._priors,
            self._regime_last_access, self._regime_last_update,
        )

    def export_checkpoint(self) -> dict:
        """Export state as serializable checkpoint."""
        with self._lock:
            return WeightTrackerCheckpoint.export(
                self._scope, self._accuracy, self._priors,
                self._regime_last_access, self._regime_last_update,
                self._config.alpha, self._config.min_weight,
            )

    def restore_from_checkpoint(self, checkpoint_data: dict) -> None:
        """Restore state from checkpoint."""
        with self._lock:
            WeightTrackerCheckpoint.restore(
                checkpoint_data, self._scope, self._accuracy, self._priors,
                self._regime_last_access, self._regime_last_update,
            )
