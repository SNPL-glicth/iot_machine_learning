"""Checkpoint mixin for BayesianWeightTracker."""
from __future__ import annotations
from .checkpoint import WeightTrackerCheckpoint


class CheckpointMixin:
    """Mixin providing checkpoint export/import."""

    def persist_immediately(self, engine_name: str, regime: str) -> None:
        """Immediately persist state."""
        self._persistence.persist_immediately(
            regime, engine_name, self._accuracy, self._priors,
            self._regime_last_access, self._regime_last_update,
        )

    def export_checkpoint(self) -> dict:
        """Export state as serializable checkpoint."""
        return WeightTrackerCheckpoint.export(
            self._scope, self._accuracy, self._priors,
            self._regime_last_access, self._regime_last_update,
            self._config.alpha, self._config.min_weight,
        )

    def restore_from_checkpoint(self, checkpoint_data: dict) -> None:
        """Restore state from checkpoint."""
        WeightTrackerCheckpoint.restore(
            checkpoint_data, self._scope, self._accuracy, self._priors,
            self._regime_last_access, self._regime_last_update,
        )
