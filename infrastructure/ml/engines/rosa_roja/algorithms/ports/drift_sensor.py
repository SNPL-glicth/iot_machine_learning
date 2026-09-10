"""DriftSensorPort protocol for drift detectors."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DriftSensorPort(Protocol):
    """Protocol implemented by cognitive drift detectors (Page-Hinkley, ADWIN, ErrorDriftDetector)."""
    name: str

    def get_drift_score(self) -> float:
        """Returns normalized DriftScore_t in [0.0, 1.0]."""
        ...

    def update(self, actual: float, predicted: float) -> None:
        """Update detector with new observation pair."""
        ...

    def reset(self) -> None:
        """Reset detector state after confirmed drift response."""
        ...