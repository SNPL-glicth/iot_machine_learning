"""Port for calibrator repository persistence (DIP decoupling)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from iot_machine_learning.domain.entities.market.calibration import (
    CalibrationComparison,
    ContextCalibrator,
)


@runtime_checkable
class CalibratorRepositoryPort(Protocol):
    """Protocol for calibrator repository operations."""

    def get_active_calibrator_v2(self) -> Optional[Any]:
        """Get currently active calibrator version metadata."""
        ...

    def save_calibrator_v2(
        self,
        calibrator: ContextCalibrator,
        comparison: CalibrationComparison,
        description: str,
        metadata: Dict[str, Any],
    ) -> str:
        """Save a new calibrator and return calibrator_id."""
        ...


class InMemoryCalibratorRepository(CalibratorRepositoryPort):
    """In-memory fallback implementation for CalibratorRepositoryPort."""

    def __init__(self) -> None:
        self._active_calibrator: Optional[Any] = None
        self._calibrators: Dict[str, Any] = {}

    def get_active_calibrator_v2(self) -> Optional[Any]:
        return self._active_calibrator

    def save_calibrator_v2(
        self,
        calibrator: ContextCalibrator,
        comparison: CalibrationComparison,
        description: str,
        metadata: Dict[str, Any],
    ) -> str:
        new_id = f"calibrator_v{len(self._calibrators) + 1}"
        self._calibrators[new_id] = {
            "calibrator": calibrator,
            "comparison": comparison,
            "description": description,
            "metadata": metadata,
        }
        return new_id
