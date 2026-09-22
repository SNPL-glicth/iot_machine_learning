"""Consolidated IoT Drift Sensor Adapter — implements DriftSensorPort.

Wraps online drift algorithms (Page-Hinkley, ADWIN, ErrorDrift) to monitor
multichannel physical IoT telemetry with zero ML Core coupling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from domain.ports.rosa_roja.drift_sensor import DriftSensorPort

logger = logging.getLogger(__name__)

DetectorType = Literal["page_hinkley", "adwin", "error_drift"]


def create_drift_detector(detector_type: DetectorType, **kwargs) -> Any:
    """Instantiates underlying drift detection algorithm."""
    if detector_type == "page_hinkley":
        from infrastructure.ml.cognitive.drift.page_hinkley import (
            PageHinkleyConfig,
            PageHinkleyDetector,
        )
        config = PageHinkleyConfig(
            delta=kwargs.get("ph_delta", 0.005),
            lambda_=kwargs.get("ph_lambda", 50.0),
            alpha=kwargs.get("ph_alpha", 0.9999),
        )
        return PageHinkleyDetector(config)

    if detector_type == "adwin":
        from infrastructure.ml.cognitive.drift.adwin import ADWINDetector
        return ADWINDetector(
            delta=kwargs.get("adwin_delta", 0.002),
            max_window_size=kwargs.get("adwin_max_window", 1000),
        )

    if detector_type == "error_drift":
        from infrastructure.ml.cognitive.drift.error_drift_detector import ErrorDriftDetector
        return ErrorDriftDetector(
            window_size=kwargs.get("window_size", 100),
            detector_type=kwargs.get("error_detector_type", "page_hinkley"),
            ph_delta=kwargs.get("ph_delta"),
            ph_lambda=kwargs.get("ph_lambda"),
            ph_alpha=kwargs.get("ph_alpha"),
            adwin_delta=kwargs.get("adwin_delta"),
            adwin_max_window=kwargs.get("adwin_max_window"),
            zscore_threshold=kwargs.get("zscore_threshold"),
        )

    raise ValueError(f"Unknown detector_type: {detector_type}")


class IoTDriftSensorAdapter(DriftSensorPort):
    """Monitors telemetry channels using online drift detectors."""

    def __init__(
        self,
        name: str,
        channels: List[str],
        detector_type: DetectorType = "page_hinkley",
        aggregation: Literal["max", "mean", "weighted"] = "max",
        channel_weights: Optional[Dict[str, float]] = None,
        **detector_kwargs,
    ) -> None:
        self.name = name
        self.channels = channels
        self.detector_type = detector_type
        self.aggregation = aggregation
        self.channel_weights = channel_weights or {}
        self._detectors: Dict[str, Any] = {
            c: create_drift_detector(detector_type, **detector_kwargs)
            for c in channels
        }

    def update(self, actual: float, predicted: float) -> None:
        """Broadcasts observation to all channel detectors."""
        for detector in self._detectors.values():
            if hasattr(detector, "update"):
                if self.detector_type == "error_drift":
                    detector.update(actual, predicted)
                else:
                    detector.update(actual)

    def update_channel(self, channel: str, actual: float, predicted: Optional[float] = None) -> None:
        """Updates a specific telemetry channel."""
        detector = self._detectors.get(channel)
        if not detector:
            return
        if self.detector_type == "error_drift":
            if predicted is not None:
                detector.update(actual, predicted)
        else:
            detector.update(actual)

    def get_drift_score(self) -> float:
        """Calculates normalized aggregate drift score in [0.0, 1.0]."""
        scores = [d.get_drift_score() for d in self._detectors.values() if hasattr(d, "get_drift_score")]
        scores = [s for s in scores if s > 0]
        if not scores:
            return 0.0
        if self.aggregation == "mean":
            return sum(scores) / len(scores)
        return max(scores)

    def get_channel_scores(self) -> Dict[str, float]:
        return {c: d.get_drift_score() if hasattr(d, "get_drift_score") else 0.0 for c, d in self._detectors.items()}

    def reset(self) -> None:
        for detector in self._detectors.values():
            if hasattr(detector, "reset"):
                detector.reset()

    def export_state(self) -> dict:
        return {
            "schema_version": 1,
            "detector_type": self.detector_type,
            "detectors": {c: d.export_state() for c, d in self._detectors.items() if hasattr(d, "export_state")},
        }

    def import_state(self, payload: dict) -> None:
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported drift sensor payload schema")
        if payload.get("detector_type") != self.detector_type:
            raise ValueError(f"Detector mismatch: {payload.get('detector_type')} vs {self.detector_type}")
        states = payload.get("detectors", {})
        for c, d in self._detectors.items():
            if c in states:
                d.import_state(states[c])


DriftSensorAdapter = IoTDriftSensorAdapter
