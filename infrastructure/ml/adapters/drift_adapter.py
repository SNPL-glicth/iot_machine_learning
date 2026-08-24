"""IoT Drift Sensor Adapter for Rosa Roja.

Wraps existing drift detectors (Page-Hinkley, ADWIN, ErrorDriftDetector)
to implement DriftSensorPort for telemetry channels (temperature, vibration, pressure, etc.).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Literal

from core.orchestration.rosa_roja.ports.drift_sensor import DriftSensorPort

logger = logging.getLogger(__name__)


class IoTDriftSensorAdapter:
    """IoT Drift Sensor Adapter implementing DriftSensorPort.
    
    Manages one drift detector per telemetry channel. Aggregates
    per-channel drift scores into a single normalized score.
    
    Example:
        adapter = IoTDriftSensorAdapter(
            channels=["channel_1", "channel_2", "channel_3"],
            detector_type="page_hinkley"
        )
        adapter.update(actual_values, predicted_values)
        score = adapter.get_drift_score()  # Returns max or mean across channels
    """
    
    def __init__(
        self,
        name: str,
        channels: List[str],
        detector_type: Literal["page_hinkley", "adwin", "error_drift"] = "page_hinkley",
        aggregation: Literal["max", "mean", "weighted"] = "max",
        channel_weights: Optional[Dict[str, float]] = None,
        **detector_kwargs,
    ):
        """
        Args:
            name: Identifier for this drift sensor group.
            channels: List of telemetry channel names to monitor.
            detector_type: Type of underlying drift detector.
            aggregation: How to combine per-channel scores ("max", "mean", "weighted").
            channel_weights: Optional weights for "weighted" aggregation.
            **detector_kwargs: Passed to underlying detector constructor.
        """
        self.name = name
        self.channels = channels
        self.detector_type = detector_type
        self.aggregation = aggregation
        self.channel_weights = channel_weights or {}
        
        # Create one detector per channel
        self._detectors: Dict[str, object] = {}
        for channel in channels:
            self._detectors[channel] = self._create_detector(**detector_kwargs)
    
    def _create_detector(self, **kwargs):
        """Create a new drift detector instance."""
        if self.detector_type == "page_hinkley":
            from infrastructure.ml.cognitive.drift.page_hinkley import (
                PageHinkleyConfig, PageHinkleyDetector
            )
            config = PageHinkleyConfig(
                delta=kwargs.get("ph_delta", 0.005),
                lambda_=kwargs.get("ph_lambda", 50.0),
                alpha=kwargs.get("ph_alpha", 0.9999),
            )
            return PageHinkleyDetector(config)
        
        elif self.detector_type == "adwin":
            from infrastructure.ml.cognitive.drift.adwin import ADWINDetector
            return ADWINDetector(
                delta=kwargs.get("adwin_delta", 0.002),
                max_window_size=kwargs.get("adwin_max_window", 1000),
            )
        
        elif self.detector_type == "error_drift":
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
        
        else:
            raise ValueError(f"Unknown detector_type: {self.detector_type}")
    
    def update(self, actual: float, predicted: float) -> None:
        """
        Update all channel detectors with actual/predicted pair.
        
        Note: This assumes single-channel or broadcast update. For per-channel
        updates, use `update_channel()` directly.
        
        Args:
            actual: Ground truth value (broadcast to all channels).
            predicted: Predicted value (broadcast to all channels).
        """
        # Broadcast to all channels - useful for unified model predictions
        for channel, detector in self._detectors.items():
            if hasattr(detector, 'update'):
                if self.detector_type == "error_drift":
                    detector.update(actual, predicted)
                else:
                    # Page-Hinkley/ADWIN take single value - use actual
                    detector.update(actual)
    
    def update_channel(self, channel: str, actual: float, predicted: float = None) -> None:
        """Update a specific channel's detector.
        
        Args:
            channel: Channel name (must be in self.channels).
            actual: Ground truth value for this channel.
            predicted: Predicted value (only used for error_drift detector).
        """
        if channel not in self._detectors:
            logger.warning(f"Channel '{channel}' not found in drift sensor '{self.name}'")
            return
        
        detector = self._detectors[channel]
        if self.detector_type == "error_drift":
            if predicted is None:
                logger.warning(f"error_drift detector requires predicted value for channel '{channel}'")
                return
            detector.update(actual, predicted)
        else:
            detector.update(actual)
    
    def update_channels(self, actuals: Dict[str, float], predicted: Dict[str, float] = None) -> None:
        """Update multiple channels at once.
        
        Args:
            actuals: Dict mapping channel -> actual value.
            predicted: Optional dict mapping channel -> predicted value (for error_drift).
        """
        for channel, actual in actuals.items():
            pred = predicted.get(channel) if predicted else None
            self.update_channel(channel, actual, pred)
    
    def get_drift_score(self) -> float:
        """
        Returns aggregated normalized DriftScore_t in [0.0, 1.0].
        
        Aggregation methods:
        - "max": Maximum drift across channels (conservative, alerts on any)
        - "mean": Average drift across channels
        - "weighted": Weighted average using channel_weights
        """
        scores = []
        for channel in self.channels:
            detector = self._detectors.get(channel)
            if detector and hasattr(detector, 'get_drift_score'):
                score = detector.get_drift_score()
                if score > 0:
                    scores.append(score)
        
        if not scores:
            return 0.0
        
        if self.aggregation == "max":
            return max(scores)
        elif self.aggregation == "mean":
            return sum(scores) / len(scores)
        elif self.aggregation == "weighted":
            total_weight = 0.0
            weighted_sum = 0.0
            for channel in self.channels:
                weight = self.channel_weights.get(channel, 1.0)
                detector = self._detectors.get(channel)
                if detector and hasattr(detector, 'get_drift_score'):
                    score = detector.get_drift_score()
                    weighted_sum += score * weight
                    total_weight += weight
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return max(scores)  # fallback
    
    def get_channel_scores(self) -> Dict[str, float]:
        """Get individual drift scores per channel for diagnostics."""
        return {
            channel: detector.get_drift_score() if hasattr(detector, 'get_drift_score') else 0.0
            for channel, detector in self._detectors.items()
        }
    
    def reset(self) -> None:
        """Reset all channel detectors."""
        for detector in self._detectors.values():
            if hasattr(detector, 'reset'):
                detector.reset()
    
    def reset_channel(self, channel: str) -> None:
        """Reset a specific channel's detector."""
        detector = self._detectors.get(channel)
        if detector and hasattr(detector, 'reset'):
            detector.reset()
    
    def is_any_drift_detected(self) -> bool:
        """Check if any channel has drift detected."""
        for detector in self._detectors.values():
            if hasattr(detector, 'is_drift_detected') and detector.is_drift_detected():
                return True
        return False
    
    def get_stats(self) -> Dict[str, dict]:
        """Get statistics for all channel detectors."""
        stats = {}
        for channel, detector in self._detectors.items():
            if hasattr(detector, 'get_stats'):
                stats[channel] = detector.get_stats()
            else:
                stats[channel] = {"drift_score": detector.get_drift_score() if hasattr(detector, 'get_drift_score') else 0.0}
        return stats

    # ------------------------------------------------------------------
    # Persistence (StatePersistable contract)
    # ------------------------------------------------------------------

    def export_state(self) -> dict:
        """Serialize per-channel detector state keyed by channel name."""
        state: Dict[str, object] = {
            "schema_version": 1,
            "detector_type": self.detector_type,
        }
        detectors_state = {}
        for channel, detector in self._detectors.items():
            if hasattr(detector, "export_state"):
                detectors_state[channel] = detector.export_state()
            else:
                raise ValueError(
                    f"Detector for channel '{channel}' does not support persistence"
                )
        state["detectors"] = detectors_state
        return state

    def import_state(self, payload: dict) -> None:
        """Restore per-channel detector state.

        Channel names and detector type must match the live configuration;
        mismatches indicate a snapshot from a different sensor setup.
        """
        if not isinstance(payload, dict) or "schema_version" not in payload:
            raise ValueError("IoTDriftSensorAdapter payload missing schema_version")
        if payload["schema_version"] != 1:
            raise ValueError(
                f"Unsupported drift sensor schema: {payload['schema_version']}"
            )
        if payload.get("detector_type") != self.detector_type:
            raise ValueError(
                f"Detector type mismatch: snapshot={payload.get('detector_type')!r}, "
                f"live={self.detector_type!r}"
            )
        detectors_state = payload.get("detectors", {})
        if not isinstance(detectors_state, dict):
            raise ValueError("Drift sensor payload 'detectors' must be a dict")
        missing = set(self._detectors) - set(detectors_state)
        if missing:
            raise ValueError(f"Snapshot missing channels: {sorted(missing)}")
        for channel, detector in self._detectors.items():
            detector.import_state(detectors_state[channel])


# Backward compatibility alias
DriftSensorAdapter = IoTDriftSensorAdapter