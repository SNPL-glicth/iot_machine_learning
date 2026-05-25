"""RUL core logic — estimates time-to-failure from deterioration signals."""

from __future__ import annotations

from typing import Optional

from .models import RULEstimate


class RULEstimator:
    """Estimate remaining useful life from anomaly and drift signals."""

    URGENCY_CRITICAL = "CRITICAL"
    URGENCY_MEDIUM = "MEDIUM"
    URGENCY_LOW = "LOW"

    CONFIDENCE_HIGH = "HIGH"
    CONFIDENCE_MEDIUM = "MEDIUM"
    CONFIDENCE_LOW = "LOW"

    _MIN_HOURS = 0.5
    _MAX_HOURS = 168.0

    def estimate(
        self,
        anomaly_score: float,
        drift_magnitude: float,
        consecutive_anomalies: int,
        sampling_interval_minutes: float = 1.0,
    ) -> Optional[RULEstimate]:
        """Return RULEstimate or None when no deterioration is detected.

        Logic:
            deterioration_rate = drift_magnitude * anomaly_score
            if rate == 0 → None
            hours = clamp(1.0 / rate, 0.5, 168)
            urgency by hours bracket
            confidence by consecutive_anomalies count
        """
        deterioration_rate = drift_magnitude * anomaly_score

        if deterioration_rate == 0.0:
            return None

        raw_hours = 1.0 / deterioration_rate
        hours = max(self._MIN_HOURS, min(self._MAX_HOURS, raw_hours))

        if hours < 4.0:
            urgency = self.URGENCY_CRITICAL
        elif hours <= 24.0:
            urgency = self.URGENCY_MEDIUM
        else:
            urgency = self.URGENCY_LOW

        if consecutive_anomalies >= 10:
            confidence = self.CONFIDENCE_HIGH
        elif consecutive_anomalies >= 5:
            confidence = self.CONFIDENCE_MEDIUM
        else:
            confidence = self.CONFIDENCE_LOW

        human_readable = f"~{hours:.0f}h remaining ({confidence} confidence)"

        return RULEstimate(
            time_to_failure_hours=hours,
            urgency=urgency,
            confidence=confidence,
            deterioration_rate=deterioration_rate,
            human_readable=human_readable,
        )
