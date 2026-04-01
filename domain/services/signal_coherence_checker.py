"""Signal coherence checker — resolves conflicts between prediction and anomaly signals.

Domain service that checks if prediction and anomaly results are coherent.
When they conflict (e.g., anomaly detected but prediction is normal),
resolves the conflict with adjusted confidence and clear reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CoherenceResult:
    """Result of coherence check between prediction and anomaly signals.

    Attributes:
        is_coherent: True if both signals agree, False if conflict detected.
        conflict_type: Type of conflict if any, None if coherent.
        resolved_value: The resolved predicted value (may be same as input).
        resolved_confidence: Confidence after resolution (adjusted if conflict).
        resolution_reason: Human-readable explanation of resolution.
    """

    is_coherent: bool
    conflict_type: Optional[str]
    resolved_value: float
    resolved_confidence: float
    resolution_reason: str


class SignalCoherenceChecker:
    """Checks coherence between prediction and anomaly detection results.

    Stateless service — no I/O, no persistence, no side effects.
    """

    # Confidence adjustment when conflict detected
    CONFLICT_CONFIDENCE_PENALTY: float = 0.3

    def check(
        self,
        predicted_value: float,
        predicted_confidence: float,
        is_anomaly: bool,
        anomaly_score: float,
        historical_values: Optional[list[float]] = None,
    ) -> CoherenceResult:
        """Check coherence between prediction and anomaly signals.

        Core rule: If anomaly is detected but predicted value is within
        normal historical range, this is a conflict — confidence should drop.

        Args:
            predicted_value: The fused predicted value.
            predicted_confidence: Original confidence from fusion.
            is_anomaly: Whether anomaly was detected.
            anomaly_score: Anomaly score (0–1, higher = more anomalous).
            historical_values: Optional historical window for range check.

        Returns:
            CoherenceResult with resolution decision.
        """
        # No anomaly → always coherent
        if not is_anomaly:
            return CoherenceResult(
                is_coherent=True,
                conflict_type=None,
                resolved_value=predicted_value,
                resolved_confidence=predicted_confidence,
                resolution_reason="No anomaly detected — signals coherent",
            )

        # Anomaly detected — check if prediction conflicts
        in_normal_range = self._is_in_normal_range(
            predicted_value, historical_values
        )

        if is_anomaly and in_normal_range:
            # Conflict: anomaly says abnormal, but prediction is normal range
            resolved_conf = max(
                0.1,  # floor confidence
                min(predicted_confidence, self.CONFLICT_CONFIDENCE_PENALTY)
            )
            return CoherenceResult(
                is_coherent=False,
                conflict_type="anomaly_prediction_conflict",
                resolved_value=predicted_value,
                resolved_confidence=resolved_conf,
                resolution_reason=(
                    f"Anomaly detected (score={anomaly_score:.3f}) but "
                    f"predicted value {predicted_value:.3f} is within "
                    f"normal historical range — confidence penalized"
                ),
            )

        # Anomaly detected and prediction also anomalous → coherent
        return CoherenceResult(
            is_coherent=True,
            conflict_type=None,
            resolved_value=predicted_value,
            resolved_confidence=predicted_confidence,
            resolution_reason=(
                f"Anomaly detected (score={anomaly_score:.3f}) and "
                f"prediction consistent with anomalous regime"
            ),
        )

    def _is_in_normal_range(
        self,
        value: float,
        historical_values: Optional[list[float]]
    ) -> bool:
        """Check if value is within normal range of historical values.

        Uses IQR method: normal range = [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        Falls back to min/max if insufficient data.
        """
        if not historical_values or len(historical_values) < 4:
            # With insufficient data, be conservative — assume in range
            return True

        sorted_vals = sorted(historical_values)
        n = len(sorted_vals)

        # Compute Q1 and Q3
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_vals[q1_idx]
        q3 = sorted_vals[q3_idx]

        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return lower_bound <= value <= upper_bound
