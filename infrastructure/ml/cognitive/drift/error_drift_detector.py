"""ErrorDriftDetector — concept drift detection based on prediction error.

Detects degradation in model performance by monitoring the error stream
(y_true, y_pred) instead of raw signal behavior. This is statistically
principled: concept drift should be defined by worsening predictions,
not by changes in the underlying signal (which may be benign).

Tracks:
    - Rolling MAE (mean absolute error)
    - Rolling variance of error
    - Underlying drift detector (Page-Hinkley or ADWIN) applied to errors

Usage:
    detector = ErrorDriftDetector(window_size=100, detector_type="page_hinkley")
    detector.update(y_true=42.0, y_pred=40.5)
    detector.update(y_true=43.0, y_pred=45.0)
    if detector.is_drift_detected():
        score = detector.get_drift_score()
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Deque, Literal, Optional

import numpy as np

from .page_hinkley import PageHinkleyConfig, PageHinkleyDetector
from .adwin import ADWINDetector

logger = logging.getLogger(__name__)


def _safe_mean(values: Deque[float]) -> float:
    """Mean with NaN/Inf guards."""
    if not values:
        return 0.0
    valid = [v for v in values if math.isfinite(v)]
    if not valid:
        return 0.0
    return float(np.mean(valid))


def _safe_std(values: Deque[float]) -> float:
    """Std with NaN/Inf guards and small-sample protection."""
    if len(values) < 2:
        return 0.0
    valid = [v for v in values if math.isfinite(v)]
    if len(valid) < 2:
        return 0.0
    return float(np.std(valid, ddof=1))


class ErrorDriftDetector:
    """Drift detector operating on prediction errors, not raw signal.

    Maintains a bounded rolling window of absolute errors. Computes
    rolling MAE and rolling std as auxiliary diagnostics. The primary
    drift signal comes from an underlying online detector (Page-Hinkley
    or ADWIN) fed with normalized error residuals.

    Args:
        window_size: Size of rolling window for MAE/variance.
        detector_type: "page_hinkley" or "adwin".
        ph_delta: Page-Hinkley delta (sensitivity).
        ph_lambda: Page-Hinkley threshold.
        ph_alpha: Page-Hinkley forgetting factor.
        adwin_delta: ADWIN confidence parameter.
        adwin_max_window: ADWIN maximum window size.
        zscore_threshold: Threshold for declaring drift from normalized error.
    """

    def __init__(
        self,
        window_size: int = 100,
        detector_type: Literal["page_hinkley", "adwin"] = "page_hinkley",
        ph_delta: Optional[float] = None,
        ph_lambda: Optional[float] = None,
        ph_alpha: Optional[float] = None,
        adwin_delta: Optional[float] = None,
        adwin_max_window: Optional[int] = None,
        zscore_threshold: Optional[float] = None,
        flags: Optional["FeatureFlags"] = None,
    ) -> None:
        if window_size < 10:
            raise ValueError(f"window_size must be ≥ 10, got {window_size}")

        # Lazy-load defaults from centralized config to avoid magic numbers
        cfg = flags
        if cfg is None:
            try:
                from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
                cfg = FeatureFlags()
            except Exception:
                cfg = None

        self._window_size = window_size
        self._errors: Deque[float] = deque(maxlen=window_size)
        self._detector_type = detector_type
        self._zscore_threshold = (
            zscore_threshold
            if zscore_threshold is not None
            else getattr(cfg, "ML_DRIFT_ZSCORE_THRESHOLD", 3.0)
        )

        if detector_type == "page_hinkley":
            config = PageHinkleyConfig(
                delta=ph_delta if ph_delta is not None else getattr(cfg, "ML_DRIFT_PH_DELTA", 0.005),
                lambda_=ph_lambda if ph_lambda is not None else getattr(cfg, "ML_DRIFT_PH_LAMBDA", 50.0),
                alpha=ph_alpha if ph_alpha is not None else getattr(cfg, "ML_DRIFT_PH_ALPHA", 0.9999),
            )
            self._detector = PageHinkleyDetector(config)
        elif detector_type == "adwin":
            self._detector = ADWINDetector(
                delta=adwin_delta if adwin_delta is not None else getattr(cfg, "ML_DRIFT_ADWIN_DELTA", 0.002),
                max_window_size=adwin_max_window if adwin_max_window is not None else getattr(cfg, "ML_DRIFT_ADWIN_MAX_WINDOW", 1000),
            )
        else:
            raise ValueError(f"Unknown detector_type: {detector_type}")

        # Running statistics for diagnostics
        self._rolling_mae = 0.0
        self._rolling_std = 0.0
        self._error_variance = 0.0
        self._n_updates = 0

    def update(self, y_true: float, y_pred: float) -> None:
        """Record a prediction outcome.

        Args:
            y_true: Ground truth value.
            y_pred: Predicted value.
        """
        # Guard: NaN/Inf inputs are silently dropped (defensive)
        if not math.isfinite(y_true) or not math.isfinite(y_pred):
            logger.debug(
                "error_drift_invalid_input_dropped",
                extra={"y_true": y_true, "y_pred": y_pred},
            )
            return

        abs_error = abs(y_true - y_pred)
        self._errors.append(abs_error)
        self._n_updates += 1

        # Update rolling statistics
        self._rolling_mae = _safe_mean(self._errors)
        self._rolling_std = _safe_std(self._errors)

        # Feed normalized residual to underlying detector
        # Normalize by rolling std to make detector scale-invariant
        normalized = self._normalize_error(abs_error)
        drift_now = self._detector.update(normalized)
        self._last_drift_result = bool(drift_now)

    def _normalize_error(self, error: float) -> float:
        """Normalize error to make drift detector scale-invariant.

        Uses z-score normalization against rolling statistics.
        With insufficient history (< 10 samples), returns raw error.
        """
        if self._n_updates < 10 or self._rolling_std < 1e-12:
            return error

        z = (error - self._rolling_mae) / self._rolling_std
        # Clamp extreme z-scores to prevent detector saturation
        return float(max(-10.0, min(10.0, z)))

    def is_drift_detected(self) -> bool:
        """True if the underlying detector has flagged drift."""
        # ADWIN: update() already returns bool; we cache the last result
        # Page-Hinkley: update() returns bool
        # Both detectors expose their state through the update return,
        # but we need the last call's result. Re-run on latest state?
        # Actually, both detectors update incrementally; we can't replay.
        # We'll track the last result in update().
        return getattr(self, "_last_drift_result", False)

    def get_drift_score(self) -> float:
        """Normalized drift severity score ∈ [0, ∞).

        0 = no drift, >1.5 = severe drift (used by GradualDriftResponse).
        Derived from normalized error deviation relative to threshold.
        """
        if self._n_updates < 10:
            return 0.0

        if not self._errors:
            return 0.0

        last_error = self._errors[-1]
        normalized = self._normalize_error(last_error)

        # Severity = |normalized| / threshold
        # This maps z=3 → severity=1.0, z=6 → severity=2.0
        severity = abs(normalized) / self._zscore_threshold if self._zscore_threshold > 0 else 0.0
        return float(max(0.0, severity))

    def get_stats(self) -> dict:
        """Return current rolling statistics for diagnostics."""
        return {
            "rolling_mae": round(self._rolling_mae, 6),
            "rolling_std": round(self._rolling_std, 6),
            "n_samples": len(self._errors),
            "n_updates": self._n_updates,
            "detector_type": self._detector_type,
        }

    def reset(self) -> None:
        """Reset detector state (e.g., after confirmed drift response)."""
        self._errors.clear()
        self._detector.reset()
        self._rolling_mae = 0.0
        self._rolling_std = 0.0
        self._error_variance = 0.0
        self._n_updates = 0
        self._last_drift_result = False
