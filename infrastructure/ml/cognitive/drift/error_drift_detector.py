"""ErrorDriftDetector — concept drift detection based on prediction error."""

from __future__ import annotations

import logging
import math
from typing import Deque, Literal, Optional

from core.parameters.numerical_constants import EPSILON

from .page_hinkley import PageHinkleyConfig, PageHinkleyDetector
from .adwin import ADWINDetector
from .error_window import ErrorWindow
from .error_normalizer import ErrorNormalizer

logger = logging.getLogger(__name__)


class ErrorDriftDetector:
    """Drift detector operating on prediction errors, not raw signal.
    
    Tracks:
        - Rolling MAE (mean absolute error)
        - Rolling variance of error
        - Underlying drift detector (Page-Hinkley or ADWIN) applied to errors
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
        # Lazy-load defaults from centralized config
        cfg = flags
        if cfg is None:
            try:
                from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
                cfg = FeatureFlags()
            except Exception:
                cfg = None
        
        self._error_window = ErrorWindow(window_size)
        self._detector_type = detector_type
        self._normalizer = ErrorNormalizer(
            zscore_threshold if zscore_threshold is not None
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
        
        self._last_drift_result = False
    
    def update(self, y_true: float, y_pred: float) -> None:
        """Record a prediction outcome."""
        if not math.isfinite(y_true) or not math.isfinite(y_pred):
            logger.debug("error_drift_invalid_input_dropped", extra={"y_true": y_true, "y_pred": y_pred})
            return
        
        abs_error = abs(y_true - y_pred)
        self._error_window.add(abs_error)
        
        # Feed normalized residual to underlying detector
        normalized = self._normalizer.normalize(
            abs_error,
            self._error_window.rolling_mae,
            self._error_window.rolling_std,
            self._error_window.n_updates,
        )
        drift_now = self._detector.update(normalized)
        self._last_drift_result = bool(drift_now)
    
    def is_drift_detected(self) -> bool:
        return self._last_drift_result
    
    def get_drift_score(self) -> float:
        if self._error_window.n_updates < 10:
            return 0.0
        if not self._error_window.errors:
            return 0.0
        
        last_error = self._error_window.get_last_error()
        normalized = self._normalizer.normalize(
            last_error,
            self._error_window.rolling_mae,
            self._error_window.rolling_std,
            self._error_window.n_updates,
        )
        return self._normalizer.compute_severity(normalized)
    
    def get_stats(self) -> dict:
        return {
            "rolling_mae": round(self._error_window.rolling_mae, 6),
            "rolling_std": round(self._error_window.rolling_std, 6),
            "n_samples": len(self._error_window.errors),
            "n_updates": self._error_window.n_updates,
            "detector_type": self._detector_type,
        }
    
    def reset(self) -> None:
        self._error_window.clear()
        self._detector.reset()
        self._last_drift_result = False

    # ------------------------------------------------------------------
    # Persistence (StatePersistable contract)
    # ------------------------------------------------------------------

    STATE_SCHEMA_VERSION = 1

    def export_state(self) -> dict:
        """Serialize error window, inner detector state and last result."""
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "error_window": self._error_window.export_state(),
            "inner_detector": self._detector.export_state(),
            "last_drift_result": self._last_drift_result,
        }

    def import_state(self, payload: dict) -> None:
        """Restore error window and inner detector state."""
        if not isinstance(payload, dict) or "schema_version" not in payload:
            raise ValueError("ErrorDriftDetector payload missing schema_version")
        if payload["schema_version"] != self.STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported ErrorDriftDetector schema: {payload['schema_version']}"
            )
        self._error_window.import_state(payload["error_window"])
        self._detector.import_state(payload["inner_detector"])
        self._last_drift_result = bool(payload.get("last_drift_result", False))