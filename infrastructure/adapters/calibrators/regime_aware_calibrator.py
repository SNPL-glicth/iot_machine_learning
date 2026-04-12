"""Regime-Aware Calibrator — Calibrador compuesto por régimen (v2 modular).

Selecciona: Isotonic (STABLE), Platt (TRENDING), Null (VOLATILE).
Mantiene calibradores separados POR (motor, régimen).
"""

from __future__ import annotations

import logging
from threading import RLock
from typing import Dict, Optional

from iot_machine_learning.domain.ports.confidence_calibrator_port import (
    CalibrationStats, CalibratedScore, ConfidenceCalibratorPort, NullCalibrator,
)
from .isotonic_calibrator import IsotonicCalibrator
from .platt_calibrator import PlattCalibrator

logger = logging.getLogger(__name__)


class RegimeAwareCalibrator(ConfidenceCalibratorPort):
    """Composite calibrator with per-(engine, regime) isolation."""
    
    def __init__(
        self,
        min_samples_platt: int = 50,
        min_samples_isotonic: int = 200,
        window_size_platt: int = 500,
        window_size_isotonic: int = 1000,
    ) -> None:
        self._min_samples_platt = min_samples_platt
        self._min_samples_isotonic = min_samples_isotonic
        self._window_size_platt = window_size_platt
        self._window_size_isotonic = window_size_isotonic
        
        self._calibrators: Dict[tuple[str, str], ConfidenceCalibratorPort] = {}
        self._lock = RLock()
        self._null = NullCalibrator()
    
    def calibrate(self, raw_score: float) -> CalibratedScore:
        """Calibrate using generic TRENDING calibrator."""
        return self.calibrate_for(raw_score, "generic", "TRENDING")
    
    def update(self, raw_score: float, actual_outcome: float) -> None:
        """Update generic TRENDING calibrator."""
        self.update_for(raw_score, actual_outcome, "generic", "TRENDING")
    
    def is_ready(self) -> bool:
        """Check generic calibrator."""
        return self.is_ready_for("generic", "TRENDING")
    
    def get_stats(self) -> CalibrationStats:
        """Get stats from generic calibrator."""
        return self.get_stats_for("generic", "TRENDING")
    
    def reset(self) -> None:
        """Reset all calibrators."""
        self.reset_for()
    
    # Extended methods
    
    def calibrate_for(
        self, raw_score: float, engine_name: str = "unknown", regime: str = "unknown",
    ) -> CalibratedScore:
        """Calibrate using (engine, regime) specific calibrator."""
        try:
            return self._get_or_create(engine_name, regime).calibrate(raw_score)
        except Exception:
            return CalibratedScore(
                calibrated_score=max(0.0, min(1.0, raw_score)),
                raw_score=raw_score, calibration_applied=False, calibration_delta=0.0,
            )
    
    def update_for(
        self, raw_score: float, actual_outcome: float, engine_name: str = "unknown", regime: str = "unknown",
    ) -> None:
        """Update (engine, regime) specific calibrator."""
        try:
            self._get_or_create(engine_name, regime).update(raw_score, actual_outcome)
        except Exception:
            pass
    
    def is_ready_for(self, engine_name: str = "unknown", regime: str = "unknown") -> bool:
        """Check if (engine, regime) calibrator is ready."""
        try:
            return self._get_or_create(engine_name, regime).is_ready()
        except Exception:
            return False
    
    def get_stats_for(self, engine_name: str = "unknown", regime: str = "unknown") -> CalibrationStats:
        """Get stats from (engine, regime) calibrator."""
        try:
            return self._get_or_create(engine_name, regime).get_stats()
        except Exception:
            return CalibrationStats(
                n_samples=0, ece=0.0, reliability={},
                is_ready=False, calibrator_type="error",
            )
    
    def reset_for(
        self, engine_name: Optional[str] = None, regime: Optional[str] = None,
    ) -> None:
        """Reset calibrators. None=reset all."""
        with self._lock:
            if engine_name is None:
                for c in self._calibrators.values():
                    c.reset()
                self._calibrators.clear()
            elif regime is None:
                for key in list(self._calibrators.keys()):
                    if key[0] == engine_name:
                        self._calibrators[key].reset()
                        del self._calibrators[key]
            elif (engine_name, regime) in self._calibrators:
                self._calibrators[(engine_name, regime)].reset()
                del self._calibrators[(engine_name, regime)]
    
    def get_all_stats(self) -> Dict[tuple[str, str], CalibrationStats]:
        """Get stats from all calibrators."""
        with self._lock:
            return {k: v.get_stats() for k, v in self._calibrators.items()}
    
    def _get_or_create(self, engine: str, regime: str) -> ConfidenceCalibratorPort:
        """Get or create calibrator for (engine, regime)."""
        key = (engine, regime)
        with self._lock:
            if key not in self._calibrators:
                self._calibrators[key] = self._create(engine, regime)
            return self._calibrators[key]
    
    def _create(self, engine_name: str, regime: str) -> ConfidenceCalibratorPort:
        """Create appropriate calibrator for regime."""
        r = regime.upper()
        if r == "STABLE":
            return IsotonicCalibrator(
                self._window_size_isotonic, self._min_samples_isotonic,
                f"{engine_name}_STABLE",
            )
        elif r == "VOLATILE":
            return self._null
        else:  # TRENDING, NOISY, etc.
            return PlattCalibrator(
                self._window_size_platt, self._min_samples_platt, 10,
                f"{engine_name}_{r}",
            )
