"""Platt Calibrator — Platt Scaling con SGD online (v2 modular).

Implementación simplificada usando utilidades matemáticas externas.
"""

from __future__ import annotations

import logging
from collections import deque
from threading import RLock
from typing import Deque

import numpy as np

from iot_machine_learning.domain.ports.confidence_calibrator_port import (
    CalibrationStats,
    CalibratedScore,
    ConfidenceCalibratorPort,
)
from .utils import fit_platt_params, platt_sigmoid, compute_ece_numpy

logger = logging.getLogger(__name__)


class PlattCalibrator(ConfidenceCalibratorPort):
    """Platt Scaling calibrator con aprendizaje online."""
    
    def __init__(
        self,
        window_size: int = 500,
        min_samples: int = 50,
        update_frequency: int = 10,
        engine_name: str = "unknown",
    ) -> None:
        self._window_size = window_size
        self._min_samples = min_samples
        self._update_frequency = update_frequency
        self._engine_name = engine_name
        
        self._scores: Deque[float] = deque(maxlen=window_size)
        self._outcomes: Deque[float] = deque(maxlen=window_size)
        self._A: float = 0.0
        self._B: float = 0.0
        self._update_count: int = 0
        self._fitted: bool = False
        self._last_ece: float = 0.0
        self._lock = RLock()
    
    def calibrate(self, raw_score: float) -> CalibratedScore:
        """Calibrar score usando Platt Scaling."""
        with self._lock:
            if len(self._scores) < self._min_samples or not self._fitted:
                return CalibratedScore(
                    calibrated_score=max(0.0, min(1.0, raw_score)),
                    raw_score=raw_score,
                    calibration_applied=False,
                    calibration_delta=0.0,
                )
            
            try:
                calibrated = float(platt_sigmoid(np.array([raw_score]), self._A, self._B)[0])
                return CalibratedScore(
                    calibrated_score=calibrated,
                    raw_score=raw_score,
                    calibration_applied=True,
                    calibration_delta=calibrated - raw_score,
                )
            except Exception:
                return CalibratedScore(
                    calibrated_score=max(0.0, min(1.0, raw_score)),
                    raw_score=raw_score,
                    calibration_applied=False,
                    calibration_delta=0.0,
                )
    
    def update(self, raw_score: float, actual_outcome: float) -> None:
        """Actualizar calibrador con nueva observación."""
        with self._lock:
            self._scores.append(raw_score)
            self._outcomes.append(actual_outcome)
            self._update_count += 1
            
            if self._update_count % self._update_frequency == 0:
                if len(self._scores) >= self._min_samples:
                    self._fit()
    
    def is_ready(self) -> bool:
        """True si hay suficientes muestras."""
        with self._lock:
            return len(self._scores) >= self._min_samples and self._fitted
    
    def get_stats(self) -> CalibrationStats:
        """Obtener estadísticas de calibración."""
        with self._lock:
            n = len(self._scores)
            if n < self._min_samples:
                return CalibrationStats(
                    n_samples=n, ece=0.0, reliability={},
                    is_ready=False, calibrator_type="platt",
                )
            
            scores_arr = np.array(self._scores)
            outcomes_arr = np.array(self._outcomes)
            calibrated_arr = platt_sigmoid(scores_arr, self._A, self._B)
            ece, reliability = compute_ece_numpy(calibrated_arr, outcomes_arr)
            self._last_ece = ece
            
            return CalibrationStats(
                n_samples=n, ece=ece, reliability=reliability,
                is_ready=self._fitted, calibrator_type="platt",
            )
    
    def reset(self) -> None:
        """Resetear calibrador."""
        with self._lock:
            self._scores.clear()
            self._outcomes.clear()
            self._A = 0.0
            self._B = 0.0
            self._update_count = 0
            self._fitted = False
            self._last_ece = 0.0
    
    def _fit(self) -> None:
        """Ajustar parámetros A y B."""
        try:
            self._A, self._B = fit_platt_params(
                np.array(self._scores),
                np.array(self._outcomes),
            )
            self._fitted = True
        except Exception:
            pass  # Keep previous params if fit fails
