"""Capa de Calibración Adaptativa de Anomalías.

Coordina la transformación s_i -> a_i en O(1) y aplica los guardrails de congelamiento.
"""
from __future__ import annotations

import logging

import numpy as np

from .guardrail import CalibrationGuardrail, CalibrationState
from .profile import DetectorCalibrationProfile, compute_calibrated_score

logger = logging.getLogger(__name__)


class AdaptiveDetectorCalibrationLayer:
    """Capa centralizada que transforma raw scores en calibrated anomaly scores a_i(x)."""

    def __init__(self, guardrail: CalibrationGuardrail | None = None) -> None:
        self._profiles: dict[str, DetectorCalibrationProfile] = {}
        self._guardrail = guardrail or CalibrationGuardrail()

    @property
    def guardrail(self) -> CalibrationGuardrail:
        return self._guardrail

    @property
    def state(self) -> CalibrationState:
        return self._guardrail.state

    def set_profile(self, detector_name: str, profile: DetectorCalibrationProfile) -> None:
        """Registra un perfil de calibración específico para un detector."""
        self._profiles[detector_name] = profile

    def get_profile(self, detector_name: str) -> DetectorCalibrationProfile | None:
        """Obtiene el perfil registrado para un detector."""
        return self._profiles.get(detector_name)

    def calibrate_detector(
        self,
        detector_name: str,
        historical_raw_scores: list[float],
        nominal_quantile: float = 0.95,
        anomaly_quantile: float = 0.998,
        is_inverted: bool = False,
    ) -> DetectorCalibrationProfile:
        """Ajusta empíricamente los umbrales de calibración a partir de datos de entrenamiento.

        Args:
            detector_name: Nombre único del subdetector.
            historical_raw_scores: Puntuaciones crudas recolectadas en el set de entrenamiento.
            nominal_quantile: Cuantil para definir la cota superior del comportamiento nominal.
            anomaly_quantile: Cuantil para definir la saturación de anomalía severa.
            is_inverted: True si menores scores indican mayor anomalía (e.g. IF, LOF).
        """
        valid_scores = [s for s in historical_raw_scores if s is not None and np.isfinite(s)]
        if len(valid_scores) < 5:
            # Fallback seguro si hay datos insuficientes
            profile = DetectorCalibrationProfile(tau_nominal=0.0, tau_anomaly=1.0, is_inverted=is_inverted)
            self._profiles[detector_name] = profile
            return profile

        scores_arr = np.array(valid_scores)
        if detector_name == "cumulative_residual":
            profile = DetectorCalibrationProfile(
                tau_nominal=0.5,
                tau_anomaly=1.0,
                is_inverted=False,
            )
            self._profiles[detector_name] = profile
            return profile

        if is_inverted:
            # Para IF/LOF: inliers tienen scores altos (> 0), outliers scores muy negativos
            # tau_nominal es el cuantil donde termina el comportamiento normal
            # tau_anomaly es la cola izquierda extrema
            tau_nom = float(np.percentile(scores_arr, (1.0 - nominal_quantile) * 100.0))
            tau_anom = float(np.percentile(scores_arr, (1.0 - anomaly_quantile) * 100.0))
            if tau_nom <= tau_anom:
                tau_nom = tau_anom + 0.1
        else:
            tau_nom = float(np.percentile(scores_arr, nominal_quantile * 100.0))
            tau_anom = float(np.percentile(scores_arr, anomaly_quantile * 100.0))
            if tau_anom <= tau_nom:
                tau_anom = tau_nom + 0.1

        profile = DetectorCalibrationProfile(
            tau_nominal=tau_nom,
            tau_anomaly=tau_anom,
            is_inverted=is_inverted,
        )
        self._profiles[detector_name] = profile
        logger.debug(
            "detector_calibration_profile_set",
            extra={"detector": detector_name, "tau_nom": tau_nom, "tau_anom": tau_anom, "inverted": is_inverted},
        )
        return profile

    def transform(self, detector_name: str, raw_score: float | None) -> float | None:
        """Aplica la Ecuación de Calibración s_i -> a_i en O(1).

        Args:
            detector_name: Nombre del detector.
            raw_score: Evidencia continua bruta o None.

        Returns:
            Calibrated anomaly score a_i en [0.0, 1.0], o None si no hay score.
        """
        if raw_score is None:
            return None

        profile = self._profiles.get(detector_name)
        if profile is None:
            # Fallback transparente si no se ha calibrado ese detector
            return float(max(0.0, min(1.0, raw_score)))

        return compute_calibrated_score(raw_score, profile)

    def observe(self, value: float, is_anomaly_active: bool = False) -> CalibrationState:
        """Actualiza el guardrail de seguridad con la lectura actual del stream."""
        return self._guardrail.observe(value, is_anomaly_active=is_anomaly_active)
