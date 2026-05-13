"""Calibrador de pesos de ensemble basado en tasas de detección.

Principio: Single Responsibility - solo calibra pesos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from core.parameters.numerical_constants import EPSILON, STAT_THRESHOLDS


@dataclass
class DetectionRateProfile:
    """Perfil de tasa de detección de un detector."""
    detector_name: str
    expected_rate: float  # Tasa esperada bajo distribución normal
    empirical_rate: float  # Tasa observada en datos reales
    samples_evaluated: int
    
    def rate_ratio(self) -> float:
        """Ratio entre tasa empírica y esperada."""
        if self.expected_rate < EPSILON.DIVISION:
            return 1.0
        return self.empirical_rate / self.expected_rate


@dataclass
class CalibratedWeights:
    """Pesos calibrados del ensemble."""
    raw_weights: Dict[str, float]  # Pesos originales
    calibrated_weights: Dict[str, float]  # Pesos ajustados
    calibration_factors: Dict[str, float]  # Factores aplicados
    
    def validate(self) -> bool:
        """Valida que pesos sumen 1.0."""
        total = sum(self.calibrated_weights.values())
        return abs(total - 1.0) < EPSILON.COMPARISON


class EnsembleCalibrator:
    """Calibra pesos de ensemble por tasas de detección."""
    
    def __init__(self, target_detection_rate: float = 0.01):
        """
        Args:
            target_detection_rate: Tasa objetivo global (1% por defecto)
        """
        self.target_rate = target_detection_rate
    
    def compute_expected_rates(self) -> Dict[str, float]:
        """Calcula tasas esperadas bajo distribución normal."""
        z_upper = STAT_THRESHOLDS.Z_SCORE_UPPER
        
        # Tasa de z-score: P(|Z| > z_upper) bajo normalidad
        # Para z=3.0: ~0.27% de cada cola = 0.54% total
        try:
            from scipy import stats
            z_score_rate = 2 * (1 - stats.norm.cdf(z_upper))
        except ImportError:
            # Fallback si scipy no está disponible: aproximar
            z_score_rate = 0.0054  # Aproximación para z=3.0
        
        return {
            'z_score': z_score_rate,
            'iqr': 0.007,  # Tukey fences detectan ~0.7% bajo normalidad
            'isolation_forest': STAT_THRESHOLDS.CONTAMINATION_DEFAULT,
            'local_outlier_factor': 0.01,  # Asumido
            'velocity_z': z_score_rate,
            'acceleration_z': z_score_rate,
        }
    
    def calibrate_by_detection_rate(
        self,
        raw_weights: Dict[str, float],
        detection_profiles: List[DetectionRateProfile]
    ) -> CalibratedWeights:
        """Calibra pesos para balancear contribuciones por tasa de detección.
        
        Estrategia:
        1. Detectores con alta tasa (muchos positivos) → peso reducido
        2. Detectores con baja tasa (pocos positivos) → peso aumentado
        3. Normalizar para que suma = 1.0
        """
        profile_map = {p.detector_name: p for p in detection_profiles}
        calibration_factors = {}
        
        for name, raw_weight in raw_weights.items():
            if name not in profile_map:
                calibration_factors[name] = 1.0
                continue
            
            profile = profile_map[name]
            rate_ratio = profile.rate_ratio()
            
            # Inverso de rate_ratio para balancear
            if rate_ratio > EPSILON.DIVISION:
                calibration_factors[name] = 1.0 / rate_ratio
            else:
                calibration_factors[name] = 1.0
        
        # Aplicar factores
        calibrated = {
            name: raw_weights[name] * calibration_factors[name]
            for name in raw_weights
        }
        
        # Normalizar
        total = sum(calibrated.values())
        if total > EPSILON.DIVISION:
            calibrated = {
                name: weight / total
                for name, weight in calibrated.items()
            }
        
        return CalibratedWeights(
            raw_weights=raw_weights,
            calibrated_weights=calibrated,
            calibration_factors=calibration_factors
        )


class DetectionRateMeasurer:
    """Mide tasas de detección empíricas en datos."""
    
    def measure_rates(
        self,
        detectors: Dict[str, object],
        data: np.ndarray,
        min_samples: int = 1000
    ) -> List[DetectionRateProfile]:
        """Mide tasa de detección de cada detector en datos reales.
        
        Args:
            detectors: Dict de {nombre: detector_instance}
            data: Datos para evaluar
            min_samples: Mínimo de samples para medición confiable
        """
        if len(data) < min_samples:
            raise ValueError(
                f"Datos insuficientes: {len(data)} < {min_samples}"
            )
        
        expected_rates = EnsembleCalibrator().compute_expected_rates()
        profiles = []
        
        for name, detector in detectors.items():
            try:
                # Detectar anomalías
                predictions = detector.predict(data)
                
                # Calcular tasa empírica
                anomaly_count = np.sum(predictions)
                empirical_rate = anomaly_count / len(data)
                
                profile = DetectionRateProfile(
                    detector_name=name,
                    expected_rate=expected_rates.get(name, 0.01),
                    empirical_rate=empirical_rate,
                    samples_evaluated=len(data)
                )
                profiles.append(profile)
            except Exception:
                # Detector fallido → usar tasa esperada
                profile = DetectionRateProfile(
                    detector_name=name,
                    expected_rate=expected_rates.get(name, 0.01),
                    empirical_rate=expected_rates.get(name, 0.01),
                    samples_evaluated=len(data)
                )
                profiles.append(profile)
        
        return profiles
