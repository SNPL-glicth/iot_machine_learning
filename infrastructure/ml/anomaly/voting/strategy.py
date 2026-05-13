"""Estrategia de voting desacoplada para ensemble de anomalías.

Una responsabilidad: combinar votos de sub-detectores en un score final.
Sin sklearn, sin I/O, sin estado de entrenamiento.
"""

from __future__ import annotations

from typing import Dict, Optional

from core.ensemble.ensemble_calibrator import EnsembleCalibrator, DetectionRateMeasurer, CalibratedWeights
from core.ensemble.ensemble_drift_coupling import EnsembleWeightState, EnsembleDriftCoupling
from core.drift.drift_coupling import DriftNotifier

from ..scoring import compute_consensus_confidence, weighted_vote


class VotingStrategy:
    """Combina votos de sub-detectores usando promedio ponderado.

    Attributes:
        _weights: Pesos por método.
        _threshold: Score > threshold → anomalía.
        _default_weight: Peso para métodos sin peso definido.
        _use_calibrated_weights: Si True, usa pesos calibrados por tasa de detección.
        _calibrator: Instance de EnsembleCalibrator.
        _calibrated_weights: Pesos calibrados (si calibración está activa).
        _weight_state: Estado de pesos con hysteresis.
        _drift_coupling: Acoplamiento drift → ensemble calibration (opcional).
    """

    def __init__(
        self,
        weights: Dict[str, float],
        threshold: float = 0.5,
        default_weight: float = 0.1,
        use_calibrated_weights: bool = False,
        drift_coupling: Optional[EnsembleDriftCoupling] = None,
    ) -> None:
        self._weights = dict(weights)
        self._threshold = threshold
        self._default_weight = default_weight
        self._use_calibrated_weights = use_calibrated_weights
        self._calibrator = EnsembleCalibrator()
        self._calibrated_weights: Optional[Dict[str, float]] = None
        self._weight_state = EnsembleWeightState()
        self._weight_state.current_weights = dict(weights)
        
        # NUEVO: Acoplamiento drift → ensemble calibration
        self._drift_coupling = drift_coupling
        if drift_coupling:
            drift_notifier = DriftNotifier()
            drift_notifier.subscribe(drift_coupling.get_listener())

    def combine(self, votes: Dict[str, float]) -> float:
        """Calcula score final a partir de votos individuales.

        Args:
            votes: Dict método → voto [0, 1].

        Returns:
            Score final en [0, 1].
        """
        effective_weights = self._get_effective_weights()
        return weighted_vote(votes, effective_weights, self._default_weight)

    def is_anomaly(self, score: float) -> bool:
        """Determina si el score indica anomalía.

        Args:
            score: Score combinado.

        Returns:
            ``True`` si score > threshold.
        """
        return score > self._threshold

    def confidence(self, votes: Dict[str, float]) -> float:
        """Calcula confianza basada en consenso entre votantes.

        Args:
            votes: Dict método → voto [0, 1].

        Returns:
            Confianza en [0.5, 1.0].
        """
        return compute_consensus_confidence(votes)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)
    
    def _get_effective_weights(self) -> Dict[str, float]:
        """Retorna pesos efectivos (calibrados o originales)."""
        # NUEVO: Usar pesos del drift coupling si está activo
        if self._drift_coupling:
            return self._drift_coupling.current_weights
        
        if self._use_calibrated_weights and self._calibrated_weights:
            return self._calibrated_weights
        return self._weights
    
    def _sync_with_bayesian_tracker(self, tracker_weights: Dict[str, float]) -> None:
        """Sincroniza pesos con BayesianWeightTracker.
        
        Args:
            tracker_weights: Pesos del BayesianWeightTracker.
        """
        # Usar pesos del tracker como prior para calibración
        if self._weight_state.should_update(tracker_weights):
            self._weight_state.update(tracker_weights)
            # Si drift coupling está activo, actualizar también
            if self._drift_coupling:
                self._drift_coupling.weight_state.update(tracker_weights)
    
    def calibrate_weights_from_data(
        self,
        detectors: Dict[str, object],
        calibration_data
    ) -> CalibratedWeights:
        """Calibra pesos del ensemble basándose en datos reales.
        
        Args:
            detectors: Dict de {nombre: detector_instance}
            calibration_data: Datos representativos para calibración
        
        Returns:
            CalibratedWeights con pesos ajustados
        """
        import numpy as np
        
        if not isinstance(calibration_data, np.ndarray):
            calibration_data = np.array(calibration_data)
        
        measurer = DetectionRateMeasurer()
        profiles = measurer.measure_rates(
            detectors=detectors,
            data=calibration_data
        )
        
        calibrated = self._calibrator.calibrate_by_detection_rate(
            raw_weights=self._weights,
            detection_profiles=profiles
        )
        
        if not calibrated.validate():
            raise ValueError("Pesos calibrados no suman 1.0")
        
        self._calibrated_weights = calibrated.calibrated_weights
        return calibrated
