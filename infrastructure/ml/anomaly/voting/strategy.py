"""Estrategia de voting para ensemble de anomalías.
Una responsabilidad: combinar votos en un score final.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.drift.drift_coupling import DriftNotifier
from core.ensemble.ensemble_drift_coupling import EnsembleWeightState

from ..scoring.functions import compute_consensus_confidence, weighted_vote


class VotingStrategy:

    def __init__(
        self,
        weights: dict[str, float],
        threshold: float = 0.75,
        default_weight: float = 0.1,
        use_calibrated_weights: bool = False,
        drift_coupling: Any | None = None,
    ) -> None:
        self._weights = dict(weights)
        self._threshold = threshold
        self._default_weight = default_weight
        self._use_calibrated_weights = use_calibrated_weights
        self._calibrated_weights: dict[str, float] | None = None
        self._weight_state = EnsembleWeightState()
        self._weight_state.current_weights = dict(weights)
        self._drift_coupling = drift_coupling
        if drift_coupling:
            drift_notifier = DriftNotifier()
            drift_notifier.subscribe(drift_coupling.get_listener())

    def _get_effective_weights(self) -> dict[str, float]:
        """Retorna pesos efectivos (drift coupling, calibrados u originales)."""
        if self._drift_coupling:
            return dict(self._drift_coupling.current_weights)
        if self._use_calibrated_weights and self._calibrated_weights:
            return self._calibrated_weights
        return self._weights

    def _sync_with_bayesian_tracker(self, tracker_weights: dict[str, float]) -> None:
        """Sincroniza pesos con BayesianWeightTracker."""
        if self._weight_state.should_update(tracker_weights):
            self._weight_state.update(tracker_weights)
            if self._drift_coupling:
                self._drift_coupling.weight_state.update(tracker_weights)

    def combine(self, votes: Mapping[str, float | None]) -> float:
        clean_votes = {m: v for m, v in votes.items() if v is not None}
        return weighted_vote(
            clean_votes,
            self._get_effective_weights(),
            self._default_weight,
        )

    def is_anomaly(self, score: float, votes: Mapping[str, float | None] | None = None) -> bool:
        """Determina si la puntuación combinada constituye una anomalía.

        Aplica el criterio de consenso estricto: el score final ponderado debe alcanzar
        o superar el umbral de votación. Si se proveen votos individuales,
        requiere además que exista quórum mínimo de evidencia activa.

        Args:
            score: Puntuación combinada del ensamble en [0, 1].
            votes: Mapa opcional de votos individuales emitidos por subdetector.

        Returns:
            True si el score cumple con el umbral de votación y el quórum.
        """
        if score < self._threshold:
            return False

        if votes is not None:
            effective_weights = self._get_effective_weights()
            fired_weight = sum(
                effective_weights[m] for m, v in votes.items()
                if m in effective_weights and v is not None and v > 0.0
            )
            # Quórum mínimo: al menos un 15% del peso del ensamble debe respaldar la alerta
            if fired_weight < 0.15:
                return False

        return True

    def confidence(self, votes: Mapping[str, float | None]) -> float:
        clean_votes = {m: v for m, v in votes.items() if v is not None}
        return compute_consensus_confidence(clean_votes)

    def calibrate_weights_from_data(self, detectors: dict[str, Any], calibration_data: Any) -> Any:
        """Calibra pesos del ensemble basándose en datos reales."""
        import numpy as np

        from core.ensemble.ensemble_calibrator import DetectionRateMeasurer, EnsembleCalibrator

        if not isinstance(calibration_data, np.ndarray):
            calibration_data = np.array(calibration_data)
        measurer = DetectionRateMeasurer()
        profiles = measurer.measure_rates(detectors=detectors, data=calibration_data)
        calibrator = EnsembleCalibrator()
        calibrated = calibrator.calibrate_by_detection_rate(
            raw_weights=self._weights, detection_profiles=profiles,
        )
        if not calibrated.validate():
            raise ValueError("Pesos calibrados no suman 1.0")
        self._calibrated_weights = calibrated.calibrated_weights
        return calibrated

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)
