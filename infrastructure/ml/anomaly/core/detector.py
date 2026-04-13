"""Ensemble de detectores de anomalías con voting ponderado.

Compone sub-detectores individuales (cada uno = una responsabilidad)
y una VotingStrategy desacoplada para combinar votos.

Arquitectura:
  SubDetector[] → votos individuales [0, 1]
  VotingStrategy → score final + decisión anomalía
  AnomalyNarrator → explicación legible

ISO 27001: Cada voto individual se registra en el resultado para
trazabilidad completa de la decisión.

.. versionchanged:: 2.0
    ``VotingAnomalyDetector`` now accepts optional ``sub_detectors``
    via constructor for dependency injection (MOD-2, ROB-2).
    ``create_default_detectors()`` extracts the default ensemble.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Optional

from iot_machine_learning.domain.entities.anomaly import AnomalyResult, AnomalySeverity
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from iot_machine_learning.domain.ports.anomaly_detection_port import AnomalyDetectionPort

from .config import AnomalyDetectorConfig
from .protocol import SubDetector
from ..narration import build_anomaly_explanation
from ..factory import create_default_detectors
from ..scoring import compute_z_score, TrainingStats, TemporalTrainingStats
from ..voting import build_vote_context, extract_acc_z, extract_vel_z, VotingStrategy

logger = logging.getLogger(__name__)



class VotingAnomalyDetector(AnomalyDetectionPort):
    """Ensemble de detectores de anomalías con voting adaptativo.

    Compone sub-detectores individuales y delega la decisión
    a una VotingStrategy desacoplada. Aprende pesos adaptativos
    basados en precision de cada detector.

    Attributes:
        _config: Configuración centralizada.
        _sub_detectors: Lista de sub-detectores individuales.
        _strategy: Estrategia de voting.
        _trained_flag: ``True`` si fue entrenado.
        _detector_outcomes: Historial de outcomes por detector.
        _adaptive_weights: Pesos adaptativos por detector.
        _outcome_count: Contador de outcomes para recalcular pesos.
        _series_id: Series ID para persistencia.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        voting_threshold: float = 0.5,
        weights: Optional[Dict[str, float]] = None,
        *,
        n_estimators: int = 100,
        random_state: int = 42,
        lof_max_neighbors: int = 20,
        min_training_points: int = 50,
        config: Optional[AnomalyDetectorConfig] = None,
        sub_detectors: Optional[List[SubDetector]] = None,
        series_id: Optional[str] = None,
        enable_adaptive_weights: bool = True,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = AnomalyDetectorConfig(
                contamination=contamination,
                voting_threshold=voting_threshold,
                min_training_points=min_training_points,
                n_estimators=n_estimators,
                random_state=random_state,
                lof_max_neighbors=lof_max_neighbors,
                weights=weights or AnomalyDetectorConfig().weights,
            )

        if sub_detectors is not None:
            self._sub_detectors: List[SubDetector] = list(sub_detectors)
        else:
            self._sub_detectors = create_default_detectors(self._config)

        self._series_id = series_id
        self._enable_adaptive = enable_adaptive_weights
        
        # Adaptive weights state
        self._detector_outcomes: Dict[str, Deque[bool]] = {
            d.method_name: deque(maxlen=50) for d in self._sub_detectors
        }
        self._outcome_count = 0
        
        # Try to load adaptive weights from DB
        if series_id and enable_adaptive_weights:
            loaded_weights = self._load_adaptive_weights(series_id)
            if loaded_weights:
                self._adaptive_weights = loaded_weights
                logger.info(
                    "adaptive_weights_loaded_from_db",
                    extra={"series_id": series_id, "n_detectors": len(loaded_weights)},
                )
            else:
                self._adaptive_weights = dict(self._config.weights)
        else:
            self._adaptive_weights = dict(self._config.weights)
        
        self._strategy = VotingStrategy(
            weights=self._adaptive_weights,
            threshold=self._config.voting_threshold,
        )

        self._trained_flag: bool = False

        # Backward-compatible attributes for tests that inspect internals
        self._stats: TrainingStats = TrainingStats(
            mean=0.0, std=1e-9, q1=0.0, q3=0.0, iqr=0.0
        )
        self._temporal_stats: TemporalTrainingStats = TemporalTrainingStats.empty()
        self._scaler = None

    @property
    def name(self) -> str:
        return "voting_anomaly_detector"

    def train(
        self,
        historical_values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> None:
        """Entrena todos los sub-detectores con datos históricos.
        
        Normaliza datos con RobustScaler antes de entrenar.
        Calcula contamination dinámica basada en datos históricos.

        Args:
            historical_values: Serie temporal de entrenamiento.
            timestamps: Timestamps correspondientes (opcional).

        Raises:
            ValueError: Si no hay suficientes datos.
        """
        if len(historical_values) < self._config.min_training_points:
            raise ValueError(
                f"Se requieren >= {self._config.min_training_points} puntos para entrenar, "
                f"recibidos {len(historical_values)}"
            )
        
        # Normalize data with RobustScaler (better for outliers)
        try:
            from sklearn.preprocessing import RobustScaler
            import numpy as np
            
            self._scaler = RobustScaler()
            values_array = np.array(historical_values).reshape(-1, 1)
            values_normalized = self._scaler.fit_transform(values_array).flatten().tolist()
            
            logger.debug(
                "anomaly_data_normalized",
                extra={
                    "series_id": self._series_id,
                    "n_values": len(values_normalized),
                },
            )
        except Exception as exc:
            logger.warning(
                "anomaly_normalization_failed",
                extra={"error": str(exc)},
            )
            # Fallback: use raw values
            values_normalized = historical_values
            self._scaler = None
        
        # Calculate dynamic contamination (percentile 95 of expected anomaly rate)
        # Min 0.01, max 0.2
        try:
            import numpy as np
            # Use IQR method to estimate outlier rate
            q1, q3 = np.percentile(historical_values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = sum(1 for v in historical_values if v < lower_bound or v > upper_bound)
            dynamic_contamination = max(0.01, min(0.2, outlier_count / len(historical_values)))
            
            # Update config contamination for IsolationForest and LOF
            self._config.contamination = dynamic_contamination
            
            logger.info(
                "dynamic_contamination_calculated",
                extra={
                    "series_id": self._series_id,
                    "contamination": round(dynamic_contamination, 4),
                    "outlier_count": outlier_count,
                },
            )
        except Exception as exc:
            logger.warning(
                "dynamic_contamination_failed",
                extra={"error": str(exc)},
            )

        kwargs: dict = {}
        if timestamps is not None and len(timestamps) == len(historical_values):
            kwargs["timestamps"] = timestamps

        for detector in self._sub_detectors:
            try:
                # Train with normalized values
                detector.train(values_normalized, **kwargs)
            except Exception as exc:
                logger.warning(
                    "sub_detector_training_failed",
                    extra={"detector": detector.method_name, "error": str(exc)},
                )

        self._trained_flag = True

        # Update backward-compatible stats attributes (use original values for stats)
        from ..scoring.training import compute_training_stats
        from ..scoring.temporal import compute_temporal_training_stats

        self._stats = compute_training_stats(historical_values)
        if "timestamps" in kwargs:
            self._temporal_stats = compute_temporal_training_stats(
                historical_values, kwargs["timestamps"]
            )
        else:
            self._temporal_stats = TemporalTrainingStats.empty()

        logger.info(
            "voting_detector_trained",
            extra={
                "n_points": len(historical_values),
                "mean": round(self._stats.mean, 4),
                "std": round(self._stats.std, 4),
                "temporal_trained": self._temporal_stats.has_temporal,
                "n_sub_detectors": len(self._sub_detectors),
            },
        )

    def detect(self, window: SensorWindow) -> AnomalyResult:
        """Detecta anomalías usando voting ponderado de sub-detectores.
        
        Aplica normalización si el detector fue entrenado con scaler.

        Args:
            window: Ventana temporal del sensor.

        Returns:
            ``AnomalyResult`` con score combinado y votos individuales.
        """
        if not self._trained_flag:
            raise RuntimeError("Detector no entrenado")

        if window.is_empty or window.last_value is None:
            return AnomalyResult.normal(series_id=str(window.sensor_id))

        value = window.last_value
        
        # Apply scaler transformation if trained with normalization
        if self._scaler is not None:
            try:
                import numpy as np
                value_normalized = self._scaler.transform(np.array([[value]])).flatten()[0]
            except Exception as exc:
                logger.warning(
                    "anomaly_detection_normalization_failed",
                    extra={"error": str(exc)},
                )
                value_normalized = value
        else:
            value_normalized = value

        # Build context for sub-detectors
        vote_kwargs = self._build_vote_context(window)

        # Collect votes from all sub-detectors
        votes: Dict[str, float] = {}
        for detector in self._sub_detectors:
            if not detector.is_trained:
                continue
            try:
                v = detector.vote(value_normalized, **vote_kwargs)
                if v is not None:
                    votes[detector.method_name] = v
            except Exception as exc:
                logger.debug(
                    "sub_detector_vote_failed",
                    extra={"detector": detector.method_name, "error": str(exc)},
                )

        # Combine votes via strategy
        final_score = self._strategy.combine(votes)
        is_anomaly = self._strategy.is_anomaly(final_score)
        confidence = self._strategy.confidence(votes)
        severity = AnomalySeverity.from_score(final_score)

        # Compute z-scores for narrator
        z = compute_z_score(value, self._stats.mean, self._stats.std)
        vel_z = extract_vel_z(window, self._temporal_stats)
        acc_z = extract_acc_z(window, self._temporal_stats)

        explanation = build_anomaly_explanation(
            votes, z_score=z, vel_z_score=vel_z, acc_z_score=acc_z,
        )

        logger.debug(
            "voting_detection",
            extra={
                "series_id": str(window.sensor_id),
                "value": value,
                "votes": {k: round(v, 3) for k, v in votes.items()},
                "final_score": round(final_score, 3),
                "is_anomaly": is_anomaly,
                "temporal_active": self._temporal_stats.has_temporal,
            },
        )

        return AnomalyResult(
            series_id=str(window.sensor_id),
            is_anomaly=is_anomaly,
            score=final_score,
            method_votes=votes,
            confidence=confidence,
            explanation=explanation,
            severity=severity,
        )

    def is_trained(self) -> bool:
        return self._trained_flag
    
    def record_outcome(
        self,
        detector_votes: Dict[str, float],
        was_anomaly: bool,
    ) -> None:
        """Record outcome for adaptive weight learning.
        
        Args:
            detector_votes: Votes from each detector (0-1)
            was_anomaly: True if actual anomaly occurred
        """
        if not self._enable_adaptive:
            return
        
        # Record if each detector predicted correctly
        for detector_name, vote in detector_votes.items():
            if detector_name in self._detector_outcomes:
                # Detector predicted anomaly if vote > 0.5
                predicted_anomaly = vote > 0.5
                was_correct = predicted_anomaly == was_anomaly
                self._detector_outcomes[detector_name].append(was_correct)
        
        self._outcome_count += 1
        
        # Recalculate weights every 20 outcomes
        if self._outcome_count >= 20:
            self._recalculate_adaptive_weights()
            self._outcome_count = 0
    
    def _recalculate_adaptive_weights(self) -> None:
        """Recalculate weights based on detector precision."""
        new_weights = {}
        
        for detector_name, outcomes in self._detector_outcomes.items():
            if len(outcomes) == 0:
                # No data, keep current weight
                new_weights[detector_name] = self._adaptive_weights.get(detector_name, 0.1)
            else:
                # Precision = correct / total
                precision = sum(outcomes) / len(outcomes)
                # Weight proportional to precision (min 0.05)
                new_weights[detector_name] = max(0.05, precision)
        
        # Normalize to sum to 1.0
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}
        
        self._adaptive_weights = new_weights
        
        # Update strategy with new weights
        self._strategy = VotingStrategy(
            weights=self._adaptive_weights,
            threshold=self._config.voting_threshold,
        )
        
        logger.info(
            "adaptive_weights_recalculated",
            extra={
                "series_id": self._series_id,
                "weights": {k: round(v, 4) for k, v in new_weights.items()},
            },
        )
        
        # Persist to DB
        if self._series_id:
            self._save_adaptive_weights()
    
    def _load_adaptive_weights(self, series_id: str) -> Optional[Dict[str, float]]:
        """Load adaptive weights from DB."""
        try:
            from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.anomaly_weights_repository import (
                AnomalyWeightsRepository,
            )
            repo = AnomalyWeightsRepository()
            return repo.load_weights(series_id)
        except Exception as exc:
            logger.warning(
                "adaptive_weights_load_failed",
                extra={"series_id": series_id, "error": str(exc)},
            )
            return None
    
    def _save_adaptive_weights(self) -> None:
        """Save adaptive weights to DB."""
        if not self._series_id:
            return
        
        try:
            from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.anomaly_weights_repository import (
                AnomalyWeightsRepository,
            )
            
            # Calculate precision scores
            precision_scores = {}
            for detector_name, outcomes in self._detector_outcomes.items():
                if len(outcomes) > 0:
                    precision_scores[detector_name] = sum(outcomes) / len(outcomes)
                else:
                    precision_scores[detector_name] = 0.5
            
            repo = AnomalyWeightsRepository()
            repo.save_weights(
                series_id=self._series_id,
                weights=self._adaptive_weights,
                precision_scores=precision_scores,
                n_outcomes={k: len(v) for k, v in self._detector_outcomes.items()},
            )
        except Exception as exc:
            logger.warning(
                "adaptive_weights_save_failed",
                extra={"series_id": self._series_id, "error": str(exc)},
            )

    # --- Private helpers ---

    def _build_vote_context(self, window: SensorWindow) -> dict:
        """Builds kwargs context for sub-detector vote() calls."""
        return build_vote_context(window, self._temporal_stats)
