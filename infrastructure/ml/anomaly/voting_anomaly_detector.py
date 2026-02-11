"""Ensemble de detectores de anomalías con voting ponderado.

REFACTORIZADO: Delega cálculos a statistical_methods.py y texto a anomaly_narrator.py.

Antes (God Object — 5 responsabilidades):
  - Cálculos Z-score, IQR, varianza (Modeling)
  - Entrenamiento sklearn IF/LOF (Infra)
  - Voting ponderado (Decisión)
  - Orquestación de 4 métodos (Orchestration)
  - Generación de texto explicativo (Narrative)

Ahora (Orchestrator — 2 responsabilidades):
  - Orquesta sub-detectores (sklearn + estadísticos)
  - Delega math a statistical_methods.py (funciones puras)
  - Delega texto a anomaly_narrator.py (narrativa pura)

Módulos extraídos:
  - statistical_methods.py (Modeling puro — compute_z_score, weighted_vote, etc.)
  - anomaly_narrator.py (Narrative puro — build_anomaly_explanation)

ISO 27001: Cada voto individual se registra en el resultado para
trazabilidad completa de la decisión.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ....domain.entities.anomaly import AnomalyResult, AnomalySeverity
from ....domain.entities.sensor_reading import SensorWindow
from ....domain.ports.anomaly_detection_port import AnomalyDetectionPort

from .anomaly_narrator import build_anomaly_explanation
from .statistical_methods import (
    TrainingStats,
    compute_consensus_confidence,
    compute_iqr_vote,
    compute_training_stats,
    compute_z_score,
    compute_z_vote,
    weighted_vote,
)

logger = logging.getLogger(__name__)

# Pesos por defecto para cada método
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "isolation_forest": 0.40,
    "z_score": 0.25,
    "iqr": 0.15,
    "local_outlier_factor": 0.20,
}


class VotingAnomalyDetector(AnomalyDetectionPort):
    """Ensemble de detectores de anomalías con voting.

    Attributes:
        _contamination: Fracción esperada de anomalías (para IF y LOF).
        _voting_threshold: Score > threshold → anomalía.
        _weights: Pesos por método.
        _trained: ``True`` si fue entrenado.
        _stats: Estadísticas de entrenamiento (TrainingStats).
        _if_model: IsolationForest entrenado (o None).
        _lof_model: LocalOutlierFactor entrenado (o None).
    """

    def __init__(
        self,
        contamination: float = 0.1,
        voting_threshold: float = 0.5,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        if not 0.0 < contamination < 0.5:
            raise ValueError(
                f"contamination debe estar en (0, 0.5), recibido {contamination}"
            )
        if not 0.0 < voting_threshold < 1.0:
            raise ValueError(
                f"voting_threshold debe estar en (0, 1), recibido {voting_threshold}"
            )

        self._contamination = contamination
        self._voting_threshold = voting_threshold
        self._weights = weights or dict(_DEFAULT_WEIGHTS)
        self._trained_flag: bool = False

        # Estadísticas (delegadas a statistical_methods)
        self._stats: TrainingStats = TrainingStats(
            mean=0.0, std=1e-9, q1=0.0, q3=0.0, iqr=0.0
        )

        # Modelos sklearn (lazy init)
        self._if_model: object = None
        self._lof_model: object = None

    @property
    def name(self) -> str:
        return "voting_anomaly_detector"

    def train(self, historical_values: List[float]) -> None:
        """Entrena todos los sub-detectores con datos históricos.

        Args:
            historical_values: Serie temporal de entrenamiento (>= 50 puntos).

        Raises:
            ValueError: Si no hay suficientes datos.
        """
        if len(historical_values) < 50:
            raise ValueError(
                f"Se requieren >= 50 puntos para entrenar, "
                f"recibidos {len(historical_values)}"
            )

        n = len(historical_values)

        # --- Estadísticas (delegado a statistical_methods) ---
        self._stats = compute_training_stats(historical_values)

        # --- IsolationForest (infra: sklearn) ---
        self._if_model = self._train_isolation_forest(historical_values, n)

        # --- LocalOutlierFactor (infra: sklearn) ---
        self._lof_model = self._train_lof(historical_values, n)

        self._trained_flag = True

        logger.info(
            "voting_detector_trained",
            extra={
                "n_points": n,
                "mean": round(self._stats.mean, 4),
                "std": round(self._stats.std, 4),
                "q1": round(self._stats.q1, 4),
                "q3": round(self._stats.q3, 4),
                "iqr": round(self._stats.iqr, 4),
                "if_available": self._if_model is not None,
                "lof_available": self._lof_model is not None,
            },
        )

    def detect(self, window: SensorWindow) -> AnomalyResult:
        """Detecta anomalías usando voting ponderado.

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
        votes: Dict[str, float] = {}

        # --- 1. Z-score (delegado a statistical_methods) ---
        z = compute_z_score(value, self._stats.mean, self._stats.std)
        votes["z_score"] = compute_z_vote(z)

        # --- 2. IQR (delegado a statistical_methods) ---
        votes["iqr"] = compute_iqr_vote(
            value, self._stats.q1, self._stats.q3, self._stats.iqr
        )

        # --- 3. IsolationForest (infra: sklearn) ---
        if_vote = self._score_isolation_forest(value)
        if if_vote is not None:
            votes["isolation_forest"] = if_vote

        # --- 4. LocalOutlierFactor (infra: sklearn) ---
        lof_vote = self._score_lof(value)
        if lof_vote is not None:
            votes["local_outlier_factor"] = lof_vote

        # --- Voting ponderado (delegado a statistical_methods) ---
        final_score = weighted_vote(votes, self._weights)
        is_anomaly = final_score > self._voting_threshold

        # Confianza (delegado a statistical_methods)
        confidence = compute_consensus_confidence(votes)

        # Severidad
        severity = AnomalySeverity.from_score(final_score)

        # Explicación (delegado a anomaly_narrator)
        explanation = build_anomaly_explanation(votes, z_score=z)

        logger.debug(
            "voting_detection",
            extra={
                "series_id": str(window.sensor_id),
                "value": value,
                "votes": {k: round(v, 3) for k, v in votes.items()},
                "final_score": round(final_score, 3),
                "is_anomaly": is_anomaly,
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

    # --- Métodos privados: sklearn wrappers ---

    def _train_isolation_forest(
        self, values: List[float], n: int
    ) -> object:
        """Entrena IsolationForest. Retorna modelo o None."""
        try:
            from sklearn.ensemble import IsolationForest
            import numpy as np

            X = np.array(values).reshape(-1, 1)
            model = IsolationForest(
                contamination=self._contamination,
                random_state=42,
                n_estimators=100,
            )
            model.fit(X)
            logger.debug("voting_if_trained", extra={"n_points": n})
            return model
        except ImportError:
            logger.warning("sklearn_not_available_if_disabled")
            return None

    def _train_lof(self, values: List[float], n: int) -> object:
        """Entrena LocalOutlierFactor. Retorna modelo o None."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            import numpy as np

            X = np.array(values).reshape(-1, 1)
            model = LocalOutlierFactor(
                n_neighbors=min(20, n // 3),
                contamination=self._contamination,
                novelty=True,
            )
            model.fit(X)
            logger.debug("voting_lof_trained", extra={"n_points": n})
            return model
        except (ImportError, Exception) as exc:
            logger.warning(
                "lof_training_failed",
                extra={"error": str(exc)},
            )
            return None

    def _score_isolation_forest(self, value: float) -> Optional[float]:
        """Score de IsolationForest para un valor. None si no disponible."""
        if self._if_model is None:
            return None
        try:
            import numpy as np
            X = np.array([[value]])
            if_score = self._if_model.decision_function(X)[0]
            return 1.0 if if_score < 0 else 0.0
        except Exception:
            return 0.0

    def _score_lof(self, value: float) -> Optional[float]:
        """Score de LOF para un valor. None si no disponible."""
        if self._lof_model is None:
            return None
        try:
            import numpy as np
            X = np.array([[value]])
            lof_score = self._lof_model.decision_function(X)[0]
            return max(0.0, min(1.0, (-lof_score - 1.0) / 2.0))
        except Exception:
            return 0.0
