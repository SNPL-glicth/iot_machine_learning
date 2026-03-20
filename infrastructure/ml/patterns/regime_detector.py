"""Detección de regímenes operacionales usando clustering.

Identifica "modos" de operación de una serie temporal (idle, activo, pico, etc.)
usando K-means para segmentar la distribución de valores históricos.

Versión simplificada sin dependencia de hmmlearn.  Usa sklearn.cluster
(ya presente en el proyecto) para K-means.

Ejemplo:
- Serie con 3 regímenes: idle=25, transición=80, pico=120
- Saber el régimen actual contextualiza anomalías: 120 es normal en "pico"
  pero anómalo en "idle".

ISO 27001: Entrenamiento y predicciones loggeadas con parámetros.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from iot_machine_learning.domain.entities.pattern import OperationalRegime
from iot_machine_learning.domain.ports.pattern_detection_port import RegimeDetectionPort

logger = logging.getLogger(__name__)

# Nombres por defecto para 2, 3 y 4 regímenes (ordenados por mean)
_DEFAULT_NAMES = {
    2: ["idle", "active"],
    3: ["idle", "active", "peak"],
    4: ["idle", "active", "peak", "overload"],
}


class RegimeDetector(RegimeDetectionPort):
    """Detector de regímenes operacionales basado en K-means.

    Attributes:
        _n_regimes: Número de regímenes a detectar.
        _regimes: Lista de regímenes entrenados (ordenados por mean).
        _trained: ``True`` si el detector fue entrenado.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        *,
        random_state: int = 42,
        n_init: int = 10,
    ) -> None:
        if n_regimes < 2:
            raise ValueError(f"n_regimes debe ser >= 2, recibido {n_regimes}")

        self._n_regimes = n_regimes
        self._random_state = random_state
        self._n_init = n_init
        self._regimes: List[OperationalRegime] = []
        self._trained: bool = False

    def train(self, historical_values: List[float]) -> None:
        """Entrena detector con datos históricos usando K-means.

        Args:
            historical_values: Serie temporal de entrenamiento.

        Raises:
            ValueError: Si no hay suficientes datos.
        """
        min_points = self._n_regimes * 10
        if len(historical_values) < min_points:
            raise ValueError(
                f"Se requieren al menos {min_points} puntos, "
                f"recibidos {len(historical_values)}"
            )

        try:
            from sklearn.cluster import KMeans
            import numpy as np
        except ImportError:
            logger.warning("sklearn_not_available_regime_detection_disabled")
            self._train_fallback(historical_values)
            return

        X = np.array(historical_values).reshape(-1, 1)

        kmeans = KMeans(
            n_clusters=self._n_regimes,
            random_state=self._random_state,
            n_init=self._n_init,
        )
        labels = kmeans.fit_predict(X)

        # Construir regímenes
        regimes: List[OperationalRegime] = []
        for i in range(self._n_regimes):
            cluster_values = X[labels == i].flatten()
            if len(cluster_values) == 0:
                continue

            mean_val = float(np.mean(cluster_values))
            std_val = float(np.std(cluster_values))

            regimes.append(OperationalRegime(
                regime_id=i,
                name=f"regime_{i}",
                mean_value=mean_val,
                std_value=std_val,
                typical_duration_s=0.0,
            ))

        # Ordenar por mean_value y asignar nombres
        regimes.sort(key=lambda r: r.mean_value)
        names = _DEFAULT_NAMES.get(len(regimes), [])

        named_regimes: List[OperationalRegime] = []
        for idx, regime in enumerate(regimes):
            name = names[idx] if idx < len(names) else f"regime_{idx}"
            named_regimes.append(OperationalRegime(
                regime_id=idx,
                name=name,
                mean_value=regime.mean_value,
                std_value=regime.std_value,
                typical_duration_s=regime.typical_duration_s,
            ))

        self._regimes = named_regimes
        self._trained = True

        logger.info(
            "regime_detector_trained",
            extra={
                "n_regimes": len(self._regimes),
                "n_points": len(historical_values),
                "regimes": [
                    {"name": r.name, "mean": round(r.mean_value, 2), "std": round(r.std_value, 2)}
                    for r in self._regimes
                ],
            },
        )

    def predict_regime(self, value: float) -> OperationalRegime:
        """Predice régimen para un valor dado.

        Usa distancia normalizada (Mahalanobis simplificada) para
        encontrar el régimen más cercano.

        Args:
            value: Valor actual de la serie.

        Returns:
            ``OperationalRegime`` más cercano.

        Raises:
            RuntimeError: Si el detector no fue entrenado.
        """
        if not self._trained:
            raise RuntimeError("Detector de regímenes no entrenado")

        min_distance = float("inf")
        closest = self._regimes[0]

        for regime in self._regimes:
            normalizer = max(regime.std_value, 1e-9)
            distance = abs(value - regime.mean_value) / normalizer
            if distance < min_distance:
                min_distance = distance
                closest = regime

        return closest

    def is_trained(self) -> bool:
        """``True`` si el detector fue entrenado."""
        return self._trained

    @property
    def regimes(self) -> List[OperationalRegime]:
        """Lista de regímenes entrenados."""
        return list(self._regimes)

    def _train_fallback(self, values: List[float]) -> None:
        """Fallback sin sklearn: usa percentiles para segmentar.

        Divide la distribución en N segmentos por percentiles.
        Menos preciso que K-means pero funcional sin dependencias.
        """
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        segment_size = n // self._n_regimes

        regimes: List[OperationalRegime] = []
        names = _DEFAULT_NAMES.get(self._n_regimes, [])

        for i in range(self._n_regimes):
            start = i * segment_size
            end = (i + 1) * segment_size if i < self._n_regimes - 1 else n
            segment = sorted_vals[start:end]

            if not segment:
                continue

            mean_val = sum(segment) / len(segment)
            var_val = sum((v - mean_val) ** 2 for v in segment) / len(segment)
            std_val = math.sqrt(var_val) if var_val > 0 else 0.0

            name = names[i] if i < len(names) else f"regime_{i}"
            regimes.append(OperationalRegime(
                regime_id=i,
                name=name,
                mean_value=mean_val,
                std_value=std_val,
            ))

        self._regimes = regimes
        self._trained = True

        logger.info(
            "regime_detector_trained_fallback",
            extra={"n_regimes": len(self._regimes), "method": "percentile"},
        )
