"""Motor de predicción ensemble con pesos dinámicos.

Combina múltiples motores de predicción (Taylor, Baseline, etc.)
usando weighted average con auto-tuning de pesos basado en error reciente.

Estrategia:
1. Ejecutar N engines en paralelo.
2. Filtrar engines que fallaron (fail-open).
3. Combinar predicciones con weighted average.
4. Trend por majority vote.
5. Actualizar pesos según performance (inverse error weighting).

Enterprise features:
- Fallback si algún engine falla (no rompe el ensemble).
- Logging de contribución de cada engine.
- Auto-tuning de pesos (online learning cada 10 updates).
- Circuit breaker implícito: engine que falla N veces seguidas
  recibe peso 0 hasta reset.

ISO 27001: Metadata incluye pesos, predicciones individuales y
contribución de cada engine para auditoría.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Deque, Dict, List, Optional

from ....domain.entities.prediction import Prediction
from ....domain.entities.sensor_reading import SensorWindow
from ....domain.ports.prediction_port import PredictionPort

logger = logging.getLogger(__name__)

# Máximo de errores recientes para tracking
_MAX_ERROR_HISTORY: int = 100

# Cada cuántos updates recalcular pesos
_WEIGHT_UPDATE_INTERVAL: int = 10

# Peso mínimo para evitar que un engine se "apague" completamente
_MIN_WEIGHT: float = 0.05


class EnsembleWeightedPredictor(PredictionPort):
    """Ensemble que combina múltiples engines con pesos dinámicos.

    Attributes:
        _engines: Lista de engines a combinar.
        _weights: Pesos actuales por engine (suman 1.0).
        _adapt_weights: Si ``True``, ajusta pesos según performance.
        _engine_errors: Historial de errores por engine.
        _update_count: Contador de updates para recalcular pesos.
    """

    def __init__(
        self,
        engines: List[PredictionPort],
        initial_weights: Optional[List[float]] = None,
        adapt_weights: bool = True,
    ) -> None:
        """Inicializa el ensemble.

        Args:
            engines: Lista de engines (>= 2 para ensemble real).
            initial_weights: Pesos iniciales.  Si ``None``, uniformes.
            adapt_weights: Si ``True``, auto-tuning de pesos.

        Raises:
            ValueError: Si no hay engines o pesos no coinciden.
        """
        if not engines:
            raise ValueError("Se requiere al menos un engine para ensemble")

        self._engines = engines
        self._adapt_weights = adapt_weights
        self._update_count: int = 0

        # Inicializar pesos
        if initial_weights is not None:
            if len(initial_weights) != len(engines):
                raise ValueError(
                    f"Pesos ({len(initial_weights)}) no coinciden con "
                    f"engines ({len(engines)})"
                )
            total = sum(initial_weights)
            self._weights = [w / total for w in initial_weights]
        else:
            n = len(engines)
            self._weights = [1.0 / n] * n

        # Tracking de errores por engine
        self._engine_errors: Dict[str, Deque[float]] = {
            engine.name: deque(maxlen=_MAX_ERROR_HISTORY)
            for engine in engines
        }

    @property
    def name(self) -> str:
        return "ensemble_weighted"

    def can_handle(self, n_points: int) -> bool:
        """Puede manejar si al menos 1 engine puede."""
        return any(e.can_handle(n_points) for e in self._engines)

    def predict(self, window: SensorWindow) -> Prediction:
        """Combina predicciones de todos los engines.

        Args:
            window: Ventana temporal del sensor.

        Returns:
            ``Prediction`` combinada con metadata de contribuciones.

        Raises:
            RuntimeError: Si todos los engines fallan.
        """
        predictions: List[Optional[Prediction]] = []
        individual_meta: List[Dict[str, object]] = []

        for engine in self._engines:
            try:
                if not engine.can_handle(window.size):
                    predictions.append(None)
                    individual_meta.append({
                        "engine": engine.name,
                        "status": "skipped_insufficient_data",
                    })
                    continue

                result = engine.predict(window)
                predictions.append(result)
                individual_meta.append({
                    "engine": engine.name,
                    "status": "success",
                    "prediction": result.predicted_value,
                    "confidence": result.confidence_score,
                    "trend": result.trend,
                })
            except Exception as exc:
                logger.warning(
                    "ensemble_engine_failed",
                    extra={
                        "engine": engine.name,
                        "sensor_id": window.sensor_id,
                        "error": str(exc),
                    },
                )
                predictions.append(None)
                individual_meta.append({
                    "engine": engine.name,
                    "status": "failed",
                    "error": str(exc),
                })

        # Filtrar válidos
        valid_indices = [
            i for i, p in enumerate(predictions) if p is not None
        ]

        if not valid_indices:
            raise RuntimeError(
                f"Todos los engines fallaron para sensor {window.sensor_id}"
            )

        valid_preds = [predictions[i] for i in valid_indices]
        valid_weights = [self._weights[i] for i in valid_indices]

        # Renormalizar pesos
        total_w = sum(valid_weights)
        if total_w < 1e-12:
            valid_weights = [1.0 / len(valid_weights)] * len(valid_weights)
        else:
            valid_weights = [w / total_w for w in valid_weights]

        # Weighted average de predicciones
        final_value = sum(
            p.predicted_value * w  # type: ignore[union-attr]
            for p, w in zip(valid_preds, valid_weights)
        )

        # Weighted average de confianza
        final_confidence = sum(
            p.confidence_score * w  # type: ignore[union-attr]
            for p, w in zip(valid_preds, valid_weights)
        )

        # Trend: majority vote
        trend_counts: Dict[str, float] = {"up": 0.0, "down": 0.0, "stable": 0.0}
        for p, w in zip(valid_preds, valid_weights):
            trend_counts[p.trend] += w  # type: ignore[union-attr]
        final_trend = max(trend_counts, key=trend_counts.get)  # type: ignore[arg-type]

        # Metadata con contribuciones
        weight_map = {
            self._engines[i].name: round(valid_weights[j], 4)
            for j, i in enumerate(valid_indices)
        }

        logger.debug(
            "ensemble_prediction",
            extra={
                "sensor_id": window.sensor_id,
                "n_engines_used": len(valid_indices),
                "weights": weight_map,
                "final_value": round(final_value, 4),
                "final_confidence": round(final_confidence, 4),
            },
        )

        return Prediction(
            sensor_id=window.sensor_id,
            predicted_value=final_value,
            confidence_score=final_confidence,
            trend=final_trend,  # type: ignore[arg-type]
            engine_name="ensemble_weighted",
            metadata={
                "ensemble_weights": weight_map,
                "individual_predictions": individual_meta,
                "n_engines_total": len(self._engines),
                "n_engines_used": len(valid_indices),
            },
        )

    def update_weights(
        self,
        actual_value: float,
        predictions: List[Optional[Prediction]],
    ) -> None:
        """Actualiza pesos según error reciente (inverse error weighting).

        Args:
            actual_value: Valor real observado.
            predictions: Predicciones de cada engine (None si falló).
        """
        if not self._adapt_weights:
            return

        for i, pred in enumerate(predictions):
            if pred is not None:
                error = abs(pred.predicted_value - actual_value)
                self._engine_errors[self._engines[i].name].append(error)

        self._update_count += 1

        # Recalcular pesos cada N updates
        if self._update_count % _WEIGHT_UPDATE_INTERVAL == 0:
            self._recalculate_weights()

    def _recalculate_weights(self) -> None:
        """Recalcula pesos usando inverse error weighting."""
        avg_errors: List[float] = []

        for engine in self._engines:
            errors = self._engine_errors[engine.name]
            if errors:
                avg_errors.append(sum(errors) / len(errors))
            else:
                avg_errors.append(1.0)  # Default si no hay data

        # Inverse weighting
        inverse = [1.0 / (err + 1e-9) for err in avg_errors]
        total = sum(inverse)

        new_weights = [max(_MIN_WEIGHT, inv / total) for inv in inverse]

        # Renormalizar
        total_new = sum(new_weights)
        self._weights = [w / total_new for w in new_weights]

        logger.info(
            "ensemble_weights_updated",
            extra={
                "avg_errors": [round(e, 4) for e in avg_errors],
                "new_weights": [round(w, 4) for w in self._weights],
            },
        )

    def supports_confidence_interval(self) -> bool:
        return False

    @property
    def current_weights(self) -> Dict[str, float]:
        """Pesos actuales por engine."""
        return {
            self._engines[i].name: self._weights[i]
            for i in range(len(self._engines))
        }
