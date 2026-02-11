"""Explicabilidad ML — Feature importance para Taylor y ensemble.

Calcula la contribución de cada componente de la predicción al resultado
final.  Para Taylor, descompone en: valor actual, velocidad, aceleración.
Para ensemble, descompone en contribución de cada engine.

ISO 27001 A.12.4.1: Las explicaciones son parte del audit trail.
Cada predicción auditable incluye feature contributions.

Nota: No usa SHAP (requiere dependencia pesada).  Implementa
descomposición analítica directa de los términos de Taylor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureContribution:
    """Contribución de un feature a la predicción.

    Attributes:
        feature_name: Nombre del feature (``"valor_actual"``, ``"velocidad"``).
        value: Valor numérico del feature.
        contribution: Cuánto aporta al resultado final (absoluto).
        direction: ``"increase"`` | ``"decrease"`` | ``"neutral"``.
        percentage: Porcentaje de contribución al total.
    """

    feature_name: str
    value: float
    contribution: float
    direction: str
    percentage: float = 0.0


class TaylorFeatureImportance:
    """Calcula importancia de features para TaylorPredictionEngine.

    Descompone la predicción de Taylor en sus términos constituyentes:
    - f(t): Valor actual (término constante)
    - f'(t)·h: Contribución de velocidad (término lineal)
    - f''(t)·h²/2!: Contribución de aceleración (término cuadrático)
    - f'''(t)·h³/3!: Contribución de jerk (término cúbico)
    """

    def explain(
        self,
        taylor_metadata: Dict[str, object],
        predicted_value: float,
    ) -> List[FeatureContribution]:
        """Explica contribución de cada término de Taylor.

        Args:
            taylor_metadata: Metadata del TaylorPredictionEngine
                (debe contener ``"derivatives"`` y ``"order"``).
            predicted_value: Valor predicho final.

        Returns:
            Lista de ``FeatureContribution`` ordenadas por importancia.
        """
        derivatives = taylor_metadata.get("derivatives", {})
        order = int(taylor_metadata.get("order", 0))
        dt = float(taylor_metadata.get("dt", 1.0))
        horizon = int(taylor_metadata.get("horizon_steps", 1))
        h = float(horizon) * dt

        f_t = float(derivatives.get("f_t", 0.0))
        f_prime = float(derivatives.get("f_prime", 0.0))
        f_double_prime = float(derivatives.get("f_double_prime", 0.0))
        f_triple_prime = float(derivatives.get("f_triple_prime", 0.0))

        contributions: List[FeatureContribution] = []

        # Término constante
        contributions.append(FeatureContribution(
            feature_name="valor_actual",
            value=f_t,
            contribution=abs(f_t),
            direction="neutral",
        ))

        # Término lineal (velocidad)
        if order >= 1:
            linear_contrib = f_prime * h
            direction = "increase" if linear_contrib > 0 else "decrease"
            contributions.append(FeatureContribution(
                feature_name="velocidad",
                value=f_prime,
                contribution=abs(linear_contrib),
                direction=direction if abs(linear_contrib) > 1e-9 else "neutral",
            ))

        # Término cuadrático (aceleración)
        if order >= 2:
            quad_contrib = f_double_prime * (h * h) / 2.0
            direction = "increase" if quad_contrib > 0 else "decrease"
            contributions.append(FeatureContribution(
                feature_name="aceleracion",
                value=f_double_prime,
                contribution=abs(quad_contrib),
                direction=direction if abs(quad_contrib) > 1e-9 else "neutral",
            ))

        # Término cúbico (jerk)
        if order >= 3:
            cubic_contrib = f_triple_prime * (h * h * h) / 6.0
            direction = "increase" if cubic_contrib > 0 else "decrease"
            contributions.append(FeatureContribution(
                feature_name="jerk",
                value=f_triple_prime,
                contribution=abs(cubic_contrib),
                direction=direction if abs(cubic_contrib) > 1e-9 else "neutral",
            ))

        # Calcular porcentajes
        total_contrib = sum(c.contribution for c in contributions)
        if total_contrib > 1e-12:
            contributions = [
                FeatureContribution(
                    feature_name=c.feature_name,
                    value=c.value,
                    contribution=c.contribution,
                    direction=c.direction,
                    percentage=round((c.contribution / total_contrib) * 100.0, 1),
                )
                for c in contributions
            ]

        # Ordenar por contribución absoluta (descendente)
        contributions.sort(key=lambda c: c.contribution, reverse=True)

        return contributions

    def generate_explanation_text(
        self,
        contributions: List[FeatureContribution],
    ) -> str:
        """Genera explicación en lenguaje natural.

        Args:
            contributions: Lista de contribuciones.

        Returns:
            String legible por humanos.
        """
        parts: List[str] = []

        for contrib in contributions:
            if contrib.feature_name == "valor_actual":
                parts.append(f"Partiendo de {contrib.value:.2f}")
            elif contrib.feature_name == "velocidad":
                if abs(contrib.value) > 0.01:
                    verb = "aumenta" if contrib.direction == "increase" else "disminuye"
                    parts.append(
                        f"la tendencia {verb} a {abs(contrib.contribution):.3f}/paso"
                    )
            elif contrib.feature_name == "aceleracion":
                if abs(contrib.value) > 0.001:
                    verb = "acelerando" if contrib.direction == "increase" else "desacelerando"
                    parts.append(
                        f"{verb} a {abs(contrib.contribution):.4f}/paso²"
                    )
            elif contrib.feature_name == "jerk":
                if abs(contrib.value) > 0.001:
                    parts.append(
                        f"con cambio de aceleración de {abs(contrib.contribution):.5f}/paso³"
                    )

        return ", ".join(parts) + "." if parts else "Sin explicación disponible."


class CounterfactualExplainer:
    """Genera explicaciones contrafactuales básicas.

    Responde: "Si X hubiera sido Y, el resultado sería Z".

    Implementación simplificada: perturba el último valor de la serie
    y recalcula la predicción para mostrar sensibilidad.
    """

    def explain(
        self,
        values: List[float],
        predicted_value: float,
        taylor_metadata: Dict[str, object],
        perturbation_pct: float = 0.1,
    ) -> List[Dict[str, object]]:
        """Genera explicaciones contrafactuales.

        Args:
            values: Serie temporal original.
            predicted_value: Predicción original.
            taylor_metadata: Metadata de Taylor.
            perturbation_pct: Porcentaje de perturbación (default 10%).

        Returns:
            Lista de escenarios contrafactuales.
        """
        if not values:
            return []

        last_value = values[-1]
        derivs = taylor_metadata.get("derivatives", {})
        order = int(taylor_metadata.get("order", 0))
        dt = float(taylor_metadata.get("dt", 1.0))
        horizon = int(taylor_metadata.get("horizon_steps", 1))
        h = float(horizon) * dt

        perturbation = abs(last_value) * perturbation_pct
        if perturbation < 0.01:
            perturbation = 0.1

        counterfactuals: List[Dict[str, object]] = []

        # Escenario 1: valor más alto
        higher_value = last_value + perturbation
        higher_pred = self._recompute_taylor(higher_value, derivs, order, h)
        counterfactuals.append({
            "scenario": f"Si el último valor fuera {higher_value:.2f} (+{perturbation:.2f})",
            "original_prediction": round(predicted_value, 4),
            "counterfactual_prediction": round(higher_pred, 4),
            "delta": round(higher_pred - predicted_value, 4),
            "sensitivity": round(
                (higher_pred - predicted_value) / perturbation, 4
            ) if perturbation > 1e-12 else 0.0,
        })

        # Escenario 2: valor más bajo
        lower_value = last_value - perturbation
        lower_pred = self._recompute_taylor(lower_value, derivs, order, h)
        counterfactuals.append({
            "scenario": f"Si el último valor fuera {lower_value:.2f} (-{perturbation:.2f})",
            "original_prediction": round(predicted_value, 4),
            "counterfactual_prediction": round(lower_pred, 4),
            "delta": round(lower_pred - predicted_value, 4),
            "sensitivity": round(
                (lower_pred - predicted_value) / (-perturbation), 4
            ) if perturbation > 1e-12 else 0.0,
        })

        return counterfactuals

    def _recompute_taylor(
        self,
        new_f_t: float,
        derivs: Dict[str, object],
        order: int,
        h: float,
    ) -> float:
        """Recalcula Taylor con un nuevo f(t)."""
        f_prime = float(derivs.get("f_prime", 0.0))
        f_double_prime = float(derivs.get("f_double_prime", 0.0))
        f_triple_prime = float(derivs.get("f_triple_prime", 0.0))

        result = new_f_t
        if order >= 1:
            result += f_prime * h
        if order >= 2:
            result += f_double_prime * (h * h) / 2.0
        if order >= 3:
            result += f_triple_prime * (h * h * h) / 6.0

        return result
