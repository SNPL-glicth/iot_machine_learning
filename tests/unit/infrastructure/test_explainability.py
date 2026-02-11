"""Tests para TaylorFeatureImportance y CounterfactualExplainer.

Verifica:
- Descomposición correcta de términos de Taylor
- Porcentajes de contribución suman ~100%
- Generación de texto legible
- Counterfactuals con perturbación
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.explainability.feature_importance import (
    CounterfactualExplainer,
    TaylorFeatureImportance,
)


class TestTaylorFeatureImportance:
    """Tests para explicabilidad de Taylor."""

    def test_order_2_contributions(self) -> None:
        """Orden 2 debe tener 3 contribuciones: valor, velocidad, aceleración."""
        explainer = TaylorFeatureImportance()

        metadata = {
            "order": 2,
            "derivatives": {
                "f_t": 20.0,
                "f_prime": 0.5,
                "f_double_prime": 0.1,
                "f_triple_prime": 0.0,
            },
            "dt": 1.0,
            "horizon_steps": 1,
        }

        contributions = explainer.explain(metadata, predicted_value=20.55)

        assert len(contributions) == 3
        names = [c.feature_name for c in contributions]
        assert "valor_actual" in names
        assert "velocidad" in names
        assert "aceleracion" in names

    def test_percentages_sum_to_100(self) -> None:
        """Porcentajes deben sumar ~100%."""
        explainer = TaylorFeatureImportance()

        metadata = {
            "order": 2,
            "derivatives": {
                "f_t": 20.0,
                "f_prime": 1.0,
                "f_double_prime": 0.5,
                "f_triple_prime": 0.0,
            },
            "dt": 1.0,
            "horizon_steps": 1,
        }

        contributions = explainer.explain(metadata, predicted_value=21.25)
        total_pct = sum(c.percentage for c in contributions)

        assert 99.0 <= total_pct <= 101.0, f"Porcentajes suman {total_pct}%"

    def test_order_1_only_two_contributions(self) -> None:
        """Orden 1 debe tener solo valor_actual y velocidad."""
        explainer = TaylorFeatureImportance()

        metadata = {
            "order": 1,
            "derivatives": {
                "f_t": 20.0,
                "f_prime": 0.5,
                "f_double_prime": 0.0,
                "f_triple_prime": 0.0,
            },
            "dt": 1.0,
            "horizon_steps": 1,
        }

        contributions = explainer.explain(metadata, predicted_value=20.5)
        assert len(contributions) == 2

    def test_order_3_four_contributions(self) -> None:
        """Orden 3 debe tener 4 contribuciones incluyendo jerk."""
        explainer = TaylorFeatureImportance()

        metadata = {
            "order": 3,
            "derivatives": {
                "f_t": 20.0,
                "f_prime": 0.5,
                "f_double_prime": 0.1,
                "f_triple_prime": 0.01,
            },
            "dt": 1.0,
            "horizon_steps": 1,
        }

        contributions = explainer.explain(metadata, predicted_value=20.56)
        assert len(contributions) == 4
        names = [c.feature_name for c in contributions]
        assert "jerk" in names

    def test_direction_correct(self) -> None:
        """Dirección de contribución debe ser correcta."""
        explainer = TaylorFeatureImportance()

        metadata = {
            "order": 2,
            "derivatives": {
                "f_t": 20.0,
                "f_prime": 1.0,  # Positivo → increase
                "f_double_prime": -0.5,  # Negativo → decrease
                "f_triple_prime": 0.0,
            },
            "dt": 1.0,
            "horizon_steps": 1,
        }

        contributions = explainer.explain(metadata, predicted_value=20.75)

        vel = next(c for c in contributions if c.feature_name == "velocidad")
        accel = next(c for c in contributions if c.feature_name == "aceleracion")

        assert vel.direction == "increase"
        assert accel.direction == "decrease"

    def test_sorted_by_contribution(self) -> None:
        """Contribuciones deben estar ordenadas por importancia."""
        explainer = TaylorFeatureImportance()

        metadata = {
            "order": 2,
            "derivatives": {
                "f_t": 20.0,
                "f_prime": 0.5,
                "f_double_prime": 0.1,
                "f_triple_prime": 0.0,
            },
            "dt": 1.0,
            "horizon_steps": 1,
        }

        contributions = explainer.explain(metadata, predicted_value=20.55)

        for i in range(len(contributions) - 1):
            assert contributions[i].contribution >= contributions[i + 1].contribution


class TestExplanationText:
    """Tests para generación de texto legible."""

    def test_generates_text(self) -> None:
        """Debe generar texto no vacío."""
        explainer = TaylorFeatureImportance()

        metadata = {
            "order": 2,
            "derivatives": {
                "f_t": 20.0,
                "f_prime": 0.5,
                "f_double_prime": 0.1,
                "f_triple_prime": 0.0,
            },
            "dt": 1.0,
            "horizon_steps": 1,
        }

        contributions = explainer.explain(metadata, predicted_value=20.55)
        text = explainer.generate_explanation_text(contributions)

        assert len(text) > 10
        assert "20.00" in text  # Valor actual
        assert "tendencia" in text.lower() or "partiendo" in text.lower()

    def test_empty_contributions(self) -> None:
        """Sin contribuciones debe retornar texto por defecto."""
        explainer = TaylorFeatureImportance()
        text = explainer.generate_explanation_text([])
        assert "sin explicación" in text.lower() or len(text) > 0


class TestCounterfactualExplainer:
    """Tests para explicaciones contrafactuales."""

    def test_generates_two_scenarios(self) -> None:
        """Debe generar 2 escenarios (higher y lower)."""
        explainer = CounterfactualExplainer()

        values = [20.0 + i * 0.1 for i in range(20)]
        metadata = {
            "order": 2,
            "derivatives": {
                "f_t": 22.0,
                "f_prime": 0.1,
                "f_double_prime": 0.0,
                "f_triple_prime": 0.0,
            },
            "dt": 1.0,
            "horizon_steps": 1,
        }

        counterfactuals = explainer.explain(
            values=values,
            predicted_value=22.1,
            taylor_metadata=metadata,
        )

        assert len(counterfactuals) == 2
        assert "scenario" in counterfactuals[0]
        assert "delta" in counterfactuals[0]
        assert "sensitivity" in counterfactuals[0]

    def test_empty_values_returns_empty(self) -> None:
        """Sin valores debe retornar lista vacía."""
        explainer = CounterfactualExplainer()
        result = explainer.explain([], 0.0, {})
        assert result == []

    def test_sensitivity_is_finite(self) -> None:
        """Sensibilidad debe ser un número finito."""
        import math

        explainer = CounterfactualExplainer()

        values = [20.0] * 10
        metadata = {
            "order": 1,
            "derivatives": {"f_t": 20.0, "f_prime": 0.5},
            "dt": 1.0,
            "horizon_steps": 1,
        }

        counterfactuals = explainer.explain(values, 20.5, metadata)

        for cf in counterfactuals:
            assert math.isfinite(cf["sensitivity"])
