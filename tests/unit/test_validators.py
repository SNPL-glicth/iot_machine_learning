"""Tests para validadores numéricos UTSAE.

Verifica:
- validate_window: NaN, Inf, vacía, min_size
- clamp_prediction: margen, edge cases, valores iguales
- safe_float: None, NaN, Inf, tipos inválidos
"""

from __future__ import annotations

import math

import pytest

from iot_machine_learning.domain.validators.numeric import (
    ValidationError,
    clamp_prediction,
    safe_float,
    validate_window,
)


class TestValidateWindow:
    """Tests para validate_window."""

    def test_valid_window(self) -> None:
        """Ventana válida no debe lanzar excepción."""
        validate_window([1.0, 2.0, 3.0], min_size=1)

    def test_empty_window_raises(self) -> None:
        """Ventana vacía debe lanzar ValidationError."""
        with pytest.raises(ValidationError, match="al menos"):
            validate_window([], min_size=1)

    def test_too_few_points_raises(self) -> None:
        """Menos puntos que min_size debe lanzar ValidationError."""
        with pytest.raises(ValidationError, match="al menos 5"):
            validate_window([1.0, 2.0], min_size=5)

    def test_nan_raises(self) -> None:
        """NaN en la ventana debe lanzar ValidationError."""
        with pytest.raises(ValidationError, match="NaN"):
            validate_window([1.0, float("nan"), 3.0])

    def test_inf_raises(self) -> None:
        """Infinity en la ventana debe lanzar ValidationError."""
        with pytest.raises(ValidationError, match="infinito"):
            validate_window([1.0, float("inf"), 3.0])

    def test_negative_inf_raises(self) -> None:
        """-Infinity en la ventana debe lanzar ValidationError."""
        with pytest.raises(ValidationError, match="infinito"):
            validate_window([1.0, float("-inf"), 3.0])

    def test_non_numeric_raises(self) -> None:
        """Valor no numérico debe lanzar ValidationError."""
        with pytest.raises(ValidationError, match="numérico"):
            validate_window([1.0, "abc", 3.0])  # type: ignore[list-item]

    def test_integers_accepted(self) -> None:
        """Enteros deben ser aceptados."""
        validate_window([1, 2, 3], min_size=1)

    def test_min_size_exact(self) -> None:
        """Exactamente min_size puntos debe pasar."""
        validate_window([1.0, 2.0, 3.0], min_size=3)


class TestClampPrediction:
    """Tests para clamp_prediction."""

    def test_within_range_no_clamp(self) -> None:
        """Valor dentro del rango no debe ser clampeado."""
        values = [10.0, 20.0, 30.0]
        clamped, was_clamped = clamp_prediction(25.0, values, margin_pct=0.3)

        assert clamped == pytest.approx(25.0)
        assert was_clamped is False

    def test_above_range_clamped(self) -> None:
        """Valor por encima del rango + margen debe ser clampeado."""
        values = [10.0, 20.0, 30.0]
        # Rango = 20, margen 30% = 6, upper = 36
        clamped, was_clamped = clamp_prediction(50.0, values, margin_pct=0.3)

        assert clamped == pytest.approx(36.0)
        assert was_clamped is True

    def test_below_range_clamped(self) -> None:
        """Valor por debajo del rango - margen debe ser clampeado."""
        values = [10.0, 20.0, 30.0]
        # Rango = 20, margen 30% = 6, lower = 4
        clamped, was_clamped = clamp_prediction(-10.0, values, margin_pct=0.3)

        assert clamped == pytest.approx(4.0)
        assert was_clamped is True

    def test_all_same_values(self) -> None:
        """Todos los valores iguales: margen basado en valor absoluto."""
        values = [20.0, 20.0, 20.0]
        # Rango = 0, margen = |20| * 0.3 = 6.0
        clamped, was_clamped = clamp_prediction(30.0, values, margin_pct=0.3)

        assert clamped == pytest.approx(26.0)
        assert was_clamped is True

    def test_all_zeros(self) -> None:
        """Todos ceros: margen mínimo fijo (0.1)."""
        values = [0.0, 0.0, 0.0]
        clamped, was_clamped = clamp_prediction(1.0, values, margin_pct=0.3)

        # Margen mínimo = 0.1, upper = 0.1
        assert clamped == pytest.approx(0.1)
        assert was_clamped is True

    def test_empty_values_raises(self) -> None:
        """Valores vacíos deben lanzar ValidationError."""
        with pytest.raises(ValidationError, match="sin valores"):
            clamp_prediction(10.0, [])

    def test_margin_zero(self) -> None:
        """Margen 0% clampea al rango + margen mínimo de seguridad (0.1)."""
        values = [10.0, 20.0, 30.0]
        clamped, was_clamped = clamp_prediction(35.0, values, margin_pct=0.0)

        # margin_pct=0 → margin=0, pero el guard mínimo es 0.1
        # upper_bound = 30.0 + 0.1 = 30.1
        assert clamped == pytest.approx(30.1)
        assert was_clamped is True


class TestSafeFloat:
    """Tests para safe_float."""

    def test_normal_float(self) -> None:
        """Float normal retorna el mismo valor."""
        assert safe_float(3.14) == pytest.approx(3.14)

    def test_integer(self) -> None:
        """Entero se convierte a float."""
        assert safe_float(42) == pytest.approx(42.0)

    def test_string_number(self) -> None:
        """String numérico se convierte a float."""
        assert safe_float("3.14") == pytest.approx(3.14)

    def test_none_returns_default(self) -> None:
        """None retorna default."""
        assert safe_float(None) == 0.0
        assert safe_float(None, default=99.0) == 99.0

    def test_nan_returns_default(self) -> None:
        """NaN retorna default."""
        assert safe_float(float("nan")) == 0.0

    def test_inf_returns_default(self) -> None:
        """Infinity retorna default."""
        assert safe_float(float("inf")) == 0.0
        assert safe_float(float("-inf")) == 0.0

    def test_invalid_string_returns_default(self) -> None:
        """String no numérico retorna default."""
        assert safe_float("abc") == 0.0

    def test_custom_default(self) -> None:
        """Default personalizado funciona."""
        assert safe_float(None, default=-1.0) == -1.0
