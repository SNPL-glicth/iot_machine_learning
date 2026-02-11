"""Tests para application/use_cases/select_engine.py.

Verifica lógica de selección de motor — sin I/O, sin instanciación real.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from iot_machine_learning.application.use_cases.select_engine import (
    select_engine_for_sensor,
)


def _make_flags(**overrides) -> MagicMock:
    """Crea un mock de FeatureFlags con defaults razonables."""
    flags = MagicMock()
    flags.ML_ROLLBACK_TO_BASELINE = overrides.get("rollback", False)
    flags.ML_ENGINE_OVERRIDES = overrides.get("overrides", {})
    flags.ML_USE_TAYLOR_PREDICTOR = overrides.get("use_taylor", False)
    flags.is_sensor_in_whitelist = MagicMock(
        return_value=overrides.get("in_whitelist", False)
    )
    flags.ML_DEFAULT_ENGINE = overrides.get("default_engine", "baseline_moving_average")
    flags.ML_TAYLOR_ORDER = overrides.get("taylor_order", 2)
    flags.ML_TAYLOR_HORIZON = overrides.get("taylor_horizon", 1)
    return flags


class TestSelectEngineForSensor:
    """Tests para selección de motor por feature flags."""

    def test_panic_button_returns_baseline(self) -> None:
        flags = _make_flags(rollback=True)
        result = select_engine_for_sensor(1, flags)
        assert result["engine_name"] == "baseline_moving_average"
        assert result["kwargs"] == {}

    def test_override_per_sensor(self) -> None:
        flags = _make_flags(overrides={42: "custom_engine"})
        result = select_engine_for_sensor(42, flags)
        assert result["engine_name"] == "custom_engine"

    def test_override_not_matching_sensor(self) -> None:
        flags = _make_flags(overrides={42: "custom_engine"}, default_engine="baseline_moving_average")
        result = select_engine_for_sensor(99, flags)
        assert result["engine_name"] == "baseline_moving_average"

    def test_taylor_whitelist(self) -> None:
        flags = _make_flags(use_taylor=True, in_whitelist=True, taylor_order=3, taylor_horizon=2)
        result = select_engine_for_sensor(1, flags)
        assert result["engine_name"] == "taylor"
        assert result["kwargs"]["order"] == 3
        assert result["kwargs"]["horizon"] == 2

    def test_taylor_not_in_whitelist(self) -> None:
        flags = _make_flags(use_taylor=True, in_whitelist=False, default_engine="baseline_moving_average")
        result = select_engine_for_sensor(1, flags)
        assert result["engine_name"] == "baseline_moving_average"

    def test_default_engine(self) -> None:
        flags = _make_flags(default_engine="ensemble")
        result = select_engine_for_sensor(1, flags)
        assert result["engine_name"] == "ensemble"

    def test_priority_panic_over_override(self) -> None:
        """Panic button tiene prioridad sobre override."""
        flags = _make_flags(rollback=True, overrides={1: "taylor"})
        result = select_engine_for_sensor(1, flags)
        assert result["engine_name"] == "baseline_moving_average"

    def test_priority_override_over_whitelist(self) -> None:
        """Override tiene prioridad sobre whitelist."""
        flags = _make_flags(
            overrides={1: "custom"},
            use_taylor=True,
            in_whitelist=True,
        )
        result = select_engine_for_sensor(1, flags)
        assert result["engine_name"] == "custom"

    def test_taylor_kwargs_for_taylor_override(self) -> None:
        """Override con 'taylor' genera kwargs de Taylor."""
        flags = _make_flags(
            overrides={1: "taylor"},
            taylor_order=3,
            taylor_horizon=5,
        )
        result = select_engine_for_sensor(1, flags)
        assert result["engine_name"] == "taylor"
        assert result["kwargs"]["order"] == 3
        assert result["kwargs"]["horizon"] == 5

    def test_non_taylor_engine_empty_kwargs(self) -> None:
        """Motor no-taylor genera kwargs vacíos."""
        flags = _make_flags(default_engine="ensemble")
        result = select_engine_for_sensor(1, flags)
        assert result["kwargs"] == {}
