"""Tests para LightGBMPredictionEngine (P5).

Verifica:
1. can_handle() requiere min 80+1 puntos
2. Fallback graceful cuando lightgbm no está instalado
3. record_actual() acumula y marca modelo como stale
4. Feature builder es stateless
"""

from __future__ import annotations

import math
import random

import pytest

try:
    import lightgbm as lgb  # type: ignore[import]
except Exception:
    lgb = None  # type: ignore[misc]

from iot_machine_learning.infrastructure.ml._experimental.lightgbm import (
    LightGBMPredictionEngine,
)
from iot_machine_learning.infrastructure.ml._experimental.lightgbm.feature_builder import (
    build_feature_vector,
    build_training_matrix,
)


class TestLightGBMFeatureBuilder:
    """P5: feature_builder es stateless."""

    def test_build_feature_vector_keys(self) -> None:
        """Feature vector debe contener las claves esperadas."""
        values = list(range(20))
        feats = build_feature_vector(values)
        expected_keys = {
            "lag_1", "lag_2", "lag_3",
            "ma_3", "ma_5", "std_5",
            "delta_1", "delta_2", "trend",
            "hour_sin", "hour_cos",
        }
        assert expected_keys.issubset(feats.keys())

    def test_build_feature_vector_deterministic(self) -> None:
        """Mismos inputs → mismos outputs."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        f1 = build_feature_vector(values)
        f2 = build_feature_vector(values)
        assert f1 == f2

    def test_build_training_matrix_min_points(self) -> None:
        """Matrix vacía si insuficientes puntos."""
        values = list(range(50))
        X, names, y = build_training_matrix(values, min_points=80)
        assert X == []
        assert names == []
        assert y == []

    def test_build_training_matrix_sufficient(self) -> None:
        """Matrix no vacía con suficientes puntos."""
        values = list(range(100))
        X, names, y = build_training_matrix(values, min_points=80)
        # range(80, 100) → 20 pares (window=values[:i], y=values[i])
        assert len(X) == 20
        assert len(y) == len(X)
        assert len(names) > 0


class TestLightGBMEngineInterface:
    """P5: contrato PredictionEngine."""

    @pytest.mark.skipif(lgb is None, reason="lightgbm not installed")
    def test_can_handle_minimum(self) -> None:
        """can_handle requiere min_train_points + 1."""
        engine = LightGBMPredictionEngine(min_train_points=80)
        assert engine.can_handle(80) is False
        assert engine.can_handle(81) is True

    def test_name(self) -> None:
        """Engine name es lightgbm_regressor."""
        engine = LightGBMPredictionEngine()
        assert engine.name == "lightgbm_regressor"

    @pytest.mark.skipif(lgb is None, reason="lightgbm not installed")
    def test_insufficient_data_fallback(self) -> None:
        """Menos de 81 puntos → fallback con confidence=0.3."""
        engine = LightGBMPredictionEngine(min_train_points=80)
        values = list(range(50))
        result = engine.predict(values)
        assert result.metadata["fallback"] == "insufficient_data"
        assert 0.0 <= result.confidence <= 1.0

    def test_record_actual_does_not_crash(self) -> None:
        """record_actual debe ser seguro."""
        engine = LightGBMPredictionEngine()
        engine.record_actual(10.0, 12.0)
        assert engine._record_count == 1

    def test_record_actual_triggers_retrain(self) -> None:
        """Tras retrain_every calls, el modelo se marca como stale."""
        engine = LightGBMPredictionEngine(retrain_every=10)
        for _ in range(9):
            engine.record_actual(10.0, 10.0)
        assert engine._model is None  # nunca entrenado
        engine.record_actual(10.0, 10.0)
        # Ahora _record_count resetea y _model sigue None (no hay predict aún)
        assert engine._record_count == 0


class TestLightGBMEngineOptionalDependency:
    """P5: graceful fallback cuando lightgbm no está instalado."""

    def test_fallback_when_lightgbm_unavailable(self, monkeypatch) -> None:
        """Si lgb is None, predict retorna confidence=0.0."""
        # Mock lightgbm as unavailable
        import iot_machine_learning.infrastructure.ml._experimental.lightgbm.engine as lgb_engine

        monkeypatch.setattr(lgb_engine, "lgb", None)
        engine = LightGBMPredictionEngine()
        values = list(range(100))
        result = engine.predict(values)

        assert result.confidence == 0.0
        assert result.metadata["fallback"] == "lightgbm_not_installed"
        assert result.metadata["lightgbm_available"] is False
